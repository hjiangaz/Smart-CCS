import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from einops import rearrange, repeat
import opt_einsum as oe
import numpy as np
import math

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

class Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout > 0:
            self.module.append(nn.Dropout(dropout))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=6):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, only_A=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        if only_A:
            return A
        return A, x

class ABMIL(nn.Module):
    def __init__(self, in_dim, n_classes=2, dropout=0.25):
        super(ABMIL, self).__init__()
        fc_size = [in_dim, 256]
        self.n_classes = n_classes
        self.path_attn_head = Attn_Net_Gated(L=fc_size[0], D=fc_size[1], dropout=dropout, n_classes=1)
        self.classifiers = nn.Linear(fc_size[0], n_classes)

    def forward(self, wsi_h):
        wsi_trans = wsi_h.squeeze(0)
        # Attention Pooling
        path = self.path_attn_head(wsi_trans, only_A=True)
        ori_path = path.view(1, -1)
        path = F.softmax(ori_path, dim=1)
        M = torch.mm(path, wsi_trans)  # all instance
        attn = path.detach().cpu().numpy()
        # ---->predict (cox head)
        logits = self.classifiers(M)

        return logits, attn
    
class ABMIL_not_Gated(nn.Module):
    def __init__(self, n_classes=6, dropout=0.25):
        super(ABMIL_not_Gated, self).__init__()
        fc_size = [1024, 128]
        self.n_classes = n_classes
        self.path_attn_head = Attn_Net(L=fc_size[0], D=fc_size[1], dropout=dropout, n_classes=1)
        self.classifiers = nn.Linear(fc_size[0], n_classes)
    
    def forward(self, wsi_h):
        wsi_trans = wsi_h.squeeze(0)
        # Attention Pooling
        path, wsi_trans = self.path_attn_head(wsi_trans)
        ori_path = path.view(1, -1)
        path = F.softmax(ori_path, dim=1)
        M = torch.mm(path, wsi_trans)
        attn = path.detach().cpu().numpy()
        # ---->predict (cox head)
        logits = self.classifiers(M)

        return logits, attn

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x)).contiguous()

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class MeanMIL(nn.Module):
    def __init__(self, in_dim=1024, n_classes=1, dropout=False, act='relu', survival = False):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(in_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)
        self.survival = survival

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[0] > 1:
            raise RuntimeError('Batch size must be 1, current batch size is:{}'.format(x.shape[0]))
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x[0]
        logits = self.head(x)
        logits = torch.mean(logits, dim=0, keepdim=True)
        
        '''
        Survival Layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(device)

    
class Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super(Feat_Projecter, self).__init__()
        self.projecter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x = [B, N, C] or [N, C]
        if len(x.shape) == 3:
            L1, L2, L3 = x.shape[0], x.shape[1], x.shape[2]
            x = x.view(-1, L3)
            x = self.projecter(x) # for BatchNorm
            x = x.view(L1, L2, -1)
        else:
            x = self.projecter(x)
        return x    

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats, **kwargs):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
    
    def forward(self, x, **kwargs):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, hid_size, output_class, dropout_v=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, hid_size)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, hid_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=hid_size)
        
    def forward(self, feats, c, **kwargs): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 


class DSMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_emb=1024, num_cls=6, use_feat_proj=True, drop_rate=0.5):
        super(DSMIL, self).__init__()
        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None
        self.i_classifier = FCLayer(in_size=dim_in, out_size=num_cls)
        self.b_classifier = BClassifier(dim_in, dim_emb, num_cls, dropout_v=drop_rate)
        
    def forward(self, X, **kwargs):
        X = X.unsqueeze(0)
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        X = X.squeeze(0) # to [N, C] for input to i and b classifier
        feats, classes = self.i_classifier(X)
        prediction_bag, A, B = self.b_classifier(feats, classes) # bag = [1, C], A = [N, C]
        
        max_prediction, _ = torch.max(classes, 0)
        logits = 0.5 * (prediction_bag + max_prediction) # logits = [1, C]

        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            # average over class heads
            attn = A.detach()
            attn = attn.mean(dim=1).unsqueeze(0)
            return logits, attn
        
        return logits
    
# This code is taken from the original S4 repository https://github.com/HazyResearch/state-spaces

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L)  # (H L)
        u_f = torch.fft.rfft(u.to(torch.float32), n=2*L)  # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y
    

class MaxMIL(nn.Module):
    def __init__(self, in_dim=1024, n_classes=1, dropout=False, act='relu', survival = False):
        super(MaxMIL, self).__init__()

        head = [nn.Linear(in_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)
        self.survival = survival

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[0] > 1:
            raise RuntimeError('Batch size must be 1, current batch size is:{}'.format(x.shape[0]))
        if len(x.shape) == 3 and x.shape[0] == 1:
            x = x[0]
        logits = self.head(x)
        logits, _ = torch.max(logits, dim=0, keepdim=True)   
        
        '''
        Survival Layer
        '''
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(device)    