"""
Test-Time Adaptation with Prototype Alignment for Smart-CCS.

Algorithm:
  1. Build class-wise prototype bank from retrospective data D_retro.
  2. For each test WSI, initialize student (Ms) and teacher (Mt) from pretrained CCS model.
  3. Compute loss: prototype alignment loss,consistency loss.
  4. Return ensemble prediction from Ms and Mt.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



def ccs_update_ema(teacher: nn.Module, student: nn.Module, momentum: float = 0.999) -> nn.Module:
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)
    return teacher



class CCSProjectionHead(nn.Module):

    def __init__(self, in_dim: int = 1024, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) → (..., proj_dim)"""
        return self.proj(x)



class CCSPrototypeAlignment:
    """Test-Time Adaptation with Prototype Alignment for Smart-CCS."""

    def __init__(
        self,
        classifier: nn.Module,
        proto_bank: dict,
        proj_dim: int = 128,
        in_dim: int = 1024,
        temperature: float = 0.07,
        ema_momentum: float = 0.999,
        lr: float = 1e-5,
        n_classes: int = 7,
        device: str = "cuda",
    ):
        self.device = device
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.n_classes = n_classes

        self.classifier_s = classifier.to(device)

        self.classifier_t = copy.deepcopy(classifier).to(device)
        for p in self.classifier_t.parameters():
            p.requires_grad_(False)

        self.proj_head = CCSProjectionHead(in_dim=in_dim, proj_dim=proj_dim).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.classifier_s.parameters()) + list(self.proj_head.parameters()),
            lr=lr,
        )

        self.proto_bank = {c: v.to(device) for c, v in proto_bank.items() if v is not None}


    def _pool_and_project(self, cell_feats: torch.Tensor) -> torch.Tensor:

        pooled = cell_feats.mean(dim=0, keepdim=True)     
        z = self.proj_head(pooled)                   
        return F.normalize(z, p=2, dim=1)

    def _get_proto_embeddings(self) -> torch.Tensor:
        parts = []
        with torch.no_grad():
            for c in range(self.n_classes):
                if c in self.proto_bank:
                    pz = F.normalize(self.proj_head(self.proto_bank[c]), p=2, dim=1)
                    parts.append(pz)
        return torch.cat(parts, dim=0) if parts else None


    def prototype_alignment_loss(
        self,
        z: torch.Tensor,
        z_aug: torch.Tensor,
        proto_embs: torch.Tensor,
    ) -> torch.Tensor:

        m = z.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)

        for i in range(m):
            anchor   = z[i : i + 1]        # (1, proj_dim)
            positive = z_aug[i : i + 1]    # (1, proj_dim)
            negs = (
                [z_aug[j : j + 1] for j in range(m) if j != i]
                + [z[j : j + 1] for j in range(m) if j != i]
            )
            if proto_embs is not None:
                negs.append(proto_embs)     # (W, proj_dim)

            keys  = torch.cat([positive] + negs, dim=0) if negs else positive
            sim   = torch.mm(anchor, keys.T) / self.temperature    # (1, 1+N)
            label = torch.zeros(1, dtype=torch.long, device=self.device)
            total_loss = total_loss + F.cross_entropy(sim, label)

        return total_loss / m

    def consistency_loss(
        self,
        logits_s: torch.Tensor,
        logits_t: torch.Tensor,
    ) -> torch.Tensor:
        probs_t    = F.softmax(logits_t.detach(), dim=1)
        log_probs_s = F.log_softmax(logits_s, dim=1)
        return -(probs_t * log_probs_s).sum(dim=1).mean()


    @torch.enable_grad()
    def adapt_and_predict(
        self,
        cell_feats: torch.Tensor,
        cell_feats_aug: torch.Tensor,
    ) -> torch.Tensor:
        """Single-WSI TTA step: adapt student, update teacher, return ensemble probs."""
        probs_list = self.adapt_batch_and_predict([cell_feats], [cell_feats_aug])
        return probs_list[0]

    @torch.enable_grad()
    def adapt_batch_and_predict(
        self,
        batch_cell_feats: list,
        batch_cell_feats_aug: list,
    ) -> list:
       
        self.optimizer.zero_grad()
        proto_embs = self._get_proto_embeddings()

        self.classifier_s.train()
        self.classifier_t.eval()

        z_list, z_aug_list = [], []
        logits_s_list,     logits_t_list     = [], []
        logits_s_aug_list, logits_t_aug_list = [], []
        feats_list = []

        for cell_feats, cell_feats_aug in zip(batch_cell_feats, batch_cell_feats_aug):
            feats     = cell_feats.to(self.device)
            feats_aug = cell_feats_aug.to(self.device)
            feats_list.append(feats)

            logits_s,     *_ = self.classifier_s(feats)
            logits_s_aug, *_ = self.classifier_s(feats_aug)

            with torch.no_grad():
                logits_t,     *_ = self.classifier_t(feats)
                logits_t_aug, *_ = self.classifier_t(feats_aug)

            logits_s_list.append(logits_s)
            logits_t_list.append(logits_t)
            logits_s_aug_list.append(logits_s_aug)
            logits_t_aug_list.append(logits_t_aug)

            z_list.append(self._pool_and_project(feats))
            z_aug_list.append(self._pool_and_project(feats_aug))

        z     = torch.cat(z_list,     dim=0)  
        z_aug = torch.cat(z_aug_list, dim=0) 

        logits_s_cat     = torch.cat(logits_s_list,     dim=0)  
        logits_t_cat     = torch.cat(logits_t_list,     dim=0)
        logits_s_aug_cat = torch.cat(logits_s_aug_list, dim=0)
        logits_t_aug_cat = torch.cat(logits_t_aug_list, dim=0)

        loss_align = self.prototype_alignment_loss(z, z_aug, proto_embs)
        loss_cons  = 0.5 * self.consistency_loss(logits_s_cat,     logits_t_cat) \
                   + 0.5 * self.consistency_loss(logits_s_aug_cat, logits_t_aug_cat)
        loss = loss_align + loss_cons

        loss.backward()
        self.optimizer.step()

        ccs_update_ema(self.classifier_t, self.classifier_s, self.ema_momentum)

        self.classifier_s.eval()
        probs_list = []
        with torch.no_grad():
            for feats in feats_list:
                logits_s_final, *_ = self.classifier_s(feats)
                logits_t_final, *_ = self.classifier_t(feats)
                probs_s = F.softmax(logits_s_final, dim=1)
                probs_t = F.softmax(logits_t_final, dim=1)
                probs_list.append((probs_s + probs_t) * 0.5) 

        return probs_list
