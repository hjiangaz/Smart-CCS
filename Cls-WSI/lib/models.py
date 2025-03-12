import torch
from dinov2.models import build_model_CCS
from .mil_model import ABMIL, MeanMIL, MaxMIL


def load_dinov2_CCS(ckpt = None):
    device = torch.device("cuda")
    print('loading checkpoint:', ckpt)
    dinov2_vits14, _ = build_model_CCS(device, 1, ckpt)
    return dinov2_vits14

def load_dinov2():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2_vits14.eval()
    dinov2_vits14 = dinov2_vits14.cuda()
    return dinov2_vits14

def load_backbone(model="dinov2", ckpt = None):
    if model == "dinov2":
        backbone = load_dinov2()
    elif model == "CCS":
        backbone = load_dinov2_CCS(ckpt)
    return backbone

def load_classifier(classifier_mode, in_dim, n_classes, dropout, act, selection_K, checkpoint=None):
    if classifier_mode == "ABMIL":
        classifier = ABMIL(in_dim, n_classes)
    elif classifier_mode == "MeanMIL":
        classifier = MeanMIL(in_dim, n_classes)
    elif classifier_mode == "MaxMIL":
        classifier = MaxMIL(in_dim, n_classes)
    else:
        raise ValueError("Not supported classifier")
    
    if checkpoint is not None:
        classifier.load_state_dict(torch.load(checkpoint))
    return classifier
