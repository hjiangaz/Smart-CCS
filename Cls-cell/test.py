import torch
import torch.nn as nn

from backbone import load_backbone
from config import get_args
from dataset import build_dataloader
from model import Classifier
from utils import AverageMeter, accuracy


def evaluate(backbone, classifier, dataloader, device):
    criterion = nn.NLLLoss()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    backbone.eval()
    classifier.eval()

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            features = backbone(images)
            probabilities, _ = classifier(features)
            loss = criterion(torch.log(probabilities.clamp(min=1e-8)), labels)

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy(probabilities, labels), batch_size)

    return loss_meter.avg, acc_meter.avg


def main(args):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    backbone, embed_dim = load_backbone(args.backbone_model, args.checkpoint)
    classifier = Classifier(args.in_dim or embed_dim, args.n_classes)
    if args.in_dim != embed_dim:
        raise ValueError(f"Expected in_dim {embed_dim} for {args.backbone_model}, got {args.in_dim}.")

    checkpoint = torch.load(args.classifier_ckpt, map_location=device)
    backbone.load_state_dict(checkpoint["backbone"])
    classifier.load_state_dict(checkpoint["classifier"])

    backbone = backbone.to(device)
    classifier = classifier.to(device)

    dataloader = build_dataloader(args.test_set, args.batch_size, False, args.num_workers)
    test_loss, test_acc = evaluate(backbone, classifier, dataloader, device)
    print(f"Test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main(get_args())
