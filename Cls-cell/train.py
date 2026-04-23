import os

import torch
import torch.nn as nn
import torch.optim as optim

from backbone import load_backbone
from config import get_args
from dataset import build_dataloader
from model import Classifier
from utils import AverageMeter, accuracy, set_seed


def run_epoch(backbone, classifier, dataloader, optimizer, criterion, device, training):
    if training:
        backbone.train()
        classifier.train()
    else:
        backbone.eval()
        classifier.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            features = backbone(images)
            probabilities, _ = classifier(features)
            loss = criterion(torch.log(probabilities.clamp(min=1e-8)), labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(probabilities, labels), batch_size)

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(path, backbone, classifier, args):
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, backbone, classifier, device):
    checkpoint = torch.load(path, map_location=device)
    backbone.load_state_dict(checkpoint["backbone"])
    classifier.load_state_dict(checkpoint["classifier"])


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader = build_dataloader(args.train_set, args.batch_size, True, args.num_workers)
    val_loader = build_dataloader(args.val_set, args.batch_size, False, args.num_workers)
    test_loader = build_dataloader(args.test_set, args.batch_size, False, args.num_workers)

    backbone, embed_dim = load_backbone(args.backbone_model, args.checkpoint)
    classifier = Classifier(args.in_dim or embed_dim, args.n_classes)
    if args.in_dim != embed_dim:
        raise ValueError(f"Expected in_dim {embed_dim} for {args.backbone_model}, got {args.in_dim}.")

    backbone = backbone.to(device)
    classifier = classifier.to(device)

    optimizer = optim.Adam(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.NLLLoss()

    best_path = os.path.join(args.output_dir, "best_model.pth")
    best_acc = -1.0

    for epoch in range(args.epoch):
        train_loss, train_acc = run_epoch(backbone, classifier, train_loader, optimizer, criterion, device, True)
        val_loss, val_acc = run_epoch(backbone, classifier, val_loader, optimizer, criterion, device, False)

        print(
            f"Epoch {epoch + 1}/{args.epoch} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, backbone, classifier, args)
            print(f"Saved best checkpoint to {best_path}")

    load_checkpoint(best_path, backbone, classifier, device)
    test_loss, test_acc = run_epoch(backbone, classifier, test_loader, optimizer, criterion, device, False)
    print(f"Test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main(get_args())
