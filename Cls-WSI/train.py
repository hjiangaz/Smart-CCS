import os.path as osp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
import os
from lib.models import load_backbone, load_classifier
from lib.datasets import get_dataloader
from lib.utils import AverageMeter,cal_acc_org
from config import get_args
import time



def main(args):
    """
    Main function to train and evaluate the model.

    Args:
        args: An object containing command-line arguments and configuration settings.
    """
    # Print all arguments and their values
    for arg_name, arg_value in vars(args).items():
        print(arg_name, '=', arg_value)

    # Prepare model saving path
    model_save_root = osp.join(args.output_folder, args.classifier + '.pth')
    os.makedirs(args.output_folder, exist_ok=True)

    # Load backbone model if not in feature extraction mode
    # backbone = load_backbone(model=args.backbone_model) 

    # Load classifier
    classifier = load_classifier(
        classifier_mode=args.classifier,
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        dropout=args.dropout,
        act=args.act,
        selection_K=args.selection_K
    )
    classifier.cuda()

    # Set up optimizer and loss function
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=10e-5)
    ce_loss = nn.CrossEntropyLoss()

    # Prepare data loaders for training and validation
    train_dataloader = get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        mode="train",
        dataset=args.dataset,
        train_set=args.train_set,
        selection_K=args.selection_K
    )

    valid_dataloader = get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        mode="val",
        dataset=args.dataset,
        val_set=args.val_set,
        selection_K=args.selection_K
    )

    # Training loop
    best_acc = 0
    start_time = time.time()

    for epoch in range(args.epoch):
        train_losses, train_acces = train_one_epoch(train_dataloader, classifier, optimizer, ce_loss, args.batch_size, args.train_mode)
        valid_losses, valid_acces = validate(valid_dataloader, classifier, ce_loss, args.batch_size, args.train_mode)

        # Calculate remaining time and print epoch summary
        avg_time_per_epoch = (time.time() - start_time) / (epoch + 1)
        remaining_time = (avg_time_per_epoch * (args.epoch - epoch - 1)) / 3600
        print_epoch_summary(epoch, args.epoch, remaining_time, train_losses, train_acces, valid_losses, valid_acces)

        # Save best model based on validation accuracy
        if valid_acces.avg > best_acc:
            best_acc = valid_acces.avg
            torch.save(classifier.state_dict(), model_save_root)
            print(f"Save best model at epoch {epoch + 1}.")

    # Testing phase
    test_model(classifier, model_save_root, args.batch_size, args)

def train_one_epoch(train_dataloader, classifier, optimizer, ce_loss, bs, train_mode):
    """
    Train the model for one epoch.

    Args:
        train_dataloader: DataLoader for the training dataset.
        classifier: The model classifier.
        optimizer: The optimizer for training.
        ce_loss: Cross-entropy loss function.
        bs: Batch size.
        backbone: Backbone model for feature extraction.
        train_mode: Mode of training ('img', 'slide', etc.).

    Returns:
        Tuple[AverageMeter, AverageMeter]: Average losses and accuracies for the epoch.
    """
    classifier.train()
    train_losses = AverageMeter()
    train_acces = AverageMeter()

    for feats, labels, path in train_dataloader:
        input, labels = feats.cuda(), labels.cuda()

        # Extract features based on the training mode
        # input = extract_features(imgs, backbone, train_mode, bs)

        # Forward pass
        # print(input.shape)
        outputs = classifier(input)
        Y_prob = outputs[0]

        # Compute loss
        loss = ce_loss(Y_prob, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        train_losses.update(loss.item(), bs)
        acc = cal_acc_org(Y_prob, labels)
        train_acces.update(acc, bs)

    return train_losses, train_acces

def validate(valid_dataloader, classifier, ce_loss, bs, train_mode):
    """
    Validate the model on the validation dataset.

    Args:
        valid_dataloader: DataLoader for the validation dataset.
        classifier: The model classifier.
        ce_loss: Cross-entropy loss function.
        bs: Batch size.
        backbone: Backbone model for feature extraction.
        train_mode: Mode of training ('img', 'slide', etc.).

    Returns:
        Tuple[AverageMeter, AverageMeter]: Average losses and accuracies for the validation.
    """
    classifier.eval()
    valid_losses = AverageMeter()
    valid_acces = AverageMeter()

    with torch.no_grad():
        for feats, labels, path in valid_dataloader:
            input, labels = feats.cuda(), labels.cuda()

            # # Extract features based on the training mode
            # input = extract_features(imgs, backbone, train_mode, bs)

            # Forward pass
            outputs = classifier(input)
            Y_prob = outputs[0]

            # Compute loss
            loss = ce_loss(Y_prob, labels)
            valid_losses.update(loss.item(), bs)
            acc = cal_acc_org(Y_prob, labels)
            valid_acces.update(acc, bs)

    return valid_losses, valid_acces

def extract_features(imgs, backbone, train_mode, bs):
    """
    Extract features from images.

    Args:
        imgs: Input images.
        backbone: Backbone model for feature extraction.
        train_mode: Mode of training ('img', 'slide', etc.).
        bs: Batch size.

    Returns:
        Tensor: Extracted features.
    """
    features_list = []
    imgs = imgs.squeeze(dim=0)
    batched_imgs = torch.split(imgs, bs, dim=0)
    for batch_imgs in batched_imgs:
        with torch.no_grad():
            embeddings = backbone(batch_imgs)
        features_list.append(embeddings)
    return torch.cat(features_list, dim=0)

def print_epoch_summary(epoch, total_epochs, remaining_time, train_losses, train_acces, valid_losses, valid_acces):
    """
    Print a summary of the epoch's training and validation results.

    Args:
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        remaining_time (float): Estimated remaining time in hours.
        train_losses: AverageMeter for training losses.
        train_acces: AverageMeter for training accuracies.
        valid_losses: AverageMeter for validation losses.
        valid_acces: AverageMeter for validation accuracies.
    """
    print("Epoch: {}/{}  Remaining Time: {:.2f} hours | train loss: {:.4f} | train acc: {:.1f}% | valid loss: {:.4f} | valid acc: {:.1f}%".format(
        epoch + 1, total_epochs, remaining_time, train_losses.avg, train_acces.avg * 100, valid_losses.avg, valid_acces.avg * 100))

def test_model(classifier, model_save_root, bs, args):
    """
    Test the model on the test dataset.

    Args:
        classifier: The model classifier.
        model_save_root: Path to the saved model.
        bs: Batch size.
        args: Command-line arguments containing dataset paths.
    """
    classifier.load_state_dict(torch.load(model_save_root))
    test_dataloader = get_dataloader(
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        mode="test",
        # model=args.backbone_model,
        dataset=args.dataset,
        test_set=args.test_set,
        selection_K=args.selection_K
    )

    test_losses = AverageMeter()
    test_acces = AverageMeter()
    classifier.eval()

    with torch.no_grad():
        for feats, labels, path in test_dataloader:
            input, labels = feats.cuda(), labels.cuda()

            # Extract features based on the training mode
            # input = extract_features(imgs, classifier, args.train_mode, bs)

            # Forward pass
            outputs = classifier(input)
            Y_prob = outputs[0]

            # Compute loss
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(Y_prob, labels)
            test_losses.update(loss.item(), bs)
            acc = cal_acc_org(Y_prob, labels)
            test_acces.update(acc, bs)

    print(f"Test loss: {test_losses.avg:.4f} | test acc: {test_acces.avg * 100:.1f}")


if __name__ == "__main__":
    args = get_args()
    main(args)