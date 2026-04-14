import os.path as osp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import os
from lib.models import load_backbone
from lib.datasets import get_dataloader
from config import get_args

def load_model(model_name, ckpt):
    """Load the backbone model based on the given model name."""
    return load_backbone(model=model_name, ckpt=ckpt)

def create_output_folder(folder):
    """Create the output folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)

def extract_features(backbone, dataloader, output_folder, batch_size):
    """Extract features from images using the backbone model."""
    for imgs, labels, slide_path in dataloader:
        name = os.path.basename(slide_path[0])
        path = os.path.join(output_folder, os.path.splitext(name)[0] + '.pt')

        if os.path.exists(path):
            continue

        print("Inferring slide:", name)
        imgs = imgs.cuda()
        labels = labels.cuda()

        features_list = []
        imgs = imgs.squeeze(dim=0)
        batched_imgs = torch.split(imgs, batch_size, dim=0)

        for batch_imgs in batched_imgs:
            with torch.no_grad():
                embeddings = backbone(batch_imgs)
            embeddings = embeddings.cpu()
            features_list.append(embeddings)

        features = torch.cat(features_list, dim=0)
        torch.save(features, path)

def extract(args):
    """
    Main function to extract features from images and save them to output files.

    Args:
        args: An object containing command-line arguments and configuration settings.
    """
    # Print all arguments and their values
    for arg_name, arg_value in vars(args).items():
        print(arg_name, '=', arg_value)

    # Create output folder and load model
    create_output_folder(args.output_folder)
    backbone = load_model(args.backbone_model, args.checkpoint)

    # Get the data loader for inference
    feat_dataloader = get_dataloader(
        batch_size=1,
        shuffle=True,
        num_workers=4,
        selection_K=args.selection_K,
        mode="infer",
        model=args.backbone_model,
        dataset=args.dataset,
        infer_set=args.infer_set
    )

    # Extract features from the images
    extract_features(backbone, feat_dataloader, args.output_folder, args.batch_size)


if __name__ == "__main__":
    args = get_args()
    extract(args)