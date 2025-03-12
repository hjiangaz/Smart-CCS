import os.path as osp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import os
from lib.models import load_backbone
from lib.datasets import get_dataloader
from config import get_args
from lib.models import load_backbone, load_classifier
import pandas as pd


def load_model(model_name, ckpt):
    """Load the backbone model based on the given model name."""
    return load_backbone(model=model_name, ckpt=ckpt)

def create_output_folder(folder):
    """Create the output folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)

def extract_features(backbone, imgs):
    """Extract features from a single image using the backbone model."""
    imgs = imgs.cuda()
    with torch.no_grad():
        embeddings = backbone(imgs)
    return embeddings.cpu()

def classify_features(classifier, features):
    """Classify the extracted features using the classifier."""
    classifier.eval()
    with torch.no_grad():
        features = features.cuda()
        outputs = classifier(features)
        Y_prob = outputs[1]
        # print(outputs)
        predicted_classes = torch.argmax(Y_prob, dim=1)
    return predicted_classes.cpu().numpy()

def inference(args):
    """
    Main function to extract features from images and classify them.

    Args:
        args: An object containing command-line arguments and configuration settings.
    """
    # Print all arguments and their values
    for arg_name, arg_value in vars(args).items():
        print(arg_name, '=', arg_value)

    # Create output folder and load backbone model
    create_output_folder(args.output_folder)
    backbone = load_model(args.backbone_model, args.checkpoint)

    # Prepare classifier
    classifier = load_classifier(
        classifier_mode=args.classifier,
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        dropout=args.dropout,
        act=args.act,
        selection_K=args.selection_K
    )
    classifier.load_state_dict(torch.load(args.classifier_ckpt))  # Load the trained classifier model
    classifier.cuda()

    # Get the data loader for inference
    feat_dataloader = get_dataloader(
        batch_size=1,  # Assuming each sample is processed one at a time
        shuffle=True,
        num_workers=4,
        selection_K=args.selection_K,
        mode="infer",
        model=args.backbone_model,
        dataset=args.dataset,
        infer_set=args.infer_set
    )
    results = []
    # Process each sample individually
    for imgs, _, slide_path in feat_dataloader:  # Assuming dataloader returns (imgs, slide_path)
        slide_name = os.path.basename(slide_path[0]).split('.')[0]  # Get slide name without extension
        print("Inferring:", slide_name)

        # Extract features for the current sample
        features = extract_features(backbone, imgs[0])

        # Classify the extracted features
        predictions = classify_features(classifier, features)

        # Print prediction in the desired format
        class_names = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC']  # Map class indices to actual class names
        detailed_class = class_names[predictions[0]]  # Get the predicted class for the single sample
        coarse_class = 'NOR' if detailed_class == 'NILM' else 'ABN'

        # Append results for Excel
        results.append([slide_name, coarse_class])

        # Print output based on args.only_screening
        if args.only_screening:
            print(f'Slide ID: {slide_name}, Screening Results: {coarse_class}')  # Output only slide name and coarse class
        else:
            print(f'Slide ID: {slide_name}, Subtype: {detailed_class}, Screening Results: {coarse_class}')  # Output fine-grained and coarse class

    # Save results to Excel
    output_file_name = os.path.splitext(os.path.basename(args.infer_set))[0] + "_Smart-CCS_results.xlsx"
    output_file_path = os.path.join(args.output_folder, output_file_name)

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(results, columns=["Slide ID", "Screening Results"])
    df.to_excel(output_file_path, index=False)

    print(f'Results saved to {output_file_path}')

if __name__ == "__main__":
    args = get_args()
    inference(args)