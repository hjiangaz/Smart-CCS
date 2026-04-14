import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Model training and evaluation parameters")

    # Model configuration
    parser.add_argument("--backbone_model", 
                        default='CCS', 
                        choices=["dinov2", "CCS"], 
                        type=str, 
                        help="Choose the backbone model to use.")

    parser.add_argument("--classifier", 
                        default="MeanMIL", 
                        choices=["ABMIL", "MeanMIL", "MaxMIL"], 
                        type=str, 
                        help="Select the classifier type.")

    parser.add_argument("--n_classes", 
                        default=7, 
                        type=int, 
                        help="Number of output classes.")

    parser.add_argument("--dropout", 
                        default=None, 
                        type=int, 
                        help="Dropout rate for regularization.")

    parser.add_argument("--in_dim", 
                        default=1024, 
                        type=int, 
                        help="Input dimension for the model.")

    parser.add_argument("--act", 
                        default="relu", 
                        type=str, 
                        help="Activation function to use.")

    parser.add_argument("--lr", 
                        default=1e-4, 
                        type=float, 
                        help="Learning rate for the optimizer.")

    parser.add_argument("--batch_size", 
                        default=1, 
                        type=int, 
                        help="Batch size for training.")

    parser.add_argument("--epoch", 
                        default=50, 
                        type=int, 
                        help="Number of epochs for training.")

    # Training configuration
    parser.add_argument("--train_mode", 
                        default="feat", 
                        choices=["img", "slide", "feat"], 
                        type=str, 
                        help="Mode of training: image, slide, or feature.")

    parser.add_argument("--selection_K", 
                        default=100, 
                        type=int, 
                        help="Number of selections for training.")

    # Dataset configuration
    parser.add_argument("--dataset", 
                        default=None, 
                        choices=["CCS_JSON_TOP", "CCS_feat_TOP", "Cervix"], 
                        type=str, 
                        help="Choose the dataset to use.")

    parser.add_argument("--train_set", 
                        default=None, 
                        type=str, 
                        help="Path to the training set.")

    parser.add_argument("--test_set", 
                        default=None, 
                        type=str, 
                        help="Path to the test set.")

    parser.add_argument("--val_set", 
                        default=None, 
                        type=str, 
                        help="Path to the validation set.")

    parser.add_argument("--infer_set", 
                        default=None, 
                        type=str, 
                        help="Path to the inference set.")

    parser.add_argument("--output_folder", 
                        default=None, 
                        type=str, 
                        help="Folder to save output results.")

    # Checkpoint configuration
    parser.add_argument("--checkpoint", 
                        default=None, 
                        type=str, 
                        help="Path to the model checkpoint for evaluation or resuming training.")
    parser.add_argument("--classifier_ckpt", 
                        default=None, 
                        type=str, 
                        help="Path to the model checkpoint for evaluation or resuming training.")
    parser.add_argument("--only_screening", 
                        default=False, 
                        type=str, 
                        help="Path to the model checkpoint for evaluation or resuming training.")

    args = parser.parse_args()
    return args