import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
# Classification head controls (only for TranAD_C)
parser.add_argument('--train_clf', 
					action='store_true', 
					help="train classification head on top of TranAD_C using error features")
parser.add_argument('--use_clf', 
					action='store_true', 
					help="use classification head for evaluation instead of POT")
parser.add_argument('--clf_version',
					type=str,
					default='v1',
					choices=['v1', 'v2', 'v3'],
					help="classifier head version: v1=basic, v2=multi-scale, v3=attention")
parser.add_argument('--freeze_backbone',
					action='store_true',
					help="freeze TranAD backbone and only train classifier")
parser.add_argument('--num_epochs',
					type=int,
					default=None,
					help="number of training epochs (overrides default)")
parser.add_argument('--clf_epochs', 
					type=int, 
					default=5,
					help="number of epochs to train the classification head")
parser.add_argument('--clf_lr', 
					type=float, 
					default=1e-3,
					help="learning rate for classification head")

# Ablation study arguments for TranAD_QC
parser.add_argument('--activation_type',
					type=str,
					default='learnable_cosine',
					choices=['learnable_cosine', 'relu', 'leaky_relu', 'fixed_cosine'],
					help="activation function type: learnable_cosine=default, relu=complex ReLU, leaky_relu=complex LeakyReLU, fixed_cosine=non-learnable cosine")
parser.add_argument('--loss_type',
					type=str,
					default='complex_aware',
					choices=['complex_aware', 'mse'],
					help="loss function type: complex_aware=custom complex loss, mse=standard complex MSE")
parser.add_argument('--focus_type',
					type=str,
					default='complex_aware',
					choices=['complex_aware', 'mse'],
					help="focus score computation: complex_aware=angle+amplitude blend, mse=squared error")

# Visualization arguments
parser.add_argument('--visualize',
					action='store_true',
					help="generate classification result visualizations")
parser.add_argument('--vis_n_folds',
					type=int,
					default=5,
					help="number of folds for cross-validation visualization")
parser.add_argument('--vis_save_dir',
					type=str,
					default='plots/classification',
					help="directory to save visualization plots")
parser.add_argument('--experiment_name',
					type=str,
					default=None,
					help="custom name for the experiment (used in visualization, checkpoints, and logs)")

args = parser.parse_args()