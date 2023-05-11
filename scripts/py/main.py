# ~~~~~~~~~~~~~~ Script overview ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
#' Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour 
#' using accelerometry and machine learning: challenges and opportunities.
#' 
#' github.com/...
#'
#' Purpose: 
#'      This script runs deep learning models. A 1-second sliding window is 
#'      applied with 50% overlap.
#'
#' Instructions: installation with conda recommended
#'      1. Create conda environment with required packages (see requirements.txt):
#'              conda create -n dclstm python=3.10 pytorch pytorch-cuda=11.7 pandas scikit-learn matplotlib -c pytorch -c nvidia
#'      2. Activate environment:
#'              conda activate dclstm
#'      3. Install pyprojroot
#'              conda install -c conda-forge pyprojroot
#'      3. Run code:
#'          python scripts/py/main.py --batchnorm
#'      
#' Notes:
#'      Recommended running with GPU. If memory allocation errors, then try
#'      using a smaller batch size (edit parameters.batch_size in this script)
#'      This code is modified from Bock et al.'s (2021) implementation 
#'      available at: https://github.com/mariusbock/dl-for-har
#'
#'      Bock et al.'s original script has more functionality, such as customising
#'      all parameters through command line arguments, but this was greatly 
#'      simplified. Several other changes were made to the pipeline such as 
#'      implementing windowing prior to cross-validation, changing log  
#'      output paths, and several other minor changes. Some parts of the 
#'      codebase were retained in this repository but are unused.
#'      
#'      Built with python 3.10, but likely works python 3.7+ (not tested)
#'
#' Date created:
#'      May 2, 2023
#'      
# ~~~~~~~~~~~~~~ Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
import pyprojroot
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
from data_processing.load_dataset import load_dataset
from data_processing.sliding_window import apply_sliding_window
from model.LSIO_cv import LSIO_cv
from model.train import predict
from misc.logging import Logger
from misc.torchutils import seed_torch

# ~~~~~~~~~~~~~~ dclstm run code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
def main(parameters):
    # ---- Init and logging ----

    # apply the chosen random seed to all relevant parts
    seed_torch(parameters.seed)

    # parameters used to calculate runtime
    start = time.time()
    log_datetime = time.strftime('%Y-%m-%d_%H.%M.%OS2')

    # saves logs to a file (standard output redirected)
    base_path = pyprojroot.find_root(pyprojroot.has_dir(".git"))
    out_dir = os.path.join(base_path, 'outputs', 'nn-results', log_datetime)
    parameters.base_path = str(base_path)
    parameters.out_dir = out_dir
    if parameters.logging:
        sys.stdout = Logger(os.path.join(out_dir, 'log.txt'))

    print(f"output path: {out_dir}")
    print('Applied settings: ')
    print(parameters.getparams())

    # ---- Data Loading ----
    print('Loading data...')
    # Load dataset
    if not parameters.dummy_data:
        data = load_dataset(parameters)
    else:
        # Simulate a random dummy data for testing & development
        import math
        dummy_size = 10000
        dummy_data = {
            'fold_id' : np.repeat(parameters.folds_to_run, math.ceil(dummy_size/len(parameters.folds_to_run)))[0:dummy_size],
            'segment_id' : np.repeat(np.arange(0, 174), math.ceil(dummy_size/174))[0:dummy_size],
            'datetime' : '2021-01-01 10:00:00.100',
            'accX' : np.random.normal(size = dummy_size),
            'accY' : np.random.normal(size = dummy_size),
            'accZ' : np.random.normal(size = dummy_size),
            'independent': np.random.choice([0,1], size = dummy_size, replace = True),
            'satellite': np.random.choice([0,1], size = dummy_size, replace = True), 
            'behaviour': np.random.choice(np.arange(0,parameters.nb_classes), size = dummy_size//200, replace = True)
        }
        # add some consecutive behaviours i.e. to get non-transition windows
        dummy_data['behaviour'] =  np.repeat(dummy_data['behaviour'], 210)[0:dummy_size]
        dummy_data = pd.DataFrame(dummy_data)

        # dummy_data = dummy_data.astype(np.float32)
        data = dummy_data

    # ---- Windowing ----
    print("Windowing data...")
    windowed_data = apply_sliding_window(
        data, parameters)
    
    # save windowed dataset for debugging/checking, if requested
    if parameters.save_windowed_data:
        X = windowed_data['win_X']
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        Xdf = pd.DataFrame(data = X)
        Xdf = Xdf.rename(columns={0: "x", 1: "y", 2: "z", 3:"indep", 4: "sat"})
        n_per_window = parameters.sampling_rate * parameters.sw_length
        Xdf['win_id'] = np.repeat(range(1, len(windowed_data['win_y'])+1), n_per_window)
        Xdf['win_segment_id'] = np.repeat(windowed_data['win_segment_id'] , n_per_window)
        Xdf['win_fold'] = np.repeat(windowed_data['win_fold'] , n_per_window)
        Xdf['win_start'] = np.repeat(windowed_data['win_start'] , n_per_window)
        Xdf['beh'] = np.repeat(windowed_data['win_y'], n_per_window)
        invert_class_dict = {v: k for k, v in parameters.classes.items()}
        Xdf['beh'] = Xdf['beh'].replace(invert_class_dict)
        Xdf.to_csv(os.path.join(out_dir, f'dclstm_windows_{log_datetime}.csv'))

    # ---- Training and cross-validation ----
    # LSIO splitting uses the(Leave One Individual Out) 
    trained_net = LSIO_cv(windowed_data, parameters.folds_to_run, parameters)

    # calculate time data creation took
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nFinal time elapsed: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    print(f"output path: {out_dir}")



# ~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----
# Simplified from original code
class ParamHolder(object):
    pass
    def getparams(self):
        return vars(parameters)

parameters = ParamHolder()
# sliding window params
parameters.sw_length = 1
parameters.sw_unit = 'seconds'
parameters.sw_overlap = 50
parameters.sampling_rate = 50
parameters.nb_channels = 3

# neural network architecture
parameters.network = 'deepconvlstm'
parameters.no_lstm = False
parameters.nb_units_lstm = 256
parameters.nb_layers_lstm = 1
parameters.conv_block_type = 'normal'
parameters.nb_conv_blocks = 2
parameters.nb_filters = 64
parameters.filter_width = 11
parameters.dilation = 1     # i.e. regular convolution, no dilation
parameters.drop_prob = 0.5
parameters.pooling = False
parameters.batch_norm = False
parameters.reduce_layer = False
parameters.pool_type = 'max'
parameters.pool_kernel_width = 2
parameters.reduce_layer_output = 8

# training
parameters.seed = 1
parameters.valid_epoch = 'best'
parameters.batch_size = 512
parameters.epochs = 100
parameters.optimizer = 'adam'
parameters.learning_rate = 1e-4
parameters.weight_decay = 1e-6
parameters.weights_init = 'xavier_normal'
parameters.smoothing = 0.0
parameters.gpu = "cuda" if torch.cuda.is_available() else "cpu"
parameters.weighted = True
parameters.shuffling = True
parameters.adj_lr = False
parameters.lr_step = 10
parameters.lr_decay = 0.9
parameters.early_stopping = True
parameters.es_patience = 15

#outputs
parameters.name = 'ruff'
parameters.logging = True
parameters.print_counts = False
parameters.verbose = False
parameters.print_freq = 200
parameters.save_test_predictions = True
parameters.save_checkpoints = True
parameters.save_analysis = True
parameters.save_gradient_plot = True
parameters.save_windowed_data = False

# Should transitions be excluded from training folds in cross-validation 
# (transitions always remain in validation folds)
parameters.train_excluding_transitions = False

parameters.classes = {
        'aggressive posturing': 0, 
        'being mounted': 1, 
        'copulation attempt': 2, 
        'dynamic squatting': 3, 
        'flying': 4,
        'foraging or drinking': 5,
        'mounting male': 6,
        'other': 7,
        'preening': 8,
        'resting': 9, 
        'static squatting': 10,
        'vigilance': 11,
        'walking or running': 12}
parameters.nb_classes = len(parameters.classes)

# which ruff_id goes into which fold
parameters.cv_fold_assignments = {
    '1301': 0,
    '7-04-105': 1,
    '1361': 1,
    'G20-0059-B6.5': 1,
    'G20-0529-B6.5': 2,
    '952': 3,
    '1331': 4,
    'G20-0071-B6.5': 4,
    '1372': 5,
    '1326': 6,
    '1399': 7,
    '1681': 7,
    'G20-0055-B6.5': 7,
    '1368': 8,
    '1333': 9
}
# If cross-validation gets interruped and needs to be resumed, then list the folds to be
# run here. Otherwise run all 10 folds by default. (might be buggy)
parameters.folds_to_run = [0,1,2,3,4,5,6,7,8,9]
# Use a dummy dataset which is small in size and fast to run, for testing
parameters.dummy_data = False

### Parse command line arguments for customized analysis
# note: prints won't get captured by Logger() defined in the main function.
if "--notrans" in sys.argv:
    print("OPTION: excluding transitions")
    parameters.train_excluding_transitions = True
if "--2layer" in sys.argv:
    print("OPTION: running with 2 layers")
    parameters.nb_layers_lstm = 2
if "--512units" in sys.argv:
    print("OPTION: running with 512 units per LSTM layer")
    parameters.nb_units_lstm = 512
if "--noweights" in sys.argv:
    print("OPTION: running without weights")
    parameters.weighted = False
if "--batchnorm" in sys.argv:
    print("OPTION: running with batch normalization")
    parameters.batch_norm = True
if "--savewindoweddata" in sys.argv:
    print("OPTION: saving windowed data file")
    parameters.save_windowed_data = True
if "--test" in sys.argv:
    print("OPTION: excluding transitions")
    parameters.dummy_data = True

# ~~~~~~~~~~~~~~ Run models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----

if __name__ == '__main__':
    main(parameters)
else:
    # for interactive use & debugging
    main(parameters)
