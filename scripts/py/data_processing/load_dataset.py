##################################################
# All functions related to preprocessing and loading data
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Alexander Hölzemann
# Email: alexander.hoelzemann(at)uni-siegen.de
##################################################
# Modified by Rowan Jacques-Hamilton rjacques(at)orn.mpg.de

import pandas as pd
import numpy as np
import glob
import re
import os

pd.options.mode.chained_assignment = None

def load_dataset(parameters):
    """
    Main function to load data
    :return: numpy float arrays, int, list of strings, int, boolean
        features, labels, number of classes, class names, sampling rate and boolean has_null
    """
    # Read all csv files
    all_files = glob.glob(os.path.join(parameters.base_path, 'data', 'clean', 
                                       'data-segments', '*.csv'))
    recording_info = pd.read_csv(os.path.join(parameters.base_path, 'data', 'clean', 
                                       'recording_info.csv'))

    # All files need a unique identifier: obs id.
    li = []
    for i in range(len(all_files)):
        df = pd.read_csv(all_files[i], header = 0)
        df['segment_id'] = re.search("(?<=s)[0-9]+(?=.csv)", all_files[i]).group(0) # add segment id
        li.append(df)
    data = pd.concat(li, axis=0, ignore_index=True)
    # Remove rows with NA for behaviour (which exist due to buffer before and after segments)
    data = data.dropna(subset = ['behaviour'])

    # Add ruff id to get fold assignments
    data = data.merge(recording_info[['recording_id', 'ruff_id']], 
                      left_on='recording_id', right_on = 'recording_id')
    # Subset to required columns
    data = data[['ruff_id','segment_id','datetime','accX','accY','accZ','behaviour']] 

    # Convert ruff_id labels to fold ID
    data = data.replace({
        'ruff_id': parameters.cv_fold_assignments,
        'behaviour': parameters.classes
    })
    data.rename(columns={'ruff_id':'fold_id'}, inplace=True)
    # Prep labels
    data['behaviour'] = data['behaviour'].astype(int)

    print("Full dataset with size: | {0} | ".format(data.shape))

    return data

