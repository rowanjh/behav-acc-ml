##################################################
# All functions related to validating a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

from logging import exception
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data_processing.sliding_window import apply_sliding_window
from misc.osutils import mkdir_if_missing
from model.DeepConvLSTM import DeepConvLSTM
from model.evaluate import evaluate_LSIOcv_scores
from model.train import train
import json

def LSIO_cv(data, folds_to_run, args):
    """
    Method to apply cross-participant cross-validation (also known as leave-one-subject-out cross-validation).
    :param data: numpy array
        Data used for applying cross-validation
    :param args: dict
        Args object containing all relevant hyperparameters and settings
    :param log_date: string
        Date information needed for saving
    :param log_timestamp: string
        Timestamp information needed for saving
    :return pytorch model
        Trained network
    """

    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')

    # Setup
    all_eval_output = None
    orig_lr = args.learning_rate
    log_dir = os.path.join(args.out_dir)
    # Save analysis parameteres
    with open(os.path.join(log_dir, "analysis-parameters.txt"), 'w') as f:
        f.write(json.dumps(args.getparams()))

    # arrays that will to store cv results
    cv_scores = np.empty((4, args.nb_classes, len(args.folds_to_run) + 1))
    cv_scores[:] = np.nan
    train_val_gap = np.zeros((4, len(args.folds_to_run) + 1))

    # # for debug 
    # folds_to_run = [2, 25]
    # LOSO CV loop
    for fold in folds_to_run:
        print('\n VALIDATING FOR CV FOLD {0} OF {1}'.format(int(fold) + 1, int(np.max(data['win_fold'])) + 1))
        print(f"\n TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # If excluding transitions from train fold
        if args.train_excluding_transitions:
            X_train = data['win_X'][(data['win_fold'] != fold) & np.logical_not(data['transition']), :, :]
            y_train = data['win_y'][(data['win_fold'] != fold) & np.logical_not(data['transition'])]
        else:
            X_train = data['win_X'][data['win_fold'] != fold, :, :]
            y_train = data['win_y'][data['win_fold'] != fold]

        X_val = data['win_X'][data['win_fold'] == fold, :, :]
        y_val = data['win_y'][data['win_fold'] == fold]
        # if there is no data for this fold, there was an error
        if X_val.shape[0] == 0:
            raise Exception(f"No data for fold {fold}")

        args.learning_rate = orig_lr

        # Initialize network, optimizer, loss function, and scheduler.
        net = DeepConvLSTM(config=vars(args))
        opt = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, args.lr_step, args.lr_decay)

        # Run training loop (run for N epochs with this CV fold as the validation set)
        net, checkpoint, val_output, train_output = train(
            X_train, y_train, X_val, y_val, network=net, optimizer=opt, 
            loss=loss, lr_scheduler=scheduler, config=vars(args))

        ### When this CV fold is done, save outputs (model, best checkpoint, val output, train output)
        fold_outdir = os.path.join(log_dir, f"fold{fold}")
        mkdir_if_missing(fold_outdir)

        # Validation/training outputs
        val_preds = pd.DataFrame({
            'segment_id': data['win_segment_id'][data['win_fold'] == fold],
            'window_start_time': data['win_start'][data['win_fold'] == fold],
            'pred_class': val_output[:,0],
            'actual_class': val_output[:,1]
            })
        train_preds = pd.DataFrame({
            'pred_class': train_output[:,0],
            'actual_class': train_output[:,1]
            })
        if args.train_excluding_transitions:
            train_preds['segment_id'] = data['win_segment_id'][(data['win_fold'] != fold) & np.logical_not(data['transition'])]
            train_preds['window_start_time'] = data['win_start'][(data['win_fold'] != fold) & np.logical_not(data['transition'])]
        else:
            train_preds['segment_id'] = data['win_segment_id'][data['win_fold'] != fold ]
            train_preds['window_start_time'] = data['win_start'][data['win_fold'] != fold]


        class_label_inverter = {v: k for k, v in args.classes.items()}
        val_preds = pd.concat([val_preds, val_preds[['pred_class','actual_class']].replace(class_label_inverter)], axis = 1)
        train_preds = pd.concat([train_preds, train_preds[['pred_class','actual_class']].replace(class_label_inverter)], axis = 1)
        val_preds.to_csv(os.path.join(fold_outdir, "preds_validation.csv"), index = False)
        train_preds.to_csv(os.path.join(fold_outdir, "preds_train.csv"), index = False)

        # Checkpoint
        c_name = os.path.join(fold_outdir, "checkpoint_best_fold{}.pth".format(str(fold)))
        torch.save(checkpoint, c_name)

        # Add results for this full to accumulator that is used to evaluate all folds in the end
        if all_eval_output is None:
            all_eval_output = val_output
        else:
            all_eval_output = np.concatenate((all_eval_output, val_output), axis=0)

        # Get and print metrics for this fold
        # allow for folds with missing classes by only considering the classes that are present
        # val[:,0] = preds. val[:,1] = groundtruth
        classes_present = np.unique([val_output[:, 1], val_output[:,0]]) 
        cv_scores[0, classes_present, int(fold)] = jaccard_score(val_output[:, 1], val_output[:, 0], average=None)
        cv_scores[1, classes_present, int(fold)] = precision_score(val_output[:, 1], val_output[:, 0], average=None)
        cv_scores[2, classes_present, int(fold)] = recall_score(val_output[:, 1], val_output[:, 0], average=None)
        cv_scores[3, classes_present, int(fold)] = f1_score(val_output[:, 1], val_output[:, 0], average=None)

        # difference in train-validation metrics
        train_val_gap[0, int(fold)] = jaccard_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                     jaccard_score(val_output[:, 1], val_output[:, 0], average='macro')
        train_val_gap[1, int(fold)] = precision_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                     precision_score(val_output[:, 1], val_output[:, 0], average='macro')
        train_val_gap[2, int(fold)] = recall_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                     recall_score(val_output[:, 1], val_output[:, 0], average='macro')
        train_val_gap[3, int(fold)] = f1_score(train_output[:, 1], train_output[:, 0], average='macro') - \
                                     f1_score(val_output[:, 1], val_output[:, 0], average='macro')

        print("SUBJECT {0} VALIDATION RESULTS: ".format(int(fold) + 1))
        print("Accuracy: {0}".format(jaccard_score(val_output[:, 1], val_output[:, 0], average=None)))
        print("Precision: {0}".format(precision_score(val_output[:, 1], val_output[:, 0], average=None)))
        print("Recall: {0}".format(recall_score(val_output[:, 1], val_output[:, 0], average=None)))
        print("F1: {0}".format(f1_score(val_output[:, 1], val_output[:, 0], average=None)))

    # After all folds are run, save results to csv file
    if args.save_analysis:
        mkdir_if_missing(log_dir)
        cv_score_acc = pd.DataFrame(cv_scores[0, :, :], index=None)
        cv_score_acc.index = args.classes.keys()
        cv_score_prec = pd.DataFrame(cv_scores[1, :, :], index=None)
        cv_score_prec.index = args.classes.keys()
        cv_score_rec = pd.DataFrame(cv_scores[2, :, :], index=None)
        cv_score_rec.index = args.classes.keys()
        cv_score_f1 = pd.DataFrame(cv_scores[3, :, :], index=None)
        cv_score_f1.index = args.classes.keys()
        tv_gap = pd.DataFrame(train_val_gap, index=None)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        # Save results to csv
        cv_score_acc.to_csv(os.path.join(log_dir, 'cv_scores_acc.csv'))
        cv_score_prec.to_csv(os.path.join(log_dir, 'cv_scores_prec.csv'))
        cv_score_rec.to_csv(os.path.join(log_dir, 'cv_scores_rec.csv'))
        cv_score_f1.to_csv(os.path.join(log_dir, 'cv_scores_f1.csv'))
        tv_gap.to_csv(os.path.join(log_dir, 'train_val_gap.csv'))

    # Evaluate whole analysis
    evaluate_LSIOcv_scores(participant_scores=cv_scores,
                                gen_gap_scores=train_val_gap,
                                input_cm=all_eval_output,
                                class_names=args.classes.keys(),
                                nb_subjects=len(args.folds_to_run),
                                filepath=log_dir,
                                filename='LSIO_CV',
                                args=args
                                )

    return net