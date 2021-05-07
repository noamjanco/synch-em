from os.path import isfile, join
from os import listdir
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed


def run_training_experiments(task, n_jobs, seed_range, SNR_range, label, *args):
    print('Started training experiments: %s'%label)
    path_learning = '2d_data_for_prior/'
    if not os.path.exists(path_learning):
        os.mkdir(path_learning)
    path_output = path_learning+label
    if not os.path.exists(path_output):
        os.mkdir(path_output)
        # TODO: put a flag in task to use samples from prior instead of dataset in make_data
        results = Parallel(n_jobs=n_jobs)(delayed(task)(seed, image_idx, SNR, *args) for SNR in SNR_range for seed in seed_range for image_idx in [seed])
        results = pd.concat(results, ignore_index=True)
        results.to_pickle(path_output + '/combined.pickle')
    print('Finished running training experiments')
    return path_output +'/'


def learn_prior(SNR, path, Ndir=360):
    filename_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    results = pd.concat([pd.read_pickle(filename) for filename in filename_list])
    df = results[(results.SNR == SNR)]
    if len(df) == 0:
        raise NameError('Prior data set is empty for the required configuration of (L,b,sigma)')
    mean_dist = np.zeros((Ndir,1))
    for i in range(len(df)):
        q = df.iloc[i].rho
        I = np.argmax(q)
        q = np.roll(q, -(I - int(Ndir / 2)))
        mean_dist += q
    mean_dist = mean_dist / len(df)
    return mean_dist