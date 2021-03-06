from os.path import isfile, join
from os import listdir
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from scipy.special import erf


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


def pmf_approx_tm_1d(x,sigma):
    L = x.shape[0]
    Rxx = np.real(np.fft.ifft(np.fft.fft(x,axis=0) *(np.fft.fft(x,axis=0)).conj(),axis=0))
    sigma_c = sigma * np.sqrt(np.sum(x ** 2))
    dx = 1e-1
    pmf = np.zeros((L,))
    for m in range(L):
        for u in np.arange(-1000,1000,dx):
            f = 1 / np.sqrt(2 * np.pi * sigma_c ** 2) * np.exp(-(u - Rxx) ** 2 / (2 * sigma_c ** 2))
            F = 1 / 2 + 1 / 2 * erf((u - Rxx) / np.sqrt(2 * sigma_c ** 2))
            dpmf = np.prod(F[0:m])*np.prod(F[m+1:]) *f[m]
            pmf[m] = pmf[m] + dpmf * dx
    return pmf

def pmf_empirical_tm_1d(x,sigma,N):
    L = x.shape[0]
    y = np.zeros((L,N))
    for j in range(N):
        y[:,j] = x[:,0] + sigma * np.random.randn(x.shape[0],)

    est_shifts = np.zeros((N,1))
    for i in range(N):
        R_xy = np.real(np.fft.ifft(np.fft.fft(y[:,i]) *(np.fft.fft(x[:,0])).conj()))
        est_shifts[i] = np.argmax(R_xy)

    h, bin_edges = np.histogram(np.mod(est_shifts,L), bins=L)
    empirical_pmf = np.expand_dims(h / np.sum(h), axis=-1)
    return empirical_pmf
