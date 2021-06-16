import scipy
from scipy import io
from scipy.special import jv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import skimage.transform
from time import time
import pandas as pd
from joblib import Parallel, delayed
from os import listdir
from os.path import isfile, join
from datetime import datetime
from scipy import stats

from plotting import save_relerr_plot,save_relerr_plot_1d,save_relerr_numiter_time_plot, save_plot_gamma, save_relerr_numiter_time_plot2, save_relerr_num_iter_plot_1d

from experiments import tm_experiment, standard_em_experiment, known_rotations_experiment, synchronize_and_match_experiment, synch_em_experiment, ppm_em_experiment, ppm_experiment
from experiments import standard_em_experiment_1d, tm_em_experiment, ppm_synch_em_experiment, synchronize_and_match_em_experiment, dist_est_em_experiment_1d, synch_em_1d_experiment, ppm_experiment_1d
from common_functions import recover_image
from make_data import make_data, make_data_1d
from synchronization import synchronize_1d, synchronize_and_match_1d
from learn_distribution_of_rotations import learn_prior, run_training_experiments, pmf_approx_tm_1d, pmf_empirical_tm_1d
import hashlib


def normalize(z):
    z_normalized = z / np.abs(z)
    z_normalized[z_normalized == 0] = 0
    return z_normalized


def synchronize_2d_tm(y, Freqs):
    L, N = y.shape
    angles = 360
    num_freqs = Freqs.shape[0]
    est_rotations = np.zeros((N,))
    for n in tqdm(range(N)):
        est_rotations[n] = relative_rotation(y[:num_freqs,1],y[:num_freqs,n],Freqs[:num_freqs],angles=angles)

    y_s = np.zeros_like(y)
    for i in range(N):
        y_s[:,i]=y[:,i] * np.exp(-1j*2*np.pi*est_rotations[i]/360*Freqs)
    return y_s, est_rotations


def synchronize_2d(y, Freqs):
    L, N = y.shape
    rho = np.zeros((N, N))
    angles = 360
    num_freqs = Freqs.shape[0]
    for n in tqdm(range(N)):
        for m in range(n):
            rho[n, m] = relative_rotation(y[:num_freqs,m],y[:num_freqs,n],Freqs[:num_freqs],angles=angles)
    for n in range(N):
        for m in range(n, N):
            rho[n, m] = (angles - rho[m, n]) % angles

    H = np.exp(1j*2*np.pi*rho/angles)
    b = np.random.randn(N) +1j*np.random.randn(N)
    b = b / np.linalg.norm(b)
    max_iter = int(5e2)
    min_step = 1e-3
    n = 0
    step = 1.

    while step > min_step and n < max_iter:
        b_prev = b
        b = H @ b
        b = normalize(b)
        step = np.linalg.norm(b-b_prev)
        n = n+1

    est_rotations = np.round(np.angle(b)/(2*np.pi)*360)
    y_s = np.zeros_like(y)
    for i in range(N):
        y_s[:,i]=y[:,i] * np.exp(-1j*2*np.pi*est_rotations[i]/360*Freqs)
    return y_s, est_rotations


def synchronize_2d_novel(y, Freqs):
    y_s, est_rotations = synchronize_2d(y, Freqs)
    L, N = y.shape
    D = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            R_nm = np.sum(np.abs(y[:, p] - y[:, q]) ** 2)
            D[p, q] = R_nm

    est_rotations_new = np.array(
        np.take_along_axis(np.repeat(np.expand_dims(est_rotations, axis=-1), N, axis=-1), D.argsort(axis=0), axis=0)[1,
        :], dtype=int)

    y_s_new = np.zeros((L, N), dtype=np.complex128)
    for i in range(N):
        y_s_new[:,i]=y[:,i] * np.exp(-1j*2*np.pi*est_rotations_new[i]/360*Freqs)
    return y_s_new, est_rotations_new


def synchronize_novel_efficient(y, Freqs, M):
    L, N = y.shape
    y_s, est_rotations = synchronize_2d(y[:,:M], Freqs)
    D = np.zeros((M, N))
    for p in range(M):
        for q in range(N):
            R_nm = np.sum(np.abs(y[:, p] - y[:, q]) ** 2)
            D[p, q] = R_nm

    est_rotations_new = np.array(
        np.take_along_axis(np.repeat(np.expand_dims(est_rotations, axis=-1), N, axis=-1), D.argsort(axis=0), axis=0)[1,
        :], dtype=int)

    y_s_new = np.zeros((L, N), dtype=np.complex128)
    for i in range(N):
        y_s_new[:,i]=y[:,i] * np.exp(-1j*2*np.pi*est_rotations_new[i]/360*Freqs)

    return y_s_new, est_rotations_new, y_s


def synchronize_novel_efficient_v2(y, Freqs, M):
    L, N = y.shape
    y_s, est_rotations = synchronize_2d(y[:,:M], Freqs)
    D = np.zeros((M, N))
    for p in range(M):
        for q in range(N):
            R_nm = np.sum(np.abs(y[:, p] - y[:, q]) ** 2)
            D[p, q] = R_nm

    matching_rotations = np.take_along_axis(np.repeat(np.expand_dims(est_rotations, axis=-1), N, axis=-1), D.argsort(axis=0), axis=0)
    est_rotations_new = np.array(np.concatenate([matching_rotations[1,:M],matching_rotations[0,M:]],axis=-1), dtype=int)

    y_s_new = np.zeros((L, N), dtype=np.complex128)
    for i in range(N):
        y_s_new[:,i]=y[:,i] * np.exp(-1j*2*np.pi*est_rotations_new[i]/360*Freqs)

    return y_s_new, est_rotations_new, y_s


def im_dist(x,y,Freqs):
    ydd = np.ones((3600,))*np.sum(np.conj(y) * y)
    zz = x
    zze = (zz @ np.ones((1,3600))) * np.exp(1j*Freqs*2*np.pi/3600*np.arange(3600))
    DIS = np.sum(np.conj(zze) * zze,axis=0) + ydd - 2*np.real(np.transpose(np.conj(zze)) @ y)
    Dis = np.min(np.real(DIS))
    if Dis < 0:
        Dis = 0
    return np.sqrt(Dis)


def im_dist2(x,y,Freqs):
    return np.sqrt(np.sum(np.abs(x-y)**2))


def align_image(x, y, Freqs):
    zz = x[:,0]
    zze = np.zeros((zz.shape[0],3600),dtype=np.complex128)
    for i in range(3600):
        zze[:,i] = zz * np.exp(1j*i*2*np.pi/3600*Freqs)
    # Align_dis = np.zeros()
    align_dis = np.sum((np.abs(zze-y))**2,axis=0)
    ind = np.argmin(align_dis)
    zz = zz * np.exp(1j*2*np.pi*ind/3600*Freqs)
    return np.expand_dims(zz,axis=-1)


def relative_rotation(x, y, Freqs, angles=360):
    zz = x
    zze = np.zeros((zz.shape[0],angles),dtype=np.complex128)
    for i in range(angles):
        zze[:,i] = zz * np.exp(1j*i*2*np.pi/angles*Freqs)
    # Align_dis = np.zeros()
    align_dis = np.sum((np.abs(zze-np.expand_dims(y,axis=-1)))**2,axis=0)
    ind = np.argmin(align_dis)
    return ind

def run_in_parallel(task, n_jobs, seed_range, SNR_range, label, *args):
    print('Started %s'%label)
    if not os.path.exists('2d_results/'):
        os.mkdir('2d_results/')
    path_output = '2d_results/'+label
    if not os.path.exists(path_output):
        os.mkdir(path_output)
        results = Parallel(n_jobs=n_jobs)(delayed(task)(seed, image_idx, SNR, *args) for SNR in SNR_range for seed in seed_range for image_idx in [seed])
        results = pd.concat(results, ignore_index=True)
        results.to_pickle(path_output + '/combined.pickle')
    print('Finished %s'%label)
    return path_output +'/'


def run_in_parallel_1d(task, n_jobs, seed_range, sigma_range, label, *args):
    print('Started %s'%label)
    if not os.path.exists('1d_results/'):
        os.mkdir('1d_results/')
    path_output = '1d_results/'+label
    if not os.path.exists(path_output):
        os.mkdir(path_output)
        results = Parallel(n_jobs=n_jobs)(delayed(task)(seed, sigma, *args) for sigma in sigma_range for seed in seed_range)
        results = pd.concat(results, ignore_index=True)
        results.to_pickle(path_output + '/combined.pickle')
    print('Finished %s'%label)
    return path_output +'/'

def image_formation():
    Ncopy = 1000
    use_sPCA = False
    image_idx = 0
    SNR = .1
    r_max = 64
    seed = 0
    Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy, SNR=SNR, seed=seed,
                                                                                   image_idx=image_idx, sPCA=use_sPCA)
    image0 = recover_image(Coeff_raw, Phi_ns, Freqs, r_max, Mean)
    Rl = np.expand_dims(np.exp(1j * 2 * np.pi *45/ 360 * Freqs),axis=-1)
    Coeff_rotated = Coeff_raw * Rl
        
    image_rotated = recover_image(Coeff_rotated, Phi_ns, Freqs, r_max, Mean)

    image_rotated_noisy = recover_image(np.expand_dims(Coeff[:,0],axis=-1), Phi_ns, Freqs, r_max, Mean)

    plt.subplot(131)
    plt.imshow(image0,cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(image_rotated,cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(image_rotated_noisy,cmap='gray')
    plt.axis('off')
    plt.savefig('mra_2d_observations_model.png')
    plt.savefig('mra_2d_observations_model.eps')
    plt.clf()


def importance_of_signal_prior():
    Ncopy = 100
    use_signal_prior = False
    use_sPCA = True
    Ndir = 360
    SNR_range = [1,0.1,0.02,0.01]#2.**(-np.arange(10))
    seed_range = range(10)
    n_jobs = 50
    seed_range = np.asarray(seed_range)+np.max(seed_range)+1
    gamma = 0
    BW = 360
    path_learning = ''
    P = Ncopy
    paths = []
    labels = []

    path_tm = run_in_parallel(ppm_experiment, n_jobs, seed_range, SNR_range, 'ppm_N_%d_sPca_%d'%(Ncopy,use_sPCA),
                              Ncopy, Ndir, use_sPCA)
    paths.append(path_tm)
    labels.append('PPM')

    #Synchronize + EM
    #path_tm = run_in_parallel(ppm_em_experiment, n_jobs, seed_range, SNR_range, 'ppm_em_N_%d_sPca_%d'%(Ncopy,use_sPCA),
    #                          Ncopy, Ndir, use_sPCA, use_signal_prior, gamma, BW, path_learning, P)
    #paths.append(path_tm)
    #labels.append('PPM+EM')

    #Standard EM on uniform data
    path_em = run_in_parallel(standard_em_experiment, n_jobs, seed_range, SNR_range,
                              'em_N_%d_sPca_%d_use_signal_prior_%d'%(Ncopy,use_sPCA, use_signal_prior),
                              Ncopy, Ndir, use_sPCA, use_signal_prior)
    paths.append(path_em)
    labels.append('EM')

    use_signal_prior = True
    path_em = run_in_parallel(standard_em_experiment, n_jobs, seed_range, SNR_range,
                              'em_N_%d_sPca_%d_use_signal_prior_%d'%(Ncopy,use_sPCA, use_signal_prior),
                              Ncopy, Ndir, use_sPCA, use_signal_prior)
    paths.append(path_em)
    labels.append('EM+Signal Prior')
    save_relerr_plot(paths, labels)


def degraded_1d():
    L = 41#21
    P = 100#100
    b = 0
    Ncopy = 500#500#5000
    use_signal_prior = False
    SNR_range = 2. ** (-np.arange(-4,5))
    # sigma_range = 1 / np.sqrt(np.asarray(SNR_range) * L)
    sigma_range = 1 / np.sqrt(np.asarray(SNR_range))
    seed_range = range(20)
    seed_range = np.asarray(seed_range)+np.max(seed_range)+1

    paths = []
    labels = []
    n_jobs = 20

    # EM on uniform data
    path_em = run_in_parallel_1d(standard_em_experiment_1d, n_jobs, seed_range, sigma_range,
                                 'em_L_%d_b_%d_N_%d_use_signal_prior_%d'%(L,b,Ncopy, use_signal_prior),
                                 L, b, Ncopy, use_signal_prior)
    paths.append(path_em)
    labels.append('EM')


    # TM + EM
    path_tm_em = run_in_parallel_1d(tm_em_experiment, n_jobs, seed_range, sigma_range,
                                    'tm_em_L_%d_b_%d_N_%d_use_signal_prior_%d' % (L, b, Ncopy, use_signal_prior),
                                    L, b, Ncopy, use_signal_prior)
    paths.append(path_tm_em)
    labels.append('TM+EM')

    # Synch + EM
    path_ppm_synch_em = run_in_parallel_1d(ppm_synch_em_experiment, n_jobs, seed_range, sigma_range,
                                           'synch_em_L_%d_b_%d_N_%d_use_signal_prior_%d' % (L, b, Ncopy, use_signal_prior),
                                           L, b, Ncopy, use_signal_prior)
    paths.append(path_ppm_synch_em)
    labels.append('Synch+EM')


    # Synchronize and Match + EM
    path_synchronize_and_match_em = run_in_parallel_1d(synchronize_and_match_em_experiment, n_jobs, seed_range, sigma_range,
                                           'synchronize_and_match_em_L_%d_b_%d_N_%d_use_signal_prior_%d_P_%d' % (L, b, Ncopy, use_signal_prior, P),
                                           L, b, Ncopy, use_signal_prior, P)
    paths.append(path_synchronize_and_match_em)
    labels.append('Synchronize and Match + EM')

    save_relerr_plot_1d(paths, labels)

def comparison_1d():
    L = 121#21#21
    P = 100#100
    b = 2#0#0#0
    Ncopy = 5000#500#500#5000
    use_signal_prior = False
    #SNR_range = 2. ** (-np.arange(-4,5))
    SNR_range = 2. ** (-np.arange(-1,2.5,0.5))
    # sigma_range = 1 / np.sqrt(np.asarray(SNR_range) * L)
    sigma_range = 1 / np.sqrt(np.asarray(SNR_range))
    seed_range = range(10)
    seed_range = np.asarray(seed_range)+np.max(seed_range)+1

    paths = []
    labels = []
    n_jobs = 50


    # PPM on uniform data
    #path_ppm = run_in_parallel_1d(ppm_experiment_1d, n_jobs, seed_range, sigma_range,
    #                             'ppm_L_%d_b_%d_N_%d_use_signal_prior_%d'%(L,b,Ncopy, use_signal_prior),
    #                             L, b, Ncopy, use_signal_prior)
    #paths.append(path_ppm)
    #labels.append('PPM')

    # Synch + EM
    #path_ppm_synch_em = run_in_parallel_1d(ppm_synch_em_experiment, n_jobs, seed_range, sigma_range,
    #                                       'synch_em_L_%d_b_%d_N_%d_use_signal_prior_%d' % (L, b, Ncopy, use_signal_prior),
    #                                       L, b, Ncopy, use_signal_prior)
    #paths.append(path_ppm_synch_em)
    #labels.append('Synch+EM')

    # EM on uniform data
    path_em = run_in_parallel_1d(standard_em_experiment_1d, n_jobs, seed_range, sigma_range,
                                 'em_L_%d_b_%d_N_%d_use_signal_prior_%d'%(L,b,Ncopy, use_signal_prior),
                                 L, b, Ncopy, use_signal_prior)
    paths.append(path_em)
    labels.append('EM')

    # EM on uniform data + distribution estimation
    path_dist_est_em = run_in_parallel_1d(dist_est_em_experiment_1d, n_jobs, seed_range, sigma_range,
                                 'dist_est_em_L_%d_b_%d_N_%d_use_signal_prior_%d'%(L,b,Ncopy, use_signal_prior),
                                 L, b, Ncopy, use_signal_prior)
    paths.append(path_dist_est_em)
    labels.append('Dist-Est EM')

    # Synch-EM 1d
    for gamma in [0,0.1, 1]:
        path_synch_em = run_in_parallel_1d(synch_em_1d_experiment, n_jobs, seed_range, sigma_range,
                                     'synch_em_L_%d_b_%d_N_%d_use_signal_prior_%d_P_%d_gamma_%f'%(L,b,Ncopy, use_signal_prior, P, gamma),
                                     L, b, Ncopy, use_signal_prior, P, gamma)
        paths.append(path_synch_em)
        labels.append('Synch-EM '+'$\gamma=%.2f$'%gamma)

    

    save_relerr_num_iter_plot_1d(paths, labels)


def draw_from_prior():
    Ncopy = 10000
    use_sPCA = True
    image_idx = 0
    SNR = 10
    filename = 'E70s.mat'
    r_max = 64
    seed = 0
    Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy, SNR=SNR, seed=seed,
                                                                                   image_idx=image_idx, sPCA=use_sPCA)
    data_prior = np.expand_dims(4*np.exp(-np.asarray(Freqs)/8),axis=-1)
    Gamma_a = np.diag(np.abs(data_prior[:,0])**2)
    np.random.seed(10)

    for i in range(9):
        x = 1/np.sqrt(2)*(np.random.multivariate_normal(np.zeros((Freqs.shape[0],)),Gamma_a) +1j * np.random.multivariate_normal(np.zeros((Freqs.shape[0],)),Gamma_a))
        Coeff_raw = np.expand_dims(x,axis=-1)
        image0 = recover_image(Coeff_raw, Phi_ns, Freqs, r_max, Mean)
        plt.subplot('33%d'%(i+1))
        plt.imshow(image0,cmap='gray')
        plt.axis('off')
    plt.savefig('draw_from_prior.png')
    plt.savefig('draw_from_prior.eps')
    plt.clf()


def main():
    #importance_of_signal_prior()

    #degraded_1d()

    #comparison_1d()

    draw_from_prior()
    
if __name__ == '__main__':
    main()
