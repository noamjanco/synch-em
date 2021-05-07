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

from plotting import save_relerr_plot,save_relerr_plot_1d,save_relerr_numiter_time_plot, save_plot_gamma

from experiments import tm_experiment, standard_em_experiment, known_rotations_experiment, synchronize_and_match_experiment, synch_em_experiment
from experiments import standard_em_experiment_1d, tm_em_experiment, ppm_synch_em_experiment
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


def strategies():
    Ncopy = 5000
    use_signal_prior = False
    use_sPCA = True
    Ndir = 360
    SNR_range = 2.**(-np.arange(10))
    seed_range = range(10)
    n_jobs = 50
    seed_range = np.asarray(seed_range)+np.max(seed_range)+1

    paths = []
    labels = []

    #Synchronize tm
    path_tm = run_in_parallel(tm_experiment, n_jobs, seed_range, SNR_range, 'tm_N_%d_sPca_%d'%(Ncopy,use_sPCA),
                              Ncopy, Ndir, use_sPCA)
    paths.append(path_tm)
    labels.append('Template Matching')

    #Standard EM on uniform data
    path_em = run_in_parallel(standard_em_experiment, n_jobs, seed_range, SNR_range,
                              'em_N_%d_sPca_%d_use_signal_prior_%d'%(Ncopy,use_sPCA, use_signal_prior),
                              Ncopy, Ndir, use_sPCA, use_signal_prior)
    paths.append(path_em)
    labels.append('EM')

    #Known rotations alignment
    path_oracle = run_in_parallel(known_rotations_experiment, n_jobs, seed_range, SNR_range,
                                  'oracle_N_%d_sPca_%d'%(Ncopy,use_sPCA), Ncopy, Ndir, use_sPCA)
    paths.append(path_oracle)
    labels.append('Known Rotations')

    save_relerr_plot(paths, labels)


def pearson_test():
    np.random.seed(1)

    R = 10
    sigma = 2
    L = 21
    N = 1000
    P = 100
    r_vec = []
    pval_vec = []
    r_vec2 = []
    pval_vec2 = []
    r_vec3 = []
    pval_vec3 = []
    dist = scipy.stats.beta(N / 2 - 1, N / 2 - 1, loc=-1, scale=2)
    r = np.arange(0, 1, 0.001)
    p = 2 * dist.cdf(-abs(r))
    p_critical = 0.05
    r_critical = r[p <= 0.05][0]

    for q in range(R):
        y, s, n, x = make_data_1d(L,N,sigma,0)
        y_s_new, s_est_new = synchronize_1d(y, method='ppm')
        y_s_new2, s_est_new2, y_s_old, s_est_old = synchronize_and_match_1d(y, P=P)
        s_est_new = np.asarray(s_est_new, dtype=int)
        s_est_new2 = np.asarray(s_est_new2, dtype=int)
        n_shifted = np.zeros((L, N))
        n_shifted2 = np.zeros((L, N))
        x_shifted = np.zeros((L, N))
        x_shifted2 = np.zeros((L, N))
        x_only = np.zeros((L, N))
        for i in range(N):
            n_shifted[:, i] = np.roll(n[:, i], -s_est_new[i])
            n_shifted2[:, i] = np.roll(n[:, i], -s_est_new2[i])
            x_shifted[:, i] = np.roll(x, s[i] - s_est_new[i])
            x_shifted2[:, i] = np.roll(x, s[i] - s_est_new2[i])
            x_only[:, i] = np.roll(x, s[i])

        for i in range(L):
            for j in range(L):
                r, pval = stats.pearsonr(x_only[i], n[j])
                r_vec.append(r)
                pval_vec.append(pval)
                r, pval = stats.pearsonr(x_shifted[i], n_shifted[j])
                r_vec2.append(r)
                pval_vec2.append(pval)
                r, pval = stats.pearsonr(x_shifted2[i], n_shifted2[j])
                r_vec3.append(r)
                pval_vec3.append(pval)

    rmin = -.25
    dr = 0.01
    rmax = np.abs(rmin) + dr
    weights = np.ones_like(r_vec) / len(r_vec)
    weights2 = np.ones_like(r_vec2) / len(r_vec2)
    alpha = 1
    plt.hist(r_vec2, bins=np.arange(rmin, rmax, dr), alpha=alpha, weights=weights2)
    plt.hist(r_vec3, bins=np.arange(rmin, rmax, dr), alpha=alpha, weights=weights2)
    plt.hist(r_vec, bins=np.arange(rmin, rmax, dr), alpha=alpha, weights=weights)
    plt.xlabel('Pearson correlation coefficient')
    plt.ylabel('Probability')
    plt.legend(['After Synchronization', 'After Synchronize and Match', 'Before Synchronization'], fontsize='small')
    plt.axvline(r_critical, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(-r_critical, color='k', linestyle='dashed', linewidth=1)
    if not os.path.exists('1d_figures/'):
        os.mkdir('1d_figures/')
    plt.savefig('1d_figures/pearson_test_1d_N_%d_R_%d_sigma_%.2f_L_%d_pval_%.4f_.png'%(N,R,sigma,L,p_critical))
    plt.savefig('1d_figures/pearson_test_1d_N_%d_R_%d_sigma_%.2f_L_%d_pval_%.4f.eps'%(N,R,sigma,L,p_critical))


def degraded_1d():
    L = 21
    b = 0
    Ncopy = 500#5000
    use_signal_prior = False
    SNR_range = 2. ** (-np.arange(-4,5))
    # sigma_range = 1 / np.sqrt(np.asarray(SNR_range) * L)
    sigma_range = 1 / np.sqrt(np.asarray(SNR_range))
    seed_range = range(10)
    seed_range = np.asarray(seed_range)+np.max(seed_range)+1

    paths = []
    labels = []
    n_jobs = 2

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

    save_relerr_plot_1d(paths, labels)


def plot_learned_distribution():
    #TODO: Replace image from the E.Coli to a sample from the image prior
    Ncopy = 5000  # 1000
    use_sPCA = True
    Ndir = 360
    P = 100  # Length of the data used for PPM in partitioned synchronization
    SNR_range = np.asarray([1, 0.1, 0.01, 0.005])
    seed_range = range(10)
    n_jobs = 50

    path_learning = run_training_experiments(synchronize_and_match_experiment, n_jobs, seed_range, SNR_range,
                    'N_%d_sPca_%d_M_%d_SNR_hash_%s' % (Ncopy, use_sPCA, P, hashlib.sha1(SNR_range.view(np.float)).hexdigest()),
                    Ncopy, Ndir, use_sPCA, P)

    for SNR in SNR_range:
        rho_prior = learn_prior(SNR, path_learning, Ndir)
        plt.plot(rho_prior)

    plt.legend(['SNR=1', 'SNR=0.1', 'SNR=0.01', 'SNR=0.005'])
    plt.xlabel('Rotation angle [degrees]')
    plt.ylabel('Probability')
    plt.savefig('mra2d_learned_rotation_dist_.png', bbox_inches='tight')
    plt.savefig('mra2d_learned_rotation_dist.eps', bbox_inches='tight')


def compare_methods(Ncopy, SNR_range, use_signal_prior, use_sPCA, Ndir, BW, P, gamma):
    seed_range = range(10)
    n_jobs = 50

    path_learning = run_training_experiments(synchronize_and_match_experiment, n_jobs, seed_range, SNR_range,
                    'N_%d_sPca_%d_M_%d_SNR_hash_%s' % (Ncopy, use_sPCA, P, hashlib.sha1(SNR_range.view(np.float)).hexdigest()),
                    Ncopy, Ndir, use_sPCA, P)


    seed_range = np.asarray(seed_range) + np.max(seed_range) + 1

    paths = []
    labels = []

    # Synchronize and Match
    path_synchronize_and_math = run_in_parallel(synchronize_and_match_experiment, n_jobs, seed_range, SNR_range,
                              'split_synch_N_%d_sPca_%d_M_%d'%(Ncopy,use_sPCA,P),
                              Ncopy, Ndir, use_sPCA, P)
    paths.append(path_synchronize_and_math)
    labels.append('Synchronize and Match')

    # Standard EM on uniform data
    path_em = run_in_parallel(standard_em_experiment, n_jobs, seed_range, SNR_range,
                              'em_N_%d_sPca_%d_use_signal_prior_%d' % (Ncopy, use_sPCA, use_signal_prior),
                              Ncopy, Ndir, use_sPCA, use_signal_prior)
    paths.append(path_em)
    labels.append('EM')

    # Synch-EM
    path_synch_em = run_in_parallel(synch_em_experiment, n_jobs, seed_range, SNR_range,
                              'synch_em_N_%d_sPca_%d_use_signal_prior_%d_gamma_%f_BW_%d_M_%d'%(Ncopy,use_sPCA, use_signal_prior, gamma, BW, P),
                              Ncopy, Ndir, use_sPCA, use_signal_prior, gamma, BW, path_learning, P)

    paths.append(path_synch_em)
    labels.append('Synch-EM')

    save_relerr_numiter_time_plot(paths, labels)


def compare_prior_weights():
    Ncopy = 5000  # 5000#1000
    use_signal_prior = False
    use_sPCA = True
    Ndir = 360
    BW = 36
    P = 100  # Length of the data used for PPM in partitioned synchronization

    SNR_range = 2. ** (-np.arange(10))
    seed_range = range(10)
    n_jobs = 50

    path_learning = run_training_experiments(synchronize_and_match_experiment, n_jobs, seed_range, SNR_range,
                                             'N_%d_sPca_%d_M_%d_SNR_hash_%s' % (Ncopy, use_sPCA, P, hashlib.sha1(SNR_range.view(np.float)).hexdigest()),
                                             Ncopy, Ndir, use_sPCA, P)

    seed_range = np.asarray(seed_range) + np.max(seed_range) + 1
    paths = []
    labels = []
    gamma_range = [100, 10, 1, 0]
    for gamma in gamma_range:
        path_synch_em = run_in_parallel(synch_em_experiment, n_jobs, seed_range, SNR_range,
                                        'synch_em_N_%d_sPca_%d_use_signal_prior_%d_gamma_%f_BW_%d_M_%d' % (
                                        Ncopy, use_sPCA, use_signal_prior, gamma, BW, P),
                                        Ncopy, Ndir, use_sPCA, use_signal_prior, gamma, BW, path_learning, P)

        paths.append(path_synch_em)
        labels.append('$\gamma=%d$'%gamma)

    save_plot_gamma(paths, labels)


def compare_approx_and_empirical_pmf():
    L = 21
    sigma = 3
    x = np.random.randn(L, 1)
    pmf = pmf_approx_tm_1d(x,sigma)
    plt.plot(pmf,'-')
    # empirical distribution
    N = 10000
    empirical_pmf = pmf_empirical_tm_1d(x, sigma, N)
    plt.plot(empirical_pmf,'+')
    plt.legend(['Approximated PMF', 'Empirical PMF'])
    plt.xlabel('Estimated shift')
    plt.ylabel('Probability')
    name = 'pmf_approx_vs_empirical_L_%d_sigma_%f_N_%d'%(L,sigma,N)
    plt.savefig('1d_figures/%s.eps' % name)
    plt.savefig('1d_figures/%s_.png' % name)


def pmf_approx_error_vs_L():
    L_range = np.asarray([3,11,21,51])
    R = 10
    sigma = 3
    error = np.zeros((len(L_range),R))
    n = 0
    N = 100000
    for L in L_range:
        for r in range(R):
            x = np.random.randn(L, 1)
            pmf = pmf_approx_tm_1d(x, sigma)
            empirical_pmf = pmf_empirical_tm_1d(x, sigma, N)
            error[n,r] = np.mean((pmf - empirical_pmf)**2)
        n += 1
    plt.plot(L_range,np.mean(error,axis=-1),'-+')
    plt.xlabel('L')
    plt.ylabel('Mean Squared Error')
    name = 'pmf_approx_err_vs_L_sigma_%f_N_%d' % (sigma, N)
    plt.savefig('1d_figures/%s.eps' % name)
    plt.savefig('1d_figures/%s_.png' % name)


def main():
    # Figure 1: Image formation. This function plots to a local file a figure containing a clean particle projection,
    #           a rotated projection and a noisy projection.
    # image_formation()

    # Figure 2: Different strategies in image recovery. This function runs Template matching, standard EM and known
    #           rotations alignment, in different SNRs, and compares the relative error averaged over different trials.
    #strategies()

    # Figure 3: Detecting induced correlation due to PPM synchronization in 1-D MRA using Pearson's correlation coefficients
    #pearson_test()


    # Figure 4: Degraded performance in 1-D MRA after synchronization
    #degraded_1d()

    # Figure 5: Learned distribution of rotations after Synchronize and Match in 2-D
    #plot_learned_distribution()

    # Figure 6: Performance comparison, N=1000
    #compare_methods(Ncopy=1000, SNR_range = 2.**(-np.arange(10)), use_signal_prior=False, use_sPCA=True, Ndir=360, BW=36, P=100, gamma=100)

    # Figure 7: Performance comparison, N=5000
    #compare_methods(Ncopy=5000, SNR_range = 2.**(-np.arange(10)), use_signal_prior=False, use_sPCA=True, Ndir=360, BW=36, P=100, gamma=100)

    # Figure 8: Performance comparison, N=10000
    #compare_methods(Ncopy=10000, SNR_range = 2.**(-np.arange(10)), use_signal_prior=False, use_sPCA=True, Ndir=360, BW=36, P=100, gamma=100)

    # Figure 9: Performance comparison, N=1000 with signal prior
    #compare_methods(Ncopy=1000, SNR_range=2.**(-np.arange(2.5,5.5,0.5)), use_signal_prior=True, use_sPCA=True, Ndir=360, BW=36, P=100, gamma=100)

    # Figure 10: Performance comparison, N=5000 with signal prior
    #compare_methods(Ncopy=5000, SNR_range=2.**(-np.arange(2.5,5.5,0.5)), use_signal_prior=True, use_sPCA=True, Ndir=360, BW=36, P=100, gamma=100)

    # Figure 11: Compare prior weights
    #compare_prior_weights()

    # Figure 12: Comparison of pmf approximation and empirical pmf for 1d tm using the signal as a refernece
    #compare_approx_and_empirical_pmf()

    # Figure 13: PMF Approximation error vs L
    pmf_approx_error_vs_L()
    
if __name__ == '__main__':
    main()
