import numpy as np
from time import time
import pandas as pd
from make_data import make_data, make_data_1d, r_max
from common_functions import recover_image, align_image
from synchronization import synchronize_2d_tm, synchronize_1d_tm, synchronize_1d, synchronize_and_match_2d
from em import EM_General_Prior, em_1d, relative_error_1d
from learn_distribution_of_rotations import learn_prior


def tm_experiment(seed, image_idx, SNR, Ncopy, Ndir, use_sPCA):
    Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy,SNR=SNR,seed=seed,image_idx=image_idx, sPCA=use_sPCA)
    image0 = recover_image(Coeff_raw, Phi_ns, Freqs, r_max, Mean)

    TT = time()
    Coeff_s, est_rotations = synchronize_2d_tm(Coeff,Freqs,Ndir)
    t_synch = time() - TT
    h, bin_edges = np.histogram((est_rotations - rotations) % 360, bins=np.arange(-0.5, 360 + 0.5))
    measured_rho = np.expand_dims(h / np.sum(h),axis=-1)
    x_s = np.expand_dims(np.mean(Coeff_s,axis=-1),axis=-1)

    x = align_image(x_s, Coeff_raw, Freqs)
    imager = recover_image(x, Phi_ns, Freqs, r_max, Mean)
    err_synch = np.sqrt(np.sum(np.sum((image0 - imager) ** 2,axis=-1),axis=0)/np.sum(np.sum(image0 ** 2, axis=-1),axis=0))
    print('#### tm ####')
    print('e = %f'%err_synch)


    results = pd.DataFrame()
    results = results.append({'use_sPCA': use_sPCA,
                              'seed': seed,
                              'SNR': SNR,
                              'sigma': sigma,
                              'N': Ncopy,
                              'L': Coeff.shape[0],
                              'image_idx': image_idx,
                              'rho': measured_rho,
                              'err': err_synch,
                              'num_iter': 0,
                              't': t_synch},
                             ignore_index=True)

    return results

def standard_em_experiment(seed, image_idx, SNR, Ncopy, Ndir, use_sPCA, use_signal_prior):
    Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy,SNR=SNR,seed=seed,image_idx=image_idx, sPCA=use_sPCA)
    image0 = recover_image(Coeff_raw, Phi_ns, Freqs, r_max, Mean)

    x, num_iter_em_uniform, t_em_uniform = EM_General_Prior(Coeff, Freqs, sigma, Ndir, 1/Ndir * np.ones((Ndir,1)), Ndir, 0, Coeff_raw, use_signal_prior, uniform=True)
    x = align_image(x, Coeff_raw, Freqs)
    imager = recover_image(x, Phi_ns, Freqs, r_max, Mean)
    err_em_uniform = np.sqrt(np.sum(np.sum((image0 - imager) ** 2,axis=-1),axis=0)/np.sum(np.sum(image0 ** 2, axis=-1),axis=0))


    results = pd.DataFrame()
    results = results.append({'use_signal_prior': use_signal_prior,
                              'use_sPCA': use_sPCA,
                              'seed': seed,
                              'SNR': SNR,
                              'sigma': sigma,
                              'N': Ncopy,
                              'L': Coeff.shape[0],
                              'image_idx': image_idx,
                              'err': err_em_uniform,
                              'num_iter': num_iter_em_uniform,
                              't': t_em_uniform},
                             ignore_index=True)

    return results

def known_rotations_experiment(seed, image_idx, SNR, Ncopy, Ndir, use_sPCA):
    Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy,SNR=SNR,seed=seed,image_idx=image_idx, sPCA=use_sPCA)
    image0 = recover_image(Coeff_raw, Phi_ns, Freqs, r_max, Mean)


    est_rotations = rotations
    Coeff_s = np.zeros_like(Coeff)
    for i in range(Ncopy):
        Coeff_s[:,i]=Coeff[:,i] * np.exp(-1j*2*np.pi*est_rotations[i]/360*Freqs)

    t_synch = 0
    h, bin_edges = np.histogram((est_rotations - rotations) % 360, bins=np.arange(-0.5, 360 + 0.5))
    measured_rho = np.expand_dims(h / np.sum(h),axis=-1)
    x_s = np.expand_dims(np.mean(Coeff_s,axis=-1),axis=-1)

    x = align_image(x_s, Coeff_raw, Freqs)
    imager = recover_image(x, Phi_ns, Freqs, r_max, Mean)
    err_synch = np.sqrt(np.sum(np.sum((image0 - imager) ** 2,axis=-1),axis=0)/np.sum(np.sum(image0 ** 2, axis=-1),axis=0))
    print('#### oracle ####')
    print('e = %f'%err_synch)


    results = pd.DataFrame()
    results = results.append({'use_sPCA': use_sPCA,
                              'seed': seed,
                              'SNR': SNR,
                              'sigma': sigma,
                              'N': Ncopy,
                              'L': Coeff.shape[0],
                              'image_idx': image_idx,
                              'rho': measured_rho,
                              'err': err_synch,
                              'num_iter': 0,
                              't': t_synch},
                             ignore_index=True)

    return results


def synchronize_and_match_experiment(seed, image_idx, SNR, Ncopy, Ndir, use_sPCA, P):
    Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy, SNR=SNR, seed=seed,
                                                                                   image_idx=image_idx, sPCA=use_sPCA)
    image0 = recover_image(Coeff_raw, Phi_ns, Freqs, r_max, Mean)

    TT = time()
    Coeff_s, est_rotations = synchronize_and_match_2d(Coeff, Freqs, P=P, L=Ndir)
    t_synch = time() - TT

    h, bin_edges = np.histogram((est_rotations - rotations) % 360, bins=np.arange(-0.5, 360 + 0.5))
    measured_rho = np.expand_dims(h / np.sum(h), axis=-1)
    x_s = np.expand_dims(np.mean(Coeff_s, axis=-1), axis=-1)

    x = align_image(x_s, Coeff_raw, Freqs)
    imager = recover_image(x, Phi_ns, Freqs, r_max, Mean)
    err_synch = np.sqrt(
        np.sum(np.sum((image0 - imager) ** 2, axis=-1), axis=0) / np.sum(np.sum(image0 ** 2, axis=-1), axis=0))
    print('#### synchronize and match ####')
    print('e = %f' % err_synch)

    results = pd.DataFrame()
    results = results.append({'use_sPCA': use_sPCA,
                              'seed': seed,
                              'SNR': SNR,
                              'sigma': sigma,
                              'N': Ncopy,
                              'L': Coeff.shape[0],
                              'image_idx': image_idx,
                              'rho': measured_rho,
                              'err': err_synch,
                              'num_iter': 0,
                              't': t_synch},
                             ignore_index=True)
    return results


def synch_em_experiment(seed, image_idx, SNR, Ncopy, Ndir, use_sPCA, use_signal_prior, gamma, BW, path_learning, P):
    Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy,SNR=SNR,seed=seed,image_idx=image_idx, sPCA=use_sPCA)
    image0 = recover_image(Coeff_raw, Phi_ns, Freqs, r_max, Mean)

    rho_prior = learn_prior(SNR, path_learning)

    TT = time()
    Coeff_s, est_rotations = synchronize_and_match_2d(Coeff, Freqs, P=P, L=Ndir)
    t_synch = time() - TT

    x_s = np.expand_dims(np.mean(Coeff_s, axis=-1), axis=-1)
    x = align_image(x_s, Coeff_raw, Freqs)
    imager = recover_image(x, Phi_ns, Freqs, r_max, Mean)
    err_synch = np.sqrt(
        np.sum(np.sum((image0 - imager) ** 2, axis=-1), axis=0) / np.sum(np.sum(image0 ** 2, axis=-1), axis=0))



    x, num_iter_synch_em, t_synch_em = EM_General_Prior(Coeff_s, Freqs, sigma, Ndir, rho_prior, BW, gamma, Coeff_raw, use_signal_prior)
    x = align_image(x, Coeff_raw, Freqs)
    imager = recover_image(x, Phi_ns, Freqs, r_max, Mean)
    err_synch_em = np.sqrt(np.sum(np.sum((image0 - imager) ** 2,axis=-1),axis=0)/np.sum(np.sum(image0 ** 2, axis=-1),axis=0))


    results = pd.DataFrame()
    results = results.append({'use_signal_prior': use_signal_prior,
                              'use_sPCA': use_sPCA,
                              'seed': seed,
                              'SNR': SNR,
                              'sigma': sigma,
                              'N': Ncopy,
                              'L': Coeff.shape[0],
                              'image_idx': image_idx,
                              'err': err_synch_em,
                              'num_iter': num_iter_synch_em,
                              'BW': BW,
                              'gamma': gamma,
                              't': t_synch+t_synch_em,
                              't_synch': t_synch,
                              't_em': t_synch_em,
                              'err_synch': err_synch},
                             ignore_index=True)

    return results


# --------------- 1d -------------
def standard_em_experiment_1d(seed, sigma, L, b, Ncopy, use_signal_prior):
    np.random.seed(seed)

    n_iter = 1000
    tol = 1e-7
    # Generate MRA measurements
    y, s, n, x = make_data_1d(L,Ncopy,sigma,b)

    if use_signal_prior:
        b_prior = b
    else:
        b_prior = None

    x_init = np.random.randn(L, 1)
    print('start em processing')
    t = time()
    x_est_em, rho_est_em, num_iter_em_uniform = em_1d(y, sigma, n_iter, tol, x_init, b=b_prior,
                                                       rho_prior=None, uniform=True)

    t_em_uniform = time() - t
    err_em_uniform = relative_error_1d(x_est_em, x)

    results = pd.DataFrame()
    results = results.append({'use_signal_prior': use_signal_prior,
                              'L': L,
                              'b': b,
                              'sigma': sigma,
                              'seed': seed,
                              'N': Ncopy,
                              'err': err_em_uniform,
                              'num_iter': num_iter_em_uniform,
                              't': t_em_uniform},
                             ignore_index=True)

    return results

def tm_em_experiment(seed, sigma, L, b, Ncopy, use_signal_prior):
    np.random.seed(seed)

    n_iter = 1000
    tol = 1e-7
    # Generate MRA measurements
    y, s, n, x = make_data_1d(L, Ncopy, sigma, b)

    if use_signal_prior:
        b_prior = b
    else:
        b_prior = None

    y_s, s = synchronize_1d_tm(y)
    x_init = np.expand_dims(np.mean(y_s, axis=1), axis=-1)

    print('start em processing')
    t = time()
    x_est_em, rho_est_em, num_iter_em_uniform = em_1d(y_s, sigma, n_iter, tol, x_init, b=b_prior,
                                                       rho_prior=None, uniform=False)

    t_em_uniform = time() - t
    err_em_uniform = relative_error_1d(x_est_em, x)

    results = pd.DataFrame()
    results = results.append({'use_signal_prior': use_signal_prior,
                              'L': L,
                              'b': b,
                              'sigma': sigma,
                              'seed': seed,
                              'N': Ncopy,
                              'err': err_em_uniform,
                              'num_iter': num_iter_em_uniform,
                              't': t_em_uniform},
                             ignore_index=True)

    return results


def ppm_synch_em_experiment(seed, sigma, L, b, Ncopy, use_signal_prior):
    np.random.seed(seed)

    n_iter = 1000
    tol = 1e-7
    # Generate MRA measurements
    y, s, n, x = make_data_1d(L, Ncopy, sigma, b)

    if use_signal_prior:
        b_prior = b
    else:
        b_prior = None

    y_s, s = synchronize_1d(y,method='ppm')
    x_init = np.expand_dims(np.mean(y_s, axis=1), axis=-1)

    print('start em processing')
    t = time()
    x_est_em, rho_est_em, num_iter_em_uniform = em_1d(y_s, sigma, n_iter, tol, x_init, b=b_prior,
                                                       rho_prior=None, uniform=False)

    t_em_uniform = time() - t
    err_em_uniform = relative_error_1d(x_est_em, x)

    results = pd.DataFrame()
    results = results.append({'use_signal_prior': use_signal_prior,
                              'L': L,
                              'b': b,
                              'sigma': sigma,
                              'seed': seed,
                              'N': Ncopy,
                              'err': err_em_uniform,
                              'num_iter': num_iter_em_uniform,
                              't': t_em_uniform},
                             ignore_index=True)

    return results
