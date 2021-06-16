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


def Bessel_ns_v5(N):
    B = io.loadmat('Bessel180.mat')['Bessel180']
    B = B[B[:,2] < np.pi*N,:]
    x,y = np.meshgrid(np.arange(-N,N+1),np.arange(-N,N+1))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    r = r / N
    n = B[:,0]
    s = B[:,1]
    R_ns = B[:,2]
    Phi_ns = np.zeros((2*N+1,2*N+1,B.shape[0]),dtype=np.complex)
    ns = np.unique(n)
    r_unique, r_I = np.unique(r.flatten(),return_inverse=True)
    for k in tqdm(range(len(ns))):
        nk = ns[k]
        Y = np.exp(1j*nk*theta.flatten())
        mask = np.where(n == nk)[0]
        r0 = np.expand_dims(r_unique,axis=-1) @ np.transpose(np.expand_dims(R_ns[mask],axis=-1))
        F = jv(nk,r0)
        F = F[r_I,:]
        phik = np.repeat(np.expand_dims(Y,axis=-1),F.shape[-1],axis=-1) * F
        phik = phik * np.repeat(np.expand_dims(1/(N*np.sqrt(np.pi)*np.abs(jv(nk+1, R_ns[mask]))),axis=0),phik.shape[0],axis=0)
        phik = np.reshape(phik,(theta.shape[0],theta.shape[1],len(mask)))
        Phi_ns[:,:,mask] = phik
    Phi_ns = np.reshape(Phi_ns,((2*N+1)**2,B.shape[0]))
    Phi_ns = Phi_ns[r.flatten() <= 1, :]
    return Phi_ns, n, s, R_ns

# Initialization: Compute the Fourier-Bessel basis and it's pseudo inverse, store in file
r_max = 64
if not os.path.isfile('basis.pickle'):
    Phi_ns_global, ang_freqs_global, rad_freqs_global, _ = Bessel_ns_v5(r_max)
    Phi_ns_pinv_global = scipy.linalg.pinv(np.hstack([Phi_ns_global, np.conj(Phi_ns_global[:, ang_freqs_global != 0])]))
    with open('basis.pickle','wb') as file:
        pickle.dump((Phi_ns_global,ang_freqs_global,rad_freqs_global,Phi_ns_pinv_global), file)
        print('Successfully wrote basis into memory')
else:
    with open('basis.pickle', 'rb') as file:
        Phi_ns_global, ang_freqs_global, rad_freqs_global, Phi_ns_pinv_global = pickle.load(file)
print('init completed')

def FB(data, r_max, Phi_ns_pinv, ang_freqs, Phi_ns):
    L = data.shape[0]
    N = np.floor(L/2)
    P = data.shape[2]
    x,y = np.meshgrid(np.arange(-N,N+1),np.arange(-N,N+1))
    r = np.sqrt(x**2 + y**2)
    data = np.reshape(data, (L**2, P))
    data = data[r.flatten() <= r_max, :]
    coeff = Phi_ns_pinv @ data
    coeff = coeff[:len(ang_freqs),:]
    Mean_Im = np.real(Phi_ns[:, ang_freqs == 0] @ np.mean(coeff[ang_freqs==0,:],axis=1))
    X = np.zeros((L,L))
    X[r <= r_max] = Mean_Im
    return coeff, X


def MP_rankEst(D, n, var_hat):
    l_D = len(D)
    lambd = l_D / n
    K = len(np.where(D > var_hat * (1 + np.sqrt(lambd))**2)[0])
    return K


def FBSPCA_MP_rankEst(P, U, D, freqs, rad_freqs, nv):
    K = MP_rankEst(D[freqs == 0], P, nv)
    gamma = len(D[freqs == 0]) / P
    freqs_tmp = np.expand_dims(freqs[freqs == 0], axis=-1)
    rad_freqs_tmp = np.expand_dims(rad_freqs[freqs == 0], axis=-1)
    D_tmp = D[freqs == 0]
    U_tmp = U[:, freqs == 0]
    freqs_tmp = freqs_tmp[0:K]
    rad_freqs_tmp = rad_freqs_tmp[0:K]
    D_tmp = D_tmp[0:K]
    l_k = 0.5 * ((D_tmp - (gamma + 1) * nv) + np.sqrt(((gamma + 1) * nv - D_tmp) ** 2 - 4 * gamma * nv ** 2))
    SNR_k = l_k / nv
    SNR = (SNR_k ** 2 - gamma) / (SNR_k + gamma)
    U_tmp = U_tmp[:, 0:K]
    weight = 1 / (1 + 1 / SNR)
    Freqs = freqs_tmp
    Rad_Freqs = rad_freqs_tmp
    UU = U_tmp
    W = weight
    for i in range(1, int(np.max(freqs)) + 1):
        K = MP_rankEst(D[freqs == i], 2 * P, nv)
        gamma = len(D[freqs == i]) / (2 * P)
        if K != 0:
            freqs_tmp = freqs[freqs == i]
            rad_freqs_tmp = rad_freqs[freqs == i]
            D_tmp = D[freqs == i]
            U_tmp = U[:, freqs == i]
            freqs_tmp = freqs_tmp[0:K]
            rad_freqs_tmp = rad_freqs_tmp[0:K]
            D_tmp = D_tmp[0:K]
            l_k = 0.5 * ((D_tmp - (gamma + 1) * nv) + np.sqrt(((gamma + 1) * nv - D_tmp) ** 2 - 4 * gamma * nv ** 2))
            SNR_k = l_k / nv
            SNR = (SNR_k ** 2 - gamma) / (SNR_k + gamma)
            U_tmp = U_tmp[:, 0:K]
            weight = 1 / (1 + 1 / SNR)
        else:
            continue
        Freqs = np.vstack([Freqs, np.expand_dims(freqs_tmp, axis=-1)])
        Rad_Freqs = np.vstack([Rad_Freqs, np.expand_dims(rad_freqs_tmp, axis=-1)])
        UU = np.hstack([UU, U_tmp])
        W = np.vstack([W, weight])
    return UU, Freqs, Rad_Freqs, W


def WF_FBSPCA(data, Mean, r_max, U, Freqs, W, filter_flag):
    L = data.shape[0]
    N = np.floor(L / 2)
    P = data.shape[2]
    x, y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    r = np.sqrt(x ** 2 + y ** 2)

    datanew = data.copy()
    for i in range(P):
        datanew[:,:,i] = datanew[:,:,i] - Mean

    datanew = np.reshape(datanew, (L ** 2, P))
    datanew = datanew[r.flatten() <= r_max, :]
    Coeff = scipy.linalg.pinv(np.hstack([U, np.conj(U[:, Freqs[:,0] != 0])])) @ datanew
    Coeff = Coeff[:len(Freqs),:]
    if filter_flag:
        Coeff = np.diag(W) @ Coeff

    return Coeff


def sPCA_Basis(Phi_ns, ang_freqs, rad_freqs, coeff, data, r_max):
    l = Phi_ns.shape[1]
    U = np.zeros((l,l),dtype=np.complex128)
    D = np.zeros((l,1),dtype=np.complex128)
    P = coeff.shape[1]
    for k in range(1,int(np.max(ang_freqs))+2):
        tmp = coeff[ang_freqs == k-1, :]
        if k == 1:
            mean_tmp = np.mean(tmp,axis=1)
            for i in range(P):
                tmp[:, i] = tmp[:, i] - mean_tmp
        C = 1 / P * np.real(tmp @ np.conj(np.transpose(tmp)))
        d, u = np.linalg.eig(C)
        # d, u = scipy.linalg.eig(C)
        id = np.argsort(d)[::-1]
        d = d[id]
        u = u[:, id]
        id1 = np.where(ang_freqs == k-1)[0][0]
        id2 = np.where(ang_freqs == k-1)[0][-1]
        U[id1:id2+1, id1:id2+1] = u
        D[id1:id2+1] = np.expand_dims(d,axis=-1)
    D = np.real(D)
    U_ns = Phi_ns @ U
    id = np.argsort(D,axis=0)[::-1][:,0]
    D = D[id]
    U_ns = U_ns[:,id]
    ang_freqs = ang_freqs[id]
    rad_freqs = rad_freqs[id]
    U = U_ns

    L = data.shape[0]
    N = np.floor(L/2)
    x, y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    r = np.sqrt(x ** 2 + y ** 2)
    test = np.reshape(data,(L ** 2,P))
    test = test[r.flatten() > r_max, :]
    nv = np.var(test.flatten())

    UU, Freqs, rad_Freqs, W = FBSPCA_MP_rankEst(P, U, D, ang_freqs, rad_freqs, nv)

    return UU, Freqs, rad_Freqs, W


def make_data(Ncopy, SNR, seed, image_idx, sPCA):
    use_caching = False
    if not os.path.exists('make_data_cache/'):
        os.mkdir('make_data_cache/')
    label = 'make_data_output_Ncopy_%d_SNR_%f_seed_%d_image_idx_%d_sPCA_%d.pickle'%(Ncopy, SNR, seed, image_idx, sPCA)
    cache_file = 'make_data_cache/' + label

    if not os.path.exists(cache_file):
        # Run make data
        np.random.seed(seed)
        D = io.loadmat('E70s.mat')['data'][:, :, image_idx]
        data = np.zeros((D.shape[0], D.shape[1], Ncopy))
        rotations = np.zeros((Ncopy,))
        for i in range(Ncopy):
            # theta = 360 * np.random.rand(1)
            theta = np.floor(360 * np.random.rand(1))
            rotations[i] = theta
            rotdata = skimage.transform.rotate(D, theta)
            data[:, :, i] = rotdata
        sigma = np.sqrt(np.var(np.reshape(data[:, :, 0], (data.shape[0] ** 2, 1))) / SNR)
        data = data + sigma * np.random.randn(data.shape[0], data.shape[1], data.shape[2])
        data_raw = np.expand_dims(skimage.transform.rotate(D, 45), axis=-1)
        r_max = np.floor(data.shape[0] / 2)

        coeff, Mean = FB(data, r_max, Phi_ns_pinv_global, ang_freqs_global, Phi_ns_global)
        coeff_raw, _ = FB(data_raw, r_max, Phi_ns_pinv_global, ang_freqs_global, Phi_ns_global)

        if sPCA:
            UU, Freqs, rad_Freqs, W = sPCA_Basis(Phi_ns_global, ang_freqs_global, rad_freqs_global, coeff, data, r_max)
            coeff = WF_FBSPCA(data=data, Mean=Mean, r_max=r_max, U=UU, Freqs=Freqs, W=W, filter_flag=0)
            coeff_raw = WF_FBSPCA(data=data_raw, Mean=Mean, r_max=r_max, U=UU, Freqs=Freqs, W=W, filter_flag=0)
            Phi_ns = UU
            ang_freqs = Freqs[:, 0]
            rad_freqs = rad_Freqs[:, 0]
        else:
            ang_freqs = ang_freqs_global.copy()
            rad_freqs = rad_freqs_global.copy()
            Phi_ns = Phi_ns_global.copy()

        num_coeffs = ang_freqs.shape[0]  # 1000
        coeff, ang_freqs, rad_freqs, Phi_ns, coeff_raw = coeff[:num_coeffs], ang_freqs[:num_coeffs], rad_freqs[
                                                                                                     :num_coeffs], Phi_ns[
                                                                                                                   :,
                                                                                                                   :num_coeffs], coeff_raw[
                                                                                                                                 :num_coeffs]
        # save results to cache file
        if use_caching:
            with open(cache_file,'wb') as file:
                pickle.dump((coeff, ang_freqs, rad_freqs, Mean, Phi_ns, sigma, coeff_raw, rotations),file)
    else:
        # Load make data results from cached output
        with open(cache_file,'rb') as file:
            coeff, ang_freqs, rad_freqs, Mean, Phi_ns, sigma, coeff_raw, rotations = pickle.load(file)


    return coeff, ang_freqs, rad_freqs, Mean, Phi_ns, sigma, coeff_raw, rotations


def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j )# / np.sqrt(N)
    return W


def generate_signal_covariance(L,b):
    F = DFT_matrix(L)
    PSD_oneside = 1 / (np.arange(1, 1 + (L + 1) / 2) ** b)
    PSD = np.concatenate([PSD_oneside, PSD_oneside[1::][::-1]])
    D = np.diag(PSD)
    Sigma = np.real(F.conj().T @ D @ F)
    Sigma = Sigma / np.mean(np.diag(Sigma))
    return Sigma


def generate_signal(L=21, b=0):
    if b == 0:
        x = np.random.randn(L)
        # x = x / np.sqrt(np.sum(x ** 2))
        return x
    else:
        Sigma = generate_signal_covariance(L,b)
        x =  np.random.multivariate_normal(np.zeros((L,)),Sigma)
        x = x - np.mean(x)
        x = x / np.sqrt(np.sum(x ** 2)) * np.sqrt(L)
        return x


def generate_observations(x,N,sigma):
    L = len(x)
    y = np.zeros((L,N))
    n = np.zeros((L,N))
    s = np.zeros((N,),dtype=int)
    for i in range(N):
        s[i] = int(np.random.randint(0,L))
        n[:,i] = sigma*np.random.randn(L)
        y[:,i] = np.roll(x,s[i]) + n[:,i]
    return y,s,n


def make_data_1d(L,N,sigma,b):
    x = generate_signal(L, b)
    y, s, n = generate_observations(x, N, sigma)
    return y, s, n, x
