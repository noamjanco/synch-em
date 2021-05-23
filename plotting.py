from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def save_relerr_plot(paths, labels):
    for path, label in zip(paths,labels):
        filename_list = [path + f for f in listdir(path) if isfile(join(path, f))]
        results = pd.concat([pd.read_pickle(filename) for filename in filename_list])
        snr_range = np.sort(results.SNR.unique())
        mean_err = np.zeros_like(snr_range)
        mean_num_iter = np.zeros_like(snr_range)
        mean_t = np.zeros_like(snr_range)


        for i,SNR in enumerate(snr_range):
            mean_err[i] = np.nanmean(results[results.SNR == SNR].err)
            mean_num_iter[i] = np.nanmean(results[results.SNR == SNR].num_iter)
            mean_t[i] = np.nanmean(results[results.SNR == SNR].t)

        plt.loglog(snr_range, mean_err,'-+',label=label)
        plt.ylabel('Mean Error')
        plt.xlabel('SNR')
        plt.legend()

    name = ''
    for label in labels:
        name += label + '_'
    name += 'N_%d_sPca_%d'%(results.iloc[0].N,results.iloc[0].use_sPCA)
    name = name.replace(' ','_')
    if not os.path.exists('2d_figures'):
        os.mkdir('2d_figures')
    plt.savefig('2d_figures/%s.png'%name)
    plt.savefig('2d_figures/%s.eps'%name)

    plt.clf()

def save_relerr_plot_1d(paths, labels):
    for path, label in zip(paths,labels):
        filename_list = [path + f for f in listdir(path) if isfile(join(path, f))]
        results = pd.concat([pd.read_pickle(filename) for filename in filename_list])
        sigma_range = np.sort(results.sigma.unique())
        L = results.iloc[0].L
        snr_range = 1 / (sigma_range ** 2)
        mean_err = np.zeros_like(snr_range)
        mean_num_iter = np.zeros_like(snr_range)
        mean_t = np.zeros_like(snr_range)


        for i,sigma in enumerate(sigma_range):
            mean_err[i] = np.nanmean(results[results.sigma == sigma].err)
            mean_num_iter[i] = np.nanmean(results[results.sigma == sigma].num_iter)
            mean_t[i] = np.nanmean(results[results.sigma == sigma].t)

        plt.loglog(snr_range, mean_err,'-+',label=label)
        plt.ylabel('Error')
        plt.xlabel('SNR')
        plt.legend()

    name = ''
    for label in labels:
        name += label + '_'
    name += 'relerr_L_%d_b_%d_N_%d'%(results.iloc[0].L,results.iloc[0].b,results.iloc[0].N)
    name = name.replace(' ','_')
    if not os.path.exists('1d_figures/'):
        os.mkdir('1d_figures/')
    plt.savefig('1d_figures/%s.png'%name)
    plt.savefig('1d_figures/%s.eps'%name)

    plt.clf()

def save_relerr_numiter_time_plot(paths, labels):
    markers = ['o','^','+','s']
    ind = 0
    for path, label in zip(paths,labels):
        filename_list = [path + f for f in listdir(path) if isfile(join(path, f))]
        results = pd.concat([pd.read_pickle(filename) for filename in filename_list])
        snr_range = np.sort(results.SNR.unique())
        mean_err = np.zeros_like(snr_range)
        mean_num_iter = np.zeros_like(snr_range)
        mean_t = np.zeros_like(snr_range)

        for i,SNR in enumerate(snr_range):
            mean_err[i] = np.nanmean(results[results.SNR == SNR].err)
            mean_num_iter[i] = np.nanmean(results[results.SNR == SNR].num_iter)
            mean_t[i] = np.nanmean(results[results.SNR == SNR].t)

        plt.subplot(311)
        plt.loglog(snr_range, mean_err,'-'+markers[ind],label=label)
        plt.ylabel('Error')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=len(labels), mode="expand", borderaxespad=0.)
        #plt.legend()

        plt.subplot(312)
        if ind != 0:
           plt.semilogx(snr_range, mean_num_iter,'-'+markers[ind],label=label)
        else:
           plt.semilogx([], [])
        # plt.semilogx(snr_range, mean_num_iter,'-'+markers[ind],label=label)
        plt.ylabel('# Iterations')
        #plt.legend()

        plt.subplot(313)
        plt.semilogx(snr_range, mean_t,'-'+markers[ind],label=label)
        plt.xlabel('SNR')
        plt.ylabel('Run time [sec]')
        #plt.legend()
        ind += 1
    name = ''
    for label in labels:
        name += label + '_'
    name += 'N_%d_sPca_%d'%(results.iloc[0].N,results.iloc[0].use_sPCA)
    name = name.replace(' ','_')
    plt.savefig('2d_figures/%s_.png'%name,bbox_inches = "tight")
    plt.savefig('2d_figures/%s.eps'%name,bbox_inches = "tight")

    plt.clf()


def save_relerr_numiter_time_plot2(paths, labels):
    markers = ['o','^','+','s']
    ind = 0
    for path, label in zip(paths,labels):
        filename_list = [path + f for f in listdir(path) if isfile(join(path, f))]
        results = pd.concat([pd.read_pickle(filename) for filename in filename_list])
        snr_range = np.sort(results.SNR.unique())
        mean_err = np.zeros_like(snr_range)
        mean_num_iter = np.zeros_like(snr_range)
        mean_t = np.zeros_like(snr_range)

        for i,SNR in enumerate(snr_range):
            if ind == 0:
                mean_err[i] = np.nanmean(results[results.SNR == SNR].err_synch)
                mean_t[i] = np.nanmean(results[results.SNR == SNR].t_synch)
            else:
                mean_err[i] = np.nanmean(results[results.SNR == SNR].err)
                mean_t[i] = np.nanmean(results[results.SNR == SNR].t)
            mean_num_iter[i] = np.nanmean(results[results.SNR == SNR].num_iter)

        plt.subplot(311)
        plt.loglog(snr_range, mean_err,'-'+markers[ind],label=label)
        plt.ylabel('Error')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=len(labels), mode="expand", borderaxespad=0.)
        #plt.legend()

        plt.subplot(312)
        if ind != 0:
           plt.semilogx(snr_range, mean_num_iter,'-'+markers[ind],label=label)
        else:
           plt.semilogx([], [])
        # plt.semilogx(snr_range, mean_num_iter,'-'+markers[ind],label=label)
        plt.ylabel('# Iterations')
        #plt.legend()

        plt.subplot(313)
        plt.semilogx(snr_range, mean_t,'-'+markers[ind],label=label)
        plt.xlabel('SNR')
        plt.ylabel('Run time [sec]')
        #plt.legend()
        ind += 1
    name = ''
    for label in labels:
        name += label + '_'
    name += 'N_%d_sPca_%d'%(results.iloc[0].N,results.iloc[0].use_sPCA)
    name = name.replace(' ','_')
    plt.savefig('2d_figures/%s_.png'%name,bbox_inches = "tight")
    plt.savefig('2d_figures/%s.eps'%name,bbox_inches = "tight")

    plt.clf()

def save_plot_gamma(paths, labels):
    markers = ['o','^','+','s']
    ind = 0
    for path, label in zip(paths,labels):
        filename_list = [path + f for f in listdir(path) if isfile(join(path, f))]
        results = pd.concat([pd.read_pickle(filename) for filename in filename_list])
        snr_range = np.sort(results.SNR.unique())
        mean_err = np.zeros_like(snr_range)
        mean_num_iter = np.zeros_like(snr_range)
        mean_t = np.zeros_like(snr_range)
        print(label)
        print(results[results.SNR == 0.1].err)
        for i,SNR in enumerate(snr_range):
            mean_err[i] = np.nanmean(results[results.SNR == SNR].err)
            mean_num_iter[i] = np.nanmean(results[results.SNR == SNR].num_iter)
            mean_t[i] = np.nanmean(results[results.SNR == SNR].t)

        plt.subplot(211)
        plt.loglog(snr_range, mean_err,'-'+markers[ind],label=label)
        plt.ylabel('Error')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=len(labels), mode="expand", borderaxespad=0.)


        plt.subplot(212)
        plt.semilogx(snr_range, mean_num_iter,'-'+markers[ind],label=label)
        plt.ylabel('# Iterations')

        ind += 1
    name = ''
    for label in labels:
        name += label + '_'
    name += 'N_%d_sPca_%d'%(results.iloc[0].N,results.iloc[0].use_sPCA)
    name = name.replace(' ','_')
    plt.savefig('2d_figures/%s_.png'%name,bbox_inches = "tight")
    plt.savefig('2d_figures/%s.eps'%name,bbox_inches = "tight")

    plt.clf()
