import numpy as np
from time import time
from make_data import generate_signal_covariance


def EM_General_Prior(Coeff, Freqs, sigma, Ndir, rho_p, BW, gamma, Coeff_raw, use_signal_prior, uniform=False, return_dist=False):
    TT = time()
    niter = 1000
    tol = 1e-5
    L = len(Freqs)
    lf = np.sum(Freqs == 0)
    x_init = np.concatenate([np.random.randn(lf,1),np.random.randn(L-lf,1)+1j*np.random.randn(L-lf,1)],axis=0)
    x = x_init
    data = Coeff
    data[lf+1:L,:] = data[lf+1:L,:]*np.sqrt(2)
    #data_prior = Coeff_raw
    # data_prior = np.expand_dims(5*np.exp(-np.asarray(Freqs)/8),axis=-1)  #Changed 10/12/2020
    data_prior = np.expand_dims(4*np.exp(-np.asarray(Freqs)/8),axis=-1)  #Changed according to learn_signa_prior
    data_prior[lf+1:L,:] = data_prior[lf+1:L,:]*np.sqrt(2)
    Cov = np.diag(np.abs(data_prior[:,0])**2)
    # rho = 1/BW * np.ones((BW,1))
    # x = np.mean(data,axis=1)
    Freqs = np.expand_dims(Freqs,axis=-1)
    iter = 0
    P = Coeff.shape[-1]
    WhiteningMatrix = np.linalg.inv(P*np.eye(L) + sigma ** 2 * np.linalg.inv(Cov))
    x = np.expand_dims(np.mean(data,axis=-1),axis=-1) ###########################
    x = np.asarray(x,dtype=np.complex128)
    data = np.asarray(data,dtype=np.complex128)
    L = len(rho_p)
    I = np.argmax(rho_p)
    rho_p = np.roll(rho_p, -(I - int(L / 2)))
    rho_p = rho_p[int(L / 2) - int(BW / 2):int(L / 2) + int(np.ceil(BW / 2))]
    rho_p = rho_p / np.sum(rho_p)

    if gamma > 0:
        rho = rho_p ######################################
    else:
        rho = rho_p#1/BW * np.ones((BW,1))
    for i in range(niter):
        iter += 1
        # print(iter)
        x_new, rho_new = EM_Prior(x, rho, data, sigma, Ndir, BW, rho_p, gamma, Freqs, WhiteningMatrix, use_signal_prior)
        # x_new, rho_new = data, rho

        # dis = im_dist(x_new, x, Freqs)
        # dis = 1
        
        dis = im_dist2(x_new, x, Freqs)
        #dis = im_dist(x_new, x, Freqs)
        # print('%f' % (time() - TT))

        if dis < tol:
            break
        x = x_new
        if not uniform:
            rho = rho_new

    x[lf+1:L,:] = x[lf+1:L,:] / np.sqrt(2)
    t_em = time() - TT

    if return_dist:
        return x, iter, t_em, rho
    else:
        return x, iter, t_em

def EM_Prior(x, rho, data, sigma, R, BW, rho_p, gamma, Freqs, WhiteningMatrix, use_signal_prior=False):
    N,M = data.shape
    Rot = np.exp(1j*2*np.pi/R*Freqs,dtype=np.complex128)
    xtmp = x * np.exp(-1j*2*np.pi/R*(Freqs)*int(np.floor(BW/2)),dtype=np.complex128)
    T = np.zeros((BW,M))

    # TT = time()
    for k in range(BW):
        T[k,:] = - np.sum((np.abs(xtmp-data)**2),axis=0)
        xtmp = xtmp * Rot
    # print('%f'%(time() - TT))
    T = T / 2/(sigma ** 2)
    T = T - np.max(T, axis=0)
    W = np.exp(T)
    W = W * rho
    W[:, np.sum(W, axis=0) == 0] += 1e-9 ##############################
    W = W / np.sum(W, axis=0)
    if not use_signal_prior:
        Rl = np.exp(
            -1j * 2 * np.pi / R * Freqs * np.expand_dims(np.arange(-int(np.floor(BW/2)),int(np.floor(BW/2)),1), axis=-1).T)
        fftx_new = np.expand_dims(np.sum(Rl * (data @ W.T), axis=1) / M,axis=-1)

    else:
        Rl = np.exp(-1j * 2 * np.pi / R * Freqs *
                    np.expand_dims(np.arange(-int(np.floor(BW / 2)), int(np.floor(BW / 2)), 1), axis=-1).T)
        fftx_new = np.expand_dims(WhiteningMatrix @ np.sum(Rl * (data @ W.T), axis=1), axis=-1)


    Wmean = np.expand_dims(np.mean(W, axis=1), axis=-1)
    rho_new = (Wmean + gamma * rho_p) / np.sum(Wmean + gamma * rho_p)

    return np.asarray(fftx_new,dtype=np.complex128), np.asarray(rho_new,dtype=np.complex128)

def im_dist(x,y,Freqs):
    Ndir = 3600
    ydd = np.ones((Ndir,))*np.real(np.sum(np.conj(y) * y))
    zz = x
    zze = (zz @ np.ones((1,Ndir))) * np.exp(1j*Freqs*2*np.pi/Ndir*np.arange(Ndir))
    DIS = np.sum(np.real(np.conj(zze) * zze),axis=0) + ydd + - 2*np.real(np.transpose(np.conj(zze)) @ y[:,0])
    Dis = np.min(DIS)
    return np.sqrt(np.max([0,Dis]))


def im_dist2(x,y,Freqs):
    return np.sqrt(np.sum(np.abs(x-y)**2))

# -------------- 1D ----------------- #
def synch_est(x_est,x):
    if len(x_est.shape) < 2:
        x_est = np.expand_dims(x_est,axis=-1)
    if len(x.shape) < 2:
        x = np.expand_dims(x,axis=-1)
    return np.roll(x_est, -int(np.argmax(np.fft.ifft(np.fft.fft(x_est,axis=0) * np.fft.fft(x,axis=0).conj(),axis=0))),axis=0)

def relative_error_1d(x_est,x):
    if len(x_est.shape) < 2:
        x_est = np.expand_dims(x_est,axis=-1)
    if len(x.shape) < 2:
        x = np.expand_dims(x,axis=-1)
    x_est = synch_est(x_est,x)
    return np.sqrt(sum(np.abs(x_est - x)**2)/np.sum(np.abs(x)**2))

def em_1d(X, sigma, num_iter, tol, x_init, b=None, rho_prior=None, uniform=False,x_true=None):
    L,N = X.shape
    rho = 1/L*np.ones((L,1))
    if rho_prior is not None:
        BW = L
        rho = 1/BW*np.ones((BW,1))

    fftx = np.fft.fft(x_init,axis=0)
    fftX = np.fft.fft(X,axis=0)
    sqnormX = np.repeat(np.expand_dims(np.sum(np.abs(X)**2,axis=0),axis=0),L,axis=0)
    iter_cnt = 0
    if b is not None:
        Sigma_x = generate_signal_covariance(L,b)
        WhiteningMatrix = np.linalg.inv(N*np.eye(L) + (sigma ** 2)*np.linalg.inv(Sigma_x))
    else:
        WhiteningMatrix = None

    rel_err = []
    for iter in range(num_iter):
        iter_cnt += 1
        if uniform:
            fftx_new = EM_Uniform_iteration(fftx, fftX, sqnormX, sigma, WhiteningMatrix)
            rho_new = rho
        else:
            if rho_prior is None:
                fftx_new, rho_new = EM_iteration(fftx, rho, fftX, sqnormX, sigma, WhiteningMatrix)
            else:
                fftx_new, rho_new = EM_iteration_Prior(fftx, rho, fftX, sqnormX, sigma, rho_prior[0], rho_prior[1], WhiteningMatrix)
                #fftx_new, rho_new = EM_RSS_iterations(fftx, rho, fftX, sqnormX, sigma, BW, rho_prior[0], rho_prior[1], WhiteningMatrix)
        if relative_error_1d(np.fft.ifft(fftx,axis=0),np.fft.ifft(fftx_new,axis=0)) < tol:
            break
        fftx = fftx_new
        rho = rho_new

        if x_true is not None:
            rel_err.append(relative_error_1d(np.fft.ifft(fftx,axis=0), x_true))

    # if x_true is not None:
    #     plt.plot(rel_err)
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Relative Error')
    # #     plt.show()
    #
    if x_true is not None:
        return np.real(np.fft.ifft(fftx,axis=0)), np.real(rho), iter_cnt#, rel_err
    else:
        return np.real(np.fft.ifft(fftx,axis=0)), np.real(rho), iter_cnt


def EM_Uniform_iteration(fftx, fftX, sqnormX, sigma, WhiteningMatrix=None):
    C = np.fft.ifft(np.conj(fftx) * fftX,axis=0)
    T = (2*C - sqnormX)/(2*(sigma**2))
    T = T - np.max(T,axis=0)
    W = np.exp(T)
    W = W / np.sum(W,axis=0)
    if WhiteningMatrix is None:
        fftx_new = np.expand_dims(np.mean(np.conj(np.fft.fft(W,axis=0)) * fftX,axis=1),axis=-1)
    else:
        fftx_new = np.fft.fft(WhiteningMatrix @ np.fft.ifft(np.expand_dims(np.sum(np.conj(np.fft.fft(W,axis=0)) * fftX,axis=1),axis=-1),axis=0),axis=0)

    return fftx_new


def EM_iteration(fftx, rho, fftX, sqnormX, sigma, WhiteningMatrix=None):
    C = np.fft.ifft(np.conj(fftx) * fftX,axis=0)
    T = (2*C - sqnormX)/(2*(sigma**2))
    T = T - np.max(T,axis=0)
    W = np.exp(T)
    W = W * rho
    W = W / np.sum(W,axis=0)
    if WhiteningMatrix is None:
        fftx_new = np.expand_dims(np.mean(np.conj(np.fft.fft(W,axis=0)) * fftX,axis=1),axis=-1)
    else:
        fftx_new = np.fft.fft(WhiteningMatrix @ np.fft.ifft(np.expand_dims(np.sum(np.conj(np.fft.fft(W,axis=0)) * fftX,axis=1),axis=-1),axis=0),axis=0)

    rho_new = np.expand_dims(np.mean(W,axis=1)/np.sum(np.mean(W,axis=1)),axis=-1)
    return fftx_new, rho_new

def EM_iteration_Prior(fftx, rho, fftX, sqnormX, sigma,q,gamma, WhiteningMatrix=None):
    C = np.fft.ifft(np.conj(fftx) * fftX,axis=0)
    T = (2*C - sqnormX)/(2*(sigma**2))
    T = T - np.max(T,axis=0)
    W = np.exp(T)
    W = W * rho
    W = W / np.sum(W,axis=0)
    if WhiteningMatrix is None:
        fftx_new = np.expand_dims(np.mean(np.conj(np.fft.fft(W,axis=0)) * fftX,axis=1),axis=-1)
    else:
        fftx_new = np.fft.fft(WhiteningMatrix @ np.fft.ifft(np.expand_dims(np.sum(np.conj(np.fft.fft(W,axis=0)) * fftX,axis=1),axis=-1),axis=0),axis=0)

    I = np.argmax(np.fft.ifft(np.fft.fft(np.mean(W, axis=1).T,axis=0) * np.conj(np.fft.fft(q,axis=0)).T,axis=0))
    q = np.roll(q, I)
    rho_new = np.expand_dims((np.mean(W,axis=1)+gamma*q)/np.sum(np.mean(W,axis=1)+gamma*q),axis=-1)
    return fftx_new, rho_new


def EM_RSS_iterations(fftx, rho, fftX, sqnormX, sigma, BW, q, gamma, WhiteningMatrix=None):
    L, N = fftX.shape
    I = np.argmax(q)
    q = np.roll(q,-(I-int(L/2)))
    q = q[int(L/2)-int(BW/2):int(L/2)+int(np.ceil(BW/2))]
    q = q[::-1]
    R = L   # Rotation resolution in 1-d corresponds to a shift by one
    Rot = np.expand_dims(np.exp(-1j*2*np.pi/R*np.arange(L)),axis=-1)
    xtmp = fftx * np.expand_dims(np.exp(1j*2*np.pi/R*(np.arange(L))*int(np.floor(BW/2))),axis=-1)
    T = np.zeros((BW,N))
    for k in range(BW):
        T[k,:] = - np.sum((np.abs(xtmp-fftX)**2),axis=0)
        xtmp = xtmp * Rot

    T = T / 2/(sigma ** 2)/L
    T = T - np.max(T, axis=0)
    W = np.exp(T)
    W = W * rho
    W = W / np.sum(W, axis=0)
    if WhiteningMatrix is None:
        Rl = np.exp(
            -1j * 2 * np.pi / R * np.expand_dims(np.arange(L), axis=-1) * np.expand_dims(np.arange(int(np.floor(BW/2)),-int(np.ceil(BW/2)),-1), axis=-1).T)
        fftx_new = np.expand_dims(np.sum(Rl * (fftX @ W.T), axis=1) / (N),axis=-1)

    else:
        Rl = np.exp(-1j * 2 * np.pi / R * np.expand_dims(np.arange(L), axis=-1) *
                    np.expand_dims(np.arange(int(np.floor(BW / 2)), -int(np.ceil(BW / 2)), -1), axis=-1).T)
        fftx_new = np.expand_dims(WhiteningMatrix @ np.sum(Rl * (fftX @ W.T), axis=1), axis=-1)

    if gamma != 0:
        I = int(np.floor(BW/2)) - np.argmax(np.mean(W, axis=1)) -1
        W = np.roll(W, -I, axis=0)
        fftx_new = fftx_new * np.expand_dims(np.exp(1j*2*np.pi/R*(np.arange(L))*I),axis=-1)

    rho_new = np.expand_dims((np.mean(W,axis=1)+gamma*q)/np.sum(np.mean(W,axis=1)+gamma*q),axis=-1)
    return fftx_new, rho_new
