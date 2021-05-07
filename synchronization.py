import numpy as np
from tqdm import tqdm


def relative_rotation(x, y, Freqs, L=360):
    zz = x
    zze = np.zeros((zz.shape[0],L),dtype=np.complex128)
    for i in range(L):
        zze[:,i] = zz * np.exp(1j*i/L*2*np.pi*Freqs)
    align_dis = np.sum((np.abs(zze-np.expand_dims(y,axis=-1)))**2,axis=0)
    ind = np.argmin(align_dis)
    return ind / L * 360

def synchronize_2d_tm(y, Freqs, L):
    M, N = y.shape
    num_freqs = Freqs.shape[0]
    est_rotations = np.zeros((N,))
    for n in tqdm(range(N)):
        est_rotations[n] = relative_rotation(y[:num_freqs,1],y[:num_freqs,n],Freqs[:num_freqs],L)

    y_s = np.zeros_like(y)
    for i in range(N):
        y_s[:,i]=y[:,i] * np.exp(-1j*2*np.pi*est_rotations[i]/360*Freqs)
    return y_s, est_rotations


def synchronize_2d(y, Freqs, L):
    M, N = y.shape
    rho = np.zeros((N, N))
    num_freqs = Freqs.shape[0]
    for n in tqdm(range(N)):
        for m in range(n):
            rho[n, m] = relative_rotation(y[:num_freqs,m],y[:num_freqs,n],Freqs[:num_freqs],L)
    for n in range(N):
        for m in range(n, N):
            rho[n, m] = (360 - rho[m, n]) % 360

    H = np.exp(1j*2*np.pi*rho/360)
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


def synchronize_and_match_2d(y, Freqs, P, L):
    M, N = y.shape
    y_s, est_rotations = synchronize_2d(y[:, :P], Freqs, L)
    D = np.zeros((P, N))
    for p in range(P):
        for q in range(N):
            R_nm = np.sum(np.abs(y[:, p] - y[:, q]) ** 2)
            D[p, q] = R_nm

    matching_rotations = np.take_along_axis(np.repeat(np.expand_dims(est_rotations, axis=-1), N, axis=-1), D.argsort(axis=0), axis=0)
    est_rotations_new = np.array(np.concatenate([matching_rotations[1, :P], matching_rotations[0, P:]], axis=-1), dtype=int)

    y_s_new = np.zeros((M, N), dtype=np.complex128)
    for i in range(N):
        y_s_new[:,i]=y[:,i] * np.exp(-1j*2*np.pi*est_rotations_new[i]/360*Freqs)

    return y_s_new, est_rotations_new


def normalize(z):
    z_normalized = z / np.abs(z)
    z_normalized[z_normalized == 0] = 0
    return z_normalized


def synchronize_1d(y, method='ppm',Lambda=1):
    L,N = y.shape
    Y = np.fft.fft(y,axis=0)
    rho = np.zeros((N,N))
    for n in range(N):
        for m in range(n):
            R_nm = np.fft.ifft(Y[:,n] * Y[:,m].conj())
            k = np.argmax(R_nm)
            rho[n,m] = k
    for n in range(N):
        for m in range(n,N):
            rho[n,m] = (L - rho[m,n]) % L

    H = np.exp(1j*2*np.pi*rho/L)
    b = np.random.randn(N) +1j*np.random.randn(N)
    b = b / np.linalg.norm(b)
    max_iter = int(5e2)
    min_step = 1e-3
    n = 0
    step = 1.
    b_prev = np.zeros_like(b)

    while step > min_step and n < max_iter:
        if method == 'ppm':
            b_prev = b
            b = H @ b
            b = normalize(b)
        elif method == 'amp':
            c = Lambda * (H@ b) - Lambda ** 2 * (1 - np.mean(np.abs(b) ** 2)) * b_prev
            b_prev = b
            b = f(np.abs(c)) * (c / np.abs(c))
        else:
            print('Unknown synchronization method, using ppm instead')
            b_prev = b
            b = H @ b
            b = normalize(b)
        step = np.linalg.norm(b-b_prev)
        n = n+1
    est_shifts = np.round(np.angle(b)/(2*np.pi)*L)

    y_s = np.zeros_like(y)
    for i in range(N):
        y_s[:,i]=np.roll(y[:,i],-int(est_shifts[i]))

    return y_s, est_shifts

def synchronize_and_match_1d(y, P):
    L, N = y.shape
    y_s, est_rotations = synchronize_1d(y[:, :P], method='ppm')
    D = np.zeros((P, N))
    for p in range(P):
        for q in range(N):
            R_nm = np.sum(np.abs(y[:, p] - y[:, q]) ** 2)
            D[p, q] = R_nm

    matching_rotations = np.take_along_axis(np.repeat(np.expand_dims(est_rotations, axis=-1), N, axis=-1), D.argsort(axis=0), axis=0)
    est_rotations_new = np.array(np.concatenate([matching_rotations[1, :P], matching_rotations[0, P:]], axis=-1), dtype=int)

    y_s_new = np.zeros((L, N))
    for i in range(N):
        y_s_new[:, i] = np.roll(y[:, i], -int(est_rotations_new[i]))

    return y_s_new, est_rotations_new, y_s, est_rotations


def synchronize_1d_tm(y):
    L,N = y.shape
    Y = np.fft.fft(y,axis=0)
    y_s = np.zeros_like(y)
    s = np.zeros((N,))
    fft_template = np.fft.fft(y[:,0],axis=0)
    for i in range(N):
        R_nm = np.fft.ifft(Y[:,i] * fft_template.conj())
        s[i] = np.argmax(R_nm)
        y_s[:,i] = np.roll(y[:,i], -int(s[i]),axis=0)
    return y_s, s