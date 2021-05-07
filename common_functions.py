import numpy as np


def align_image(x, y, Freqs):
    zz = x[:,0]
    zze = np.zeros((zz.shape[0],3600),dtype=np.complex128)
    for i in range(3600):
        zze[:,i] = zz * np.exp(1j*i*2*np.pi/3600*Freqs)
    align_dis = np.sum((np.abs(zze-y))**2,axis=0)
    ind = np.argmin(align_dis)
    zz = zz * np.exp(1j*2*np.pi*ind/3600*Freqs)
    return np.expand_dims(zz,axis=-1)

def recover_image(z, Phi_ns, Freqs, r_max, Mean):
    L = 2*r_max+1
    N = r_max
    x, y = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    r = np.sqrt(x ** 2 + y ** 2)
    imager = np.zeros((L**2,))
    UU = Phi_ns
    tmp = np.real(UU[:,Freqs == 0] @ z[Freqs == 0,:]) + 2*np.real(UU[:,Freqs != 0] @ z[Freqs != 0,:])
    imager[r.flatten() <= r_max] = imager[r.flatten() <= r_max] + tmp[:,0]
    imager = np.reshape(imager,(L,L)) + Mean
    return imager
