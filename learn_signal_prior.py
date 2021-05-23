from make_data import make_data
import numpy as np
import matplotlib.pyplot as plt


def learn_signal_prior():
    Ncopy = 100
    SNR = 0.1
    seed = 1
    image_idx = 5
    use_sPCA = True
    r_max = 64
    Freqs_t = np.zeros((0,))
    Coeff_raw_t = np.zeros((0,1))
    for image_idx in range(5):
        Coeff, Freqs, rad_freqs, Mean, Phi_ns, sigma, Coeff_raw, rotations = make_data(Ncopy=Ncopy,SNR=SNR,seed=seed,image_idx=image_idx, sPCA=use_sPCA)
        Freqs_t = np.concatenate([Freqs_t, Freqs],axis=0)
        Coeff_raw_t = np.concatenate([Coeff_raw_t, Coeff_raw],axis=0)

    Freqs = Freqs_t
    Coeff_raw = Coeff_raw_t
    plt.subplot(211)
    plt.scatter(Freqs, np.abs(Coeff_raw),alpha=.5,marker='+')

    loss_vec = []
    x = np.expand_dims(Freqs, axis=-1)
    y = np.abs(Coeff_raw)
    alpha_range = np.arange(1 / 32, 0.3, 1 / 32)
    # alpha_range = [0,0.125]
    for alpha in alpha_range:
        x0 = 0
        A = np.concatenate([np.exp(-alpha * (x - x0))], axis=-1)
        w = np.linalg.pinv(A) @ y
        loss = np.mean((y - w[0] * np.exp(-alpha * (x - x0))) ** 2)
        loss_vec.append(loss)
        #plt.plot(w[0] * np.exp(-alpha * (np.arange(np.min(Freqs), np.max(Freqs)) - x0)), 'k', alpha=0.1)
    alpha_star = alpha_range[np.argmin(loss_vec)]
    A = np.concatenate([np.exp(-alpha_star * (x - x0))], axis=-1)
    w_opt = np.linalg.pinv(A) @ y
    plt.plot(w[0] * np.exp(-alpha_star * (np.arange(np.min(Freqs), np.max(Freqs)) - x0)), 'r', alpha=0.5)

    plt.xlabel('Angular Frequency')
    plt.ylabel('Absolute Coefficient')
    plt.text(75, 1, r'$C=%.2f e^{%.3f \cdot f}$' % (w_opt, alpha_star), bbox=dict(facecolor='yellow', alpha=0.5))

    plt.subplot(212)
    plt.plot(alpha_range, loss_vec)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('loss')
    plt.text(alpha_star - 0.025, np.mean(loss_vec), r'$\alpha^*=%.3f$' % alpha_star,
             bbox=dict(facecolor='yellow', alpha=0.5))
    plt.tight_layout()
    plt.savefig('2d_figures/fit_signal_prior_.png',bbox_inches = "tight")
    plt.savefig('2d_figures/fit_signal_prior_.eps',bbox_inches = "tight")


def main():
    learn_signal_prior()


if __name__ == '__main__':
    main()
