"""
File contains description of different random fields to generate
Expected Euler Characteristic

Every class - Gaussian related RF, with method to get rho
Gaussian RF has also method to get LKC for stationary isotropic fields
"""

import numpy as np
import scipy
from scipy.stats import norm
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermval
import math
import matplotlib.pyplot as plt
import torch
import os


def get_lkc(dim, L, scale):
    # Computing LKC for isotropic RF
    #alpha = 1 / (4 * scale**2)
    # Book: page 101, and multiply on lambda_2^{i/2}, where
    # lambda_2 is second spatial derivative of C(s, t)
    #
    # LKC paper: Application of Random Fields in Human Brain Mapping
    # J. Cao and K.J.Worsley
    alpha = 1 / (scale**2)
    if dim == 2:
        if isinstance(L, list):
            if len(L) == 2:
                a, b = L
            else:
                a = b = L[0]
        else:
            a = b = L

        lkc = np.array([1, (a + b) * np.sqrt(2 * alpha), a * b * 2 * alpha])
    elif dim == 3:
        if isinstance(L, list):
            if len(L) == 3:
                a, b, c = L
            else:
                a = b = c = L[0]
        else:
            a = b = c = L

        lkc = np.array([
            1, (a + b + c) * np.sqrt(2 * alpha),
            (a * b + b * c + a * c) * 2 * alpha,
            a * b * c * (2 * alpha)**(3 / 2)
        ])
    else:
        raise NotImplementedError("No lkc for dim > 3")
    return lkc


class Gaussian:
    def __init__(self, dim=2, scale=5, L=20):
        # scale are necessary to get LKC

        if dim is None:
            if isinstance(L, list):
                dim = len(L)
            else:
                dim = 2

        self.dim = dim
        self.scale = scale
        self.L = L  #length of RF in square, cube, ... can be list if dims are different

    def get_rho(self, u, D=None):
        # page 135 of the book
        # D is number of components rho(u) to compute

        # 3d
        # u = u.reshape(-1, 1)
        # H = np.hstack([np.ones((len(u), 1)), u, (u**2 - 1)])
        # rho = H * np.hstack([
        #     np.exp(-u**2 / 2) / (2 * np.pi)**(2 / 2),
        #     np.exp(-u**2 / 2) / (2 * np.pi)**(3 / 2),
        #     np.exp(-u**2 / 2) / (2 * np.pi)**(4 / 2)
        # ])

        if D is None:
            D = self.dim + 1

        # 2d
        u = u.reshape(-1, 1)
        # H = np.hstack([np.ones((len(u), 1)), np.ones((len(u), 1)), u])

        # rho_t = H * np.hstack([(1 - norm.cdf(u)),
        #                      np.exp(-u**2 / 2) / (2 * np.pi)**(2 / 2),
        #                      np.exp(-u**2 / 2) / (2 * np.pi)**(3 / 2)])
        # print ('In get_rho(), rho_t', rho_t.shape)
        rho = [1 - norm.cdf(u)]  # rho{0}
        for d in range(1, D):
            # Attention! We use u/2 in hermite, because in python
            # these functions model for physic hermite polynomial,
            # while in Worsley's papers/book they use probabilists' Hermite.
            # To compute it using physics hermite, we need to do u/2.
            # Page 108 (footnote) in the book.
            H = scipy.special.eval_hermite(d - 1, u / 2)
            rho.append((2 * np.pi)**(-(d + 1) / 2) * H * np.exp(-u**2 / 2))
        rho = np.hstack(rho)

        return rho

    def get_eec(self, u):
        # Expected Euler Characteristic
        lkc = get_lkc(dim=self.dim, L=self.L, scale=self.scale)
        rho = self.get_rho(u)
        eec = np.matmul(rho, lkc)

        return eec


class ChiSquared:
    # Rewrite get_rho and get_lkc for any d as a function
    def __init__(self, dim=None, df=1, scale=10, L=20):
        # scale are necessary to get LKC
        if dim is None:
            if isinstance(L, list):
                dim = len(L)
            else:
                dim = 2
        self.dim = dim
        self.scale = scale
        self.df = df
        self.L = L

    def get_rho(self, u, D=None):
        # !!! PAGE 137 of the book
        # and "Application of Random Fields in Human Brain Mapping,
        # J. Cao and K. J. Worsley
        u = u.reshape(-1, 1)
        k = self.df
        if D is None:
            D = self.dim + 1
        rho = []

        # j = 0
        rho.append(1 - scipy.stats.chi2.cdf(u, k))

        # j >= 1
        for j in range(1, D):
            # first_coef = u**((k - j)/2) * np.exp(-u / 2) /\
            #     ((2 * np.pi)**(j / 2) * math.gamma(k / 2) * 2**((k - 2) / 2))
            first_coef = u**((k - j)/2) * np.exp(-u / 2) /\
                ((2 * np.pi)**(j / 2) * math.gamma(k / 2) * 2**((k - j) / 2))

            second_coef = 0
            for l in range(0, int((j - 1) / 2) + 1):
                for m in range(0, j - 1 - 2 * l + 1):
                    if k >= j - m - 2 * l:
                        #old version is scipy.misc.comb
                        second_coef += scipy.special.comb(k - 1, j - 1 - m - 2 * l) *\
                        (-1)**(j-1+m+l)*math.factorial(j-1)/\
                        (math.factorial(m)*math.factorial(l)*2**l)*\
                        u**(m+l)

            rho.append(first_coef * second_coef)

        rho = np.hstack(rho)
        return rho

    def get_eec(self, u):
        # Expected Euler Characteristic
        lkc = get_lkc(dim=self.dim, L=self.L, scale=self.scale)
        rho = self.get_rho(u)
        eec = np.matmul(rho, lkc)

        return eec


def get_threshold(test='rf',
                  dim=None,
                  rf_type='gauss',
                  alpha=0.05,
                  scale=1,
                  df=100,
                  L=50,
                  x=None,
                  n_std=2,
                  is_batch=False):
    if test == 'rf':
        if rf_type == "gauss":
            rf = Gaussian(dim=dim, scale=scale, L=L)
            u = np.linspace(1, 7,
                            10**3)  #-5, 5; but then cannot interp1d properly
        else:
            rf = ChiSquared(dim=dim, scale=scale, df=df, L=L)
            u = np.linspace(df, df * 20, 10**3)
        # Set threshold set
        #u = np.linspace(-2, 5, 10**3)
        u = np.array(u)
        eec = rf.get_eec(u).reshape((-1))

        # Select threshold u which gives specific EEC, using interpolation
        f = interp1d(eec, u, kind='linear', fill_value="extrapolate")
        thresh = f(alpha)
        #print(thresh)
    elif test == 'cochran':
        #cochran C-test
        #https://en.wikipedia.org/wiki/Cochran%27s_C_test
        # Assume batch_size fo x is 1, otherwise we need to x.sum(1)
        # we consider that x is vector of variances.
        # n = number of data points per data series
        # N = number of data series that remain in the data set
        # It is iterative process by removing significant values, while can,
        # and final threshold is selected

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        def get_cochran_thresh(x):
            x = x.reshape(-1)
            thresh = x.max() + 1  # in case cochran cannot find any values
            while True:
                c_stat = x / x.sum()
                N = len(x)  #total number of pixels
                c_critval = (1 + (N - 1) / scipy.stats.f.ppf(
                    q=1 - alpha / N, dfn=(n_std - 1), dfd=(N - 1) * (n_std - 1))\
                )**(-1)

                x = x[c_stat > c_critval]
                if len(x) == 0:
                    break
                else:
                    thresh = c_critval
            return thresh

        # currently not sure how else to accelerate, because for every batch is
        # different thresh
        if is_batch:
            thresh = np.zeros(len(x))
            for i in range(len(x)):
                thresh[i] = get_cochran_thresh(x[i])

            thresh = thresh.reshape((len(thresh), ) + (1, ) *
                                    (len(x.shape) - 1))
        else:
            thresh = get_cochran_thresh(x)

    elif test == 'quantile':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if is_batch:
            thresh = np.quantile(x.reshape(len(x), -1), 1 - alpha, 1)
            thresh = thresh.reshape((len(thresh), ) + (1, ) *
                                    (len(x.shape) - 1))

        else:
            thresh = np.quantile(x, 1 - alpha)

    return thresh


def get_significant(x,
                    test='rf',
                    rf_type='gauss',
                    alpha=0.05,
                    scale=1,
                    df=100,
                    L=50,
                    n_std=1,
                    is_batch=True):
    # if axis 0 of x is not a batch, then is_batch=False,
    # that is important to accelerate get_threshold computation
    thresh = get_threshold(test=test,
                           rf_type=rf_type,
                           alpha=alpha,
                           scale=scale,
                           df=df,
                           L=L,
                           x=x,
                           n_std=n_std,
                           is_batch=is_batch)

    if isinstance(x, torch.Tensor):
        x_thresh = x.clone()
        thresh = torch.tensor(thresh).to(x)  #.repeat((1, ) + x.shape[1:])
    else:
        x_thresh = x.copy()

    x_thresh[x >= thresh] = 1
    x_thresh[x < thresh] = 0  #np.nan?
    return x_thresh  #, u, eec


def plot_rf(
        x,
        mask=None,
        plot_path=None,
        ax=None,
        figsize=(10, 10),
        empty_plot=False,  #True,
        is_mask=False,
        base_cmap='jet'):  #='gist_earth_r'):
    # is_mask means if x is mask or not, different from argument mask
    # if mask is provided, it is plotted on top of x
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    x = x.squeeze()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        plt.sca(ax)

    if is_mask:
        cmap = "bwr"  #"binary"
        interpolation = None
    else:
        cmap = base_cmap  # 'Greys' 'terrain_r' 'gist_earth_r'
        interpolation = "bilinear"

    plt.imshow(x, cmap=cmap, interpolation=interpolation)  #
    if is_mask:
        # fix clim to be always from -1 to 1, when we plot masks and their diff
        plt.clim(-1, 1)
    # else:
    #     #plt.clim(0, 1)
    #     plt.clim(-3, 3)

    if mask is not None:
        #cmap = plt.get_cmap('Greys')
        #cmap.set_bad('r', 1.0)
        if len(x.shape) == 3:
            #rgb image
            # mask = np.transpose(mask, (1, 2, 0))
            # mask = np.repeat(mask, 3, axis=2)

            ###############
            mask = mask.squeeze()

            # Jet color
            # alpha = 0.5
            # heatmap = np.uint8(255 * mask)
            # import matplotlib.cm as cm
            # jet = cm.get_cmap("jet")

            # # Use RGB values of the colormap
            # jet_colors = jet(np.arange(256))[:, :3]
            # jet_heatmap = jet_colors[heatmap]
            # x_masked = jet_heatmap * alpha + x

            alpha = 1
            color = np.zeros(x.shape)
            color[:, :, 0] = 1  #red color
            x_masked = x.copy()  #interpolate here
            x_masked[mask == 1] = color[mask == 1]
            x_masked = (1 - alpha) * x + alpha * x_masked

            plt.imshow(x_masked, cmap='jet',
                       interpolation='bilinear')  #Reds 'none'

        else:
            mask = mask.squeeze()
            alpha = 1
            color = np.zeros(x.shape + (3, ))
            color[:, :, 0] = 1  #red color
            rgb_x = x.copy()  #interpolate here
            rgb_x = 1 - (rgb_x - rgb_x.min()) / (rgb_x.max() - rgb_x.min())
            rgb_x = np.stack([rgb_x] * 3, axis=2)
            x_masked = rgb_x.copy()
            x_masked[mask == 1] = color[mask == 1]
            x_masked = (1 - alpha) * rgb_x + alpha * x_masked
            plt.imshow(x_masked, cmap='jet',
                       interpolation='bilinear')  #Reds 'none'

            #x_masked = np.ma.masked_where(mask < 1, x)  #values from x
            # plt.imshow(x_masked, cmap='Reds',
            #            interpolation='bilinear')  #Reds 'none'

            #save mask for future adjustments
            os.makedirs("./tmp/masks/", exist_ok=True)
            np.save(
                './tmp/masks/%s.npy' %
                (os.path.basename(plot_path).split('.')[0]), mask)

    if True:  #empty_plot:
        #plt.axis('off')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    else:
        plt.colorbar()

    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()


if __name__ == '__main__':
    G = Gaussian(scale=10, L=32)
    u = np.array([0.1])  #, 0.5, 1])
    rho = G.get_rho(u)  #
    print(rho)

    thresh = get_threshold(
        test='rf',
        rf_type='gauss',  #gauss chisq
        alpha=0.05 / 95,
        scale=10,
        df=1,
        L=64)
    print(thresh)

    thresh = get_threshold(
        test='rf',
        rf_type='chisq',  #gauss chisq
        alpha=0.05,
        scale=10,
        df=1,
        L=32)
    print(thresh)
    #return thresh

    # threshold, u, eec = get_threshold(
    #     test='rf',
    #     rf_type='gauss',  #gauss chisq
    #     alpha=0.05,
    #     scale=10,
    #     df=1,
    #     L=32)
    # print("Threshold: %f" % threshold)

    # plt.plot(u, eec)
    # plt.savefig('../plot_eec_chisq.png')
    # plt.close()
