from __future__ import division, print_function

import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng



try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

# try:
#     # Python 2: 'xrange' is the iterative version
#     range = xrange
# except NameError:
#     # Python 3: 'range' is iterative - no need for 'xrange'
#     pass


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))
        # diagS = np.diag(S)
        # np.diag(diagS(1:svp) - 1/mu)

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)
        sv = 10

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            # U, S, V = np.linalg.svd(self.D - Sk + self.mu_inv * Yk, full_matrices=False)
            # diagS = np.diag(S)
            # svp = np.size(np.char.find(diagS > self.mu_inv))
            # if (svp < sv):
            #     sv = min(svp + 1, n)
            # else:
            #     sv = min(svp + round(0.05*n), n)
            # Lk = U[:, 1:svp] * np.diag(diagS[1:svp] - 1/mu) * V[:, 1:svp] 


            
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            # diag1 = np.diag(self.D - Lk + (self.mu_inv * Yk))
            # svp1 = np.size(np.char.find(diag1 > self.mu_inv))
            # if (svp < sv):
            #     sv = min(svp1 + 1, n)
            # else:
            #     sv = min(svp1 + round(0.05*n), n)
            # Sk = np.diag(diag1[1:svp1] - self.mu_inv * self.lmbda)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1
#             if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
#                 print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')

def Clustering(A, n, lamb, mu=1.2, rho = 1.3):
    i = 0
    while (i<3):
        rpca = R_pca(A+np.identity(n), mu, lam)
        X, E = rpca.fit()
        # A = X
        if(X.trace()>n):
            lamb /= 2
        else:
            lamb *= 2
        i += 1
        mu *= rho
    X = np.around(X)
    E = np.around(E)
    X = X.astype(int)
    E = E.astype(int)
    print(X)
    print(E)
    print("Rank of X:", np.linalg.matrix_rank(X))
    print("Rank of E: ", np.linalg.matrix_rank(E))

if __name__=="__main__":
    # A = np.matrix('1, 1, 0, 1; 0, 1, 1, 1; 1, 1, 0, 0; 0, 0, 1, 1')
    # A = np.matrix('1, 1, 0, 1; 1, 1, 0, 0; 0, 0, 1, 1; 0, 0, 1, 1')
    
    # A = np.array([[0, 1, 0, 1],
    #           [1, 0, 1, 0],
    #           [0, 1, 0, 1],
    #           [1, 0, 1, 0]])
    
    
    n = 5
    A = np.random.randint(2, size=(n, n))
    print(A)
    print(np.linalg.matrix_rank(A))
    lam = 1/(32*np.sqrt(n))
    # A = [[1, 1, 0], [0, 1, 1], [0, 1, 0]]
    # rpca = R_pca(A, 1, 2*lam)
    # X, E = rpca.fit()
    # print(X)
    # print(E)
    # print(np.linalg.matrix_rank(X))
    Clustering(A, n, lam)



