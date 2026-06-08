# Reimplemented from https://github.com/jmtomczak/popi4sb/blob/master/algorithms/population_optimization_algorithms.py
# Relevant paper: https://www.mdpi.com/2227-9717/9/1/98
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor


class RevDEKNN:
    def __init__(self, eval_fn, f=.5, p=.9, clip=(-1, 1)):
        self.f, self.p = f, p
        self.dist = stats.norm
        self.clip = clip

        r = np.asarray([[1, f, -f],
                        [-f, 1. - f ** 2, f + f ** 2],
                        [f + f ** 2, -f + f ** 2 + f ** 3, 1. - 2. * f ** 2 - f ** 3]])

        self.R = np.expand_dims(r, 0)  # 1 x 3 x 3

        self.nn = KNeighborsRegressor(n_neighbors=3)

        self.X = None
        self.E = None

    def proposal(self, theta, e=None):

        if self.X is None:
            self.X = theta
            self.E = e
        else:
            if self.X.shape[0] < 10000:
                self.X = np.concatenate((self.X, theta), 0)
                self.E = np.concatenate((self.E, e), 0)

        self.nn.fit(self.X, self.E)

        theta_0 = np.expand_dims(theta, 1)  # B x 1 x D

        indices_1 = np.random.permutation(theta.shape[0])
        indices_2 = np.random.permutation(theta.shape[0])
        theta_1 = np.expand_dims(theta[indices_1], 1)
        theta_2 = np.expand_dims(theta[indices_2], 1)

        tht = np.concatenate((theta_0, theta_1, theta_2), 1)  # B x 3 x D

        y = np.matmul(self.R, tht)

        theta_new = np.concatenate((y[:, 0], y[:, 1], y[:, 2]), 0)

        p_1 = np.random.binomial(1, self.p, theta_new.shape)
        theta_new = p_1 * theta_new + (1. - p_1) * np.concatenate((tht[:,0], tht[:,1], tht[:,2]), 0)

        e_pred = self.nn.predict(theta_new)

        ind = np.argsort(e_pred.squeeze())

        return theta_new[ind[:theta.shape[0]]]

    def step(self, theta, e_old, x_obs, mod, params):
        # (1. Generate)
        theta_new = self.proposal(theta, e_old)
        theta_new = np.clip(theta_new, a_min=self.clip[0], a_max=self.clip[1])

        # (2. Evaluate)
        e_new = calculate_fitness(x_obs, theta_new, mod, params)

        # (3. Select)
        theta_cat = np.concatenate((theta, theta_new), 0)
        e_cat = np.concatenate((e_old, e_new), 0)

        indx = np.argsort(e_cat.squeeze())

        return theta_cat[indx[:theta.shape[0]], :], e_cat[indx[:theta.shape[0]], :]
