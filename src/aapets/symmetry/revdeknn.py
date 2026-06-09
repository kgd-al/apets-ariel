# Reimplemented from https://github.com/jmtomczak/popi4sb/blob/master/algorithms/population_optimization_algorithms.py
# Relevant paper: https://www.mdpi.com/2227-9717/9/1/98
#
# With modifications from
# https://github.com/ci-group/revolve/blob/jlo/experiment/learning_CPPN_directed/pyrevolve/algorithms/revdeknn.py
# https://github.com/ci-group/revolve/blob/jlo/experiment/learning_CPPN_directed/pyrevolve/evolution/population/population.py#L375
from typing import Tuple, Any, List

import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor

from ..common.misc.debug import kgd_debug
from .config import Config


class RevDEKNN:
    def __init__(self, eval_fn, config: Config):
        self.calculate_fitnesses = eval_fn
        self.config = config
        self.f, self.cr = config.rev_de_knn_gamma, config.rev_de_knn_cross
        self.clip_min = -config.rev_de_knn_clip
        self.clip_max = config.rev_de_knn_clip

        f = self.f
        r = np.asarray([[1, f, -f],
                        [-f, 1. - f ** 2, f + f ** 2],
                        [f + f ** 2, -f + f ** 2 + f ** 3, 1. - 2. * f ** 2 - f ** 3]])

        self.R = np.expand_dims(r, 0)  # 1 x 3 x 3

        self.nn = KNeighborsRegressor(n_neighbors=config.rev_de_knn_neighborhood)

        self.X = None
        self.E = None

    def _proposal(self, theta: np.ndarray, E=None):
        if self.X is None:
            self.X = theta
            self.E = E
        else:
            if self.X.shape[0] < 10000:
                self.X = np.concatenate((self.X, theta), 0)
                self.E = np.concatenate((self.E, E), 0)

        self.nn.fit(self.X, self.E)

        theta_0 = np.expand_dims(theta, 1)  # B x 1 x D

        indices_1 = np.random.permutation(theta.shape[0])
        indices_2 = np.random.permutation(theta.shape[0])
        theta_1 = np.expand_dims(theta[indices_1], 1)
        theta_2 = np.expand_dims(theta[indices_2], 1)

        tht = np.concatenate((theta_0, theta_1, theta_2), 1)  # B x 3 x D

        y = np.matmul(self.R, tht)

        theta_new = np.concatenate((y[:, 0], y[:, 1], y[:, 2]), 0)

        p_1 = np.random.binomial(1, self.cr, theta_new.shape)
        theta_new = p_1 * theta_new + (1.0 - p_1) * np.concatenate(
            (tht[:, 0], tht[:, 1], tht[:, 2]), 0
        )

        E_pred = self.nn.predict((theta_new))

        ind = np.argsort(E_pred.squeeze())

        return theta_new[ind[: theta.shape[0]]]

    def step(self, theta, E_old, data_old: List[Any]):
        """
        theta: numpy.ndarray 2D with first dimension population size, second dimension # weights
               this is what we are optimizing
        E_old: numpy.ndarray 1D with each item a fitness value for the individual at the same index in the population
        data_old: piggy-back data to sort alongside the rest
        """

        # (1. Generate)
        theta_new = self._proposal(theta, E_old)
        theta_new = np.clip(
            theta_new,
            a_min=self.clip_min,
            a_max=self.clip_max,
        )

        # (2. Evaluate)
        E_new, data_new = self.calculate_fitnesses(
            theta_new,
        )

        # (3. Select)
        theta_cat = np.concatenate((theta, theta_new), 0)
        E_cat = np.concatenate((E_old, E_new), 0)
        data_cat = np.concatenate((data_old, data_new), 0)

        indx = np.argsort(E_cat.squeeze())

        return theta_cat[indx[: theta.shape[0]], :], E_cat[indx[: theta.shape[0]], :], data_cat[indx[: theta.shape[0]]]
