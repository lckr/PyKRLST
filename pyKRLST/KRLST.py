""" Kernel Recursive Least-Squares Tracker Algorithm

    References: 
        M. Lazaro-Gredilla, S. Van Vaerenbergh and I. Santamaria, "A Bayesian
        approach to tracking with kernel recursive least-squares", 2011 IEEE
        Workshop on Machine Learning for Signal Processing (MLSP 2011),
        Beijing, China, September 2011.
        S. Van Vaerenbergh, M. Lazaro-Gredilla, and I. Santamaria, ‘Kernel 
        Recursive Least-Squares Tracker for Time-Varying Regression’, IEEE Trans.
        Neural Netw. Learning Syst., vol. 23, no. 8, pp. 1313–1326, Aug. 2012,
        doi: 10.1109/TNNLS.2012.2200500.

"""


# Authors: Lucas Krauß <lucas.krauss@pm.me>
#
# License: MIT 

import numpy as np
import warnings
from typing import Tuple
from sklearn.gaussian_process.kernels import Kernel


class KRLST:
    """Kernel Recursive Least-Squares Tracker Algorithm

    Maintains a fixed budget via growing and pruning and regularization.
    Assumes a fixed value for the lengthscale, the regularization factor and the signal and noise powers.
    """

    def __init__(
        self, kernel: Kernel, l: float, c: float, M: int, forgetmode: str = "B2P"
    ):
        """[summary]

        Args:
            kernel (Kernel): Kernel object
            l (float): Forgetting factor. l \in [0,1]
            c (float): Noise-to-signal ratio (regularization)
            M (int): Budget, i.e., maximum size of dictionary
            forgetmode (str): Either back-to-prior ('B2P') or uncertainty injection ('UI')
        """
        self._kernel = kernel

        if l < 0 or l > 1:
            raise ValueError("Parameter `l` is out of allowed range of [0,1].")
        self._lambda = l

        self._c = c
        self._M = M

        if not (forgetmode in ["B2P", "UI"]):
            raise ValueError("Parameter `forgetmode` can either be 'B2P' or 'UI'.")
        self._forgetmode = forgetmode

        self._jitter = 1e-10
        self._is_init = False

    def train(self, x: np.ndarray, y: float, t: int):
        """[summary]

        Args:
            x (np.ndarray): Single data point
            y (float): Single regression target
        """
        if not self._is_init:  # Initialize model

            kss = self._kernel(x) + self._jitter
            self.Q = 1 / kss
            self.mu = (y * kss) / (kss + self._c)
            self.Sigma = kss - ((kss ** 2) / (kss + self._c))  # Check this

            self.basis = 0  # Dictionary indicies
            self.Xb = x  # Dictionary
            self.m = 1  # Dict size

            self.nums02ML = y ** 2 / (kss + self._c)
            self.dens02ML = 1
            self.s02 = self.nums02ML / self.dens02ML

            self._is_init = True

        else:  # Update model

            if self._lambda < 1:
                # Forgetting

                if self._forgetmode == "B2P":  # Back-to-prior
                    Kt = self._kernel(self.Xb)
                    self.Sigma = self._lambda * self.Sigma + (1 - self._lambda) * Kt
                    self.mu = np.sqrt(self._lambda) * self.mu

                elif self._forgetmode == "UI":  # Uncertainty injection
                    self.Sigma = self.Sigma / self._lambda
                else:
                    raise ValueError(
                        "Undefined forgetting strategy.\nSupported forgetting strategies are 'B2P' and 'UI'."
                    )

            # Predict new sample
            kbs = self._kernel(self.Xb, np.atleast_2d(x))
            kss = self._kernel(x) + self._jitter

            q = self.Q @ kbs
            ymean = q.T @ self.mu
            gamma2 = kss - kbs.T @ q
            gamma2[gamma2 < 0] = 0

            h = self.Sigma @ q
            sf2 = gamma2 + q.T @ h
            sf2[sf2 < 0] = 0

            sy2 = self._c + sf2

            # Include new sample and add new basis
            Q_old = self.Q.copy()
            p = np.block([[q], [-1]])
            self.Q = np.block(
                [[self.Q, np.zeros((self.m, 1))], [np.zeros((1, self.m)), 0]]
            ) + (1 / gamma2) * (p @ p.T)

            p = np.block([[h], [sf2]])
            self.mu = np.block([[self.mu], [ymean]]) + ((y - ymean) / sy2) * p
            self.Sigma = np.block([[self.Sigma, h], [h.T, sf2]]) - (1 / sy2) * (p @ p.T)
            self.basis = np.block([[self.basis], [t]])
            self.m = self.m + 1
            self.Xb = np.block([[self.Xb], [x]])

            # Estimate s02 via maximum likelihood
            self.nums02ML = self.nums02ML + self._lambda * (y - ymean) ** 2 / sy2
            self.dens02ML = self.dens02ML + self._lambda
            self.s02 = self.nums02ML / self.dens02ML

            # Remove basis if necessary
            if (self.m > self._M) or (gamma2 < self._jitter):

                if gamma2 < self._jitter:
                    if gamma2 < self._jitter / 10:
                        warnings.warn(
                            "Numerical roundoff error is too high. Try increasing jitter noise."
                        )
                    criterium = np.block([np.ones((1, self.m - 1)), 0])
                else:  # MSE pruning
                    errors = (self.Q @ self.mu).reshape(-1) / np.diag(self.Q)
                    criterium = np.abs(errors)

                r = np.argmin(criterium)
                smaller = criterium > criterium[r]

                if r == self.m:  # remove the element we just added
                    self.Q = Q_old
                else:
                    Qs = self.Q[smaller, r]
                    qs = self.Q[r, r]
                    self.Q = self.Q[smaller][:, smaller]
                    self.Q = self.Q - (Qs.reshape(-1, 1) * Qs.reshape(1, -1)) / qs

                self.mu = self.mu[smaller]
                self.Sigma = self.Sigma[smaller][:, smaller]
                self.basis = self.basis[smaller]
                self.m = self.m - 1
                self.Xb = self.Xb[smaller, :]

    def eval(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """[summary]

        Args:
            x (np.ndarray): [description]

        Returns:
            float, float: [description]
        """
        kbs = self._kernel(self.Xb, np.atleast_2d(x))
        mean_est = kbs.T @ self.Q @ self.mu
        sf2 = (
            1
            + self._jitter
            + np.sum(
                kbs * ((self.Q @ self.Sigma @ self.Q - self.Q) @ kbs), axis=0
            ).reshape(-1, 1)
        )
        sf2[sf2 < 0] = 0
        var_est = self.s02 * (self._c + sf2)

        return mean_est, var_est


