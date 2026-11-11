"""
Shared toy model used by both notebooks.

Latent variables
    (z1, z2) ~ N(mu, Sigma),  Sigma = [[s1^2, rho s1 s2], [rho s1 s2, s2^2]]
Observation model
    x | z1, z2 ~ N(phi(z1, z2), sigma_x^2),  phi(z1, z2) = z1^2 + z2^2

Everything is kept deliberately small (two scalars) so that every quantity the
notebooks estimate by Monte Carlo can also be computed "exactly", either in
closed form or by Gauss-Hermite quadrature to machine-level accuracy. That gives
us a ground truth to measure bias, variance and MSE against.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

LOG_2PI = np.log(2.0 * np.pi)


@dataclass(frozen=True)
class Model:
    mu1: float = 0.0
    s1: float = 1.0
    mu2: float = 0.0
    s2: float = 1.0
    rho: float = 0.5
    sigma_x: float = 1.0
    x: float = 1.0  # the observation x_t

    # ------------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------------
    @property
    def mean(self):
        return np.array([self.mu1, self.mu2])

    @property
    def cov(self):
        c = self.rho * self.s1 * self.s2
        return np.array([[self.s1**2, c], [c, self.s2**2]])

    def cond_mean(self, z1):
        """E[z2 | z1]."""
        return self.mu2 + self.rho * self.s2 * (np.asarray(z1) - self.mu1) / self.s1

    @property
    def cond_std(self):
        """sd(z2 | z1) -- does not depend on z1 for a Gaussian."""
        return self.s2 * np.sqrt(1.0 - self.rho**2)

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------
    def sample_z1(self, n, rng):
        return self.mu1 + self.s1 * rng.standard_normal(n)

    def sample_z2_given_z1(self, z1, m, rng):
        """For each z1[i], draw m values of z2 | z1[i]. Returns shape (len(z1), m)."""
        z1 = np.atleast_1d(z1)
        eps = rng.standard_normal((z1.size, m))
        return self.cond_mean(z1)[:, None] + self.cond_std * eps

    def sample_joint(self, n, rng):
        """Direct sampling from the joint via a Cholesky factor. Returns (z1, z2)."""
        L = np.linalg.cholesky(self.cov)
        z = self.mean + rng.standard_normal((n, 2)) @ L.T
        return z[:, 0], z[:, 1]

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    @staticmethod
    def phi(z1, z2):
        return z1**2 + z2**2

    def log_lik(self, z1, z2):
        """log p(x | z1, z2) -- broadcasts over z1, z2."""
        r = (self.x - self.phi(z1, z2)) / self.sigma_x
        return -0.5 * LOG_2PI - np.log(self.sigma_x) - 0.5 * r**2

    def lik(self, z1, z2):
        return np.exp(self.log_lik(z1, z2))

    # ------------------------------------------------------------------
    # Part 1 references: the *linear* target  E_{z1, z2}[ log p(x | z1, z2) ]
    # ------------------------------------------------------------------
    def cond_expected_log_lik(self, z1):
        """
        g(z1) = E[ log p(x | z1, z2) | z1 ]  in closed form.

        With a = x - z1^2 and z2 | z1 ~ N(m, s^2):
            E[(a - z2^2)^2] = a^2 - 2a E[z2^2] + E[z2^4],
            E[z2^2] = m^2 + s^2,  E[z2^4] = m^4 + 6 m^2 s^2 + 3 s^4.
        """
        z1 = np.asarray(z1, dtype=float)
        a = self.x - z1**2
        m = self.cond_mean(z1)
        s2 = self.cond_std**2
        e2 = m**2 + s2
        e4 = m**4 + 6 * m**2 * s2 + 3 * s2**2
        sq = a**2 - 2 * a * e2 + e4
        return -0.5 * LOG_2PI - np.log(self.sigma_x) - 0.5 * sq / self.sigma_x**2

    def exact_expected_log_lik(self):
        """
        E[log p(x | z1, z2)] exactly. g(z1) is a degree-4 polynomial in z1, so
        Gauss-Hermite quadrature with >= 3 nodes integrates it with no error.
        """
        return self._outer_expectation(self.cond_expected_log_lik, n_nodes=10)

    def variance_components(self):
        """
        Exact law-of-total-variance split of  ell = log p(x | z1, z2):

            sigma2_total   = Var(ell)
            sigma2_between = Var( E[ell | z1] )   (variation explained by z1)
            sigma2_within  = E[ Var(ell | z1) ]   (residual variation from z2)

        ell is a degree-4 polynomial in (z1, z2), so ell^2 has degree 8 and
        Gauss-Hermite with 10 nodes in each direction is exact.
        """
        t, w = hermegauss(10)
        w = w / np.sqrt(2.0 * np.pi)
        z1 = self.mu1 + self.s1 * t
        z2 = self.cond_mean(z1)[:, None] + self.cond_std * t[None, :]
        ell = self.log_lik(z1[:, None], z2)
        mean = np.sum(w[:, None] * w[None, :] * ell)
        total = np.sum(w[:, None] * w[None, :] * ell**2) - mean**2
        g = self.cond_expected_log_lik(z1)
        between = np.sum(w * g**2) - np.sum(w * g) ** 2
        return {"total": total, "between": between, "within": total - between}

    # ------------------------------------------------------------------
    # Part 2 references: the *nested* target  E_{z1}[ log E_{z2|z1}[ p(x | z1, z2) ] ]
    # ------------------------------------------------------------------
    def _inner_log_moment(self, z1, power, n_nodes):
        """log E[ p(x | z1, z2)^power | z1 ] by Gauss-Hermite over z2, in log space.

        Working in log space matters: for |z1| large, I(z1) underflows to 0 in
        floating point while log I(z1) is a perfectly ordinary number.
        """
        t, w = hermegauss(n_nodes)
        log_w = np.log(w / np.sqrt(2.0 * np.pi))  # weights of a standard normal
        z1 = np.atleast_1d(np.asarray(z1, dtype=float))
        z2 = self.cond_mean(z1)[:, None] + self.cond_std * t[None, :]
        a = power * self.log_lik(z1[:, None], z2) + log_w[None, :]
        a_max = a.max(axis=1, keepdims=True)
        return (a_max + np.log(np.exp(a - a_max).sum(axis=1, keepdims=True)))[:, 0]

    def cond_log_marginal_lik(self, z1, n_nodes=150):
        """log I(z1),  with  I(z1) = E[ p(x | z1, z2) | z1 ]."""
        return self._inner_log_moment(z1, 1, n_nodes)

    def cond_marginal_lik(self, z1, n_nodes=150):
        """I(z1) = E[ p(x | z1, z2) | z1 ]."""
        return np.exp(self.cond_log_marginal_lik(z1, n_nodes))

    def nested_target(self, n_outer=80):
        """J = E_{z1}[ log I(z1) ] by nested quadrature."""
        return self._outer_expectation(self.cond_log_marginal_lik, n_outer)

    def nested_bias_constant(self, n_outer=80):
        """
        Leading-order bias of the NMC estimator:  E[J_hat] - J ~ -C / M, with
            C = E_{z1}[ Var(p | z1) / (2 I(z1)^2) ]   (second-order delta method).
        """
        def h(z1):
            log_i1 = self._inner_log_moment(z1, 1, 150)
            log_i2 = self._inner_log_moment(z1, 2, 150)
            return 0.5 * np.expm1(log_i2 - 2.0 * log_i1)  # (E p^2 - I^2) / (2 I^2)

        return self._outer_expectation(h, n_outer)

    def nested_outer_variance(self, n_outer=80):
        """Var_{z1}( log I(z1) ): the variance of the NMC estimator per outer sample as M -> oo."""
        mean = self.nested_target(n_outer)
        sq = self._outer_expectation(lambda z1: self.cond_log_marginal_lik(z1) ** 2, n_outer)
        return sq - mean**2

    def log_evidence(self, n_outer=80):
        """log p(x) = log E_{z1}[ I(z1) ]  (both latents integrated out)."""
        return float(np.log(self._outer_expectation(self.cond_marginal_lik, n_outer)))

    # ------------------------------------------------------------------
    def _outer_expectation(self, f, n_nodes):
        """E_{z1 ~ N(mu1, s1^2)}[ f(z1) ] by Gauss-Hermite quadrature."""
        t, w = hermegauss(n_nodes)
        w = w / np.sqrt(2.0 * np.pi)
        return float(np.sum(w * f(self.mu1 + self.s1 * t)))
