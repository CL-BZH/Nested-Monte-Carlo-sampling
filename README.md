# Hierarchical vs. Nested Monte Carlo

Two notebooks about estimating expectations that involve **two layers of randomness**. The code for
the two is almost identical: an outer loop over $z_1$ and an inner loop over $z_2 \mid z_1$. The
statistics are completely different, and the difference comes down to where a nonlinearity sits.

| Notebook | Target | Key facts |
|---|---|---|
| [**Part 1 — Hierarchical (two-stage) sampling**](01_hierarchical_sampling.ipynb) | $\mathbb{E}_{z_1}\big[\mathbb{E}_{z_2\mid z_1}[h(z_1,z_2)]\big]$ | Tower property, so the estimator is unbiased for every $M$. $\operatorname{Var} = \sigma_B^2/N + \sigma_W^2/(NM)$. Optimal $M = \sqrt{(c_1/c_2)(\sigma_W^2/\sigma_B^2)}$. Rao–Blackwellisation as $M\to\infty$. |
| [**Part 2 — Nested Monte Carlo**](02_nested_monte_carlo.ipynb) | $\mathbb{E}_{z_1}\big[f\big(\mathbb{E}_{z_2\mid z_1}[h(z_1,z_2)]\big)\big]$, $f$ nonlinear | Bias $\approx -C/M$, so the estimator is inconsistent at fixed $M$. MSE $= b(M)^2 + v_M/N$. Optimal $M \propto T^{1/3}$, and the best MSE is $O(T^{-2/3})$ instead of $O(T^{-1})$. Richardson debiasing. Links to IWAE and experimental design. |

Start with Part 1. Part 2 reuses the same model and contrasts with it throughout.

## The running example

Both notebooks use one deliberately tiny model:

$$
(z_1, z_2) \sim \mathcal{N}\!\left(0,\ \begin{bmatrix}1 & \rho\\ \rho & 1\end{bmatrix}\right),\qquad
x \mid z_1, z_2 \sim \mathcal{N}\big(z_1^2 + z_2^2,\ 1\big),\qquad x = 1,\ \rho = 0.5 .
$$

| Quantity | Value | Studied in |
|---|---|---|
| $\mu = \mathbb{E}[\log p(x\mid z_1,z_2)]$ | $-\tfrac12\log 2\pi - 3 \approx -3.9189$ (closed form) | Part 1 |
| $J = \mathbb{E}_{z_1}[\log p(x\mid z_1)]$ | $\approx -2.2087$ (nested quadrature) | Part 2 |
| $\log p(x)$ | $\approx -1.4183$ (quadrature) | for context ($\mu \le J \le \log p(x)$ by Jensen) |

Because everything is two-dimensional, every quantity that the notebooks estimate by Monte Carlo is
also computed exactly or by Gauss–Hermite quadrature. Each claim about bias, variance or MSE is
therefore **checked against ground truth**, not just asserted.

## Repository layout

```
01_hierarchical_sampling.ipynb   Part 1 (executed, with outputs)
02_nested_monte_carlo.ipynb      Part 2 (executed, with outputs)
model.py                         the shared model: samplers, likelihood, exact and quadrature references
src/*.py                         plain-text (jupytext) sources of the notebooks, convenient for diffs
requirements.txt
```

## Running

```bash
pip install -r requirements.txt
jupyter lab
```

To rebuild the notebooks from their sources:

```bash
jupytext --to ipynb -o 01_hierarchical_sampling.ipynb src/01_hierarchical_sampling.py
jupyter nbconvert --to notebook --execute --inplace 01_hierarchical_sampling.ipynb
```

Part 1 runs in a few seconds, and Part 2 in about a minute on a laptop CPU.

## References

* T. Rainforth, R. Cornish, H. Yang, A. Warrington, F. Wood. *On Nesting Monte Carlo Estimators.* ICML 2018.
* Y. Burda, R. Grosse, R. Salakhutdinov. *Importance Weighted Autoencoders.* ICLR 2016.
* M. B. Giles. *Multilevel Monte Carlo methods.* Acta Numerica, 2015.
* C.-H. Rhee, P. W. Glynn. *Unbiased Estimation with Square Root Convergence for SDE Models.* Operations Research, 2015.
* C. P. Robert, G. Casella. *Monte Carlo Statistical Methods*, 2nd ed. Springer, 2004.
