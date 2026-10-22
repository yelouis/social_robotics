from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict P(y=1 | x) = sigmoid(w · x + b)."""
    X_arr = np.asarray(X, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)
    z = X_arr @ w_arr + float(b)
    return expit(z)


def prior_regularized_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    w0: Optional[np.ndarray] = None,
    C: float = 1.0,
    is_soft: bool = False,
    maxiter: int = 2000,
    gtol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Prior-regularized logistic regression per Wave D contract 04_wave_d_robot_check.md §4.

    Minimizes:
        1/2 ||w - w0||^2 + C * sum_i s_i * logloss(y_i, w · x_i + b)
    with b unpenalized and solver initialized at w = w0, b = 0.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    n, d = X_arr.shape

    if w0 is not None:
        w0_arr = np.asarray(w0, dtype=np.float64)
        if w0_arr.shape != (d,):
            raise ValueError(f"w0 shape {w0_arr.shape} does not match feature dimension {d}")
    else:
        w0_arr = np.zeros(d, dtype=np.float64)

    if is_soft:
        # Soft labels q in (0, 1): each item contributes (x, 1) with s1*q and (x, 0) with s0*(1-q)
        q = np.asarray(y, dtype=np.float64)
        sum_q = float(np.sum(q))
        sum_1_minus_q = float(np.sum(1.0 - q))

        if sum_q <= 0 or sum_1_minus_q <= 0:
            raise ValueError("Soft labels must have non-zero sums for both classes.")

        s1 = n / (2.0 * sum_q)
        s0 = n / (2.0 * sum_1_minus_q)

        # Expand into 2n items
        X_expanded = np.vstack([X_arr, X_arr])
        y_expanded = np.concatenate([np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.float64)])
        sw_expanded = np.concatenate([s1 * q, s0 * (1.0 - q)])

        X_eff = X_expanded
        y_eff = y_expanded
        sw_eff = sw_expanded
    else:
        y_arr = np.asarray(y, dtype=np.float64)
        X_eff = X_arr
        y_eff = y_arr

        if sample_weight is None:
            # Balanced class weights: s_i = n / (2 * n_{y_i})
            n1 = float(np.sum(y_arr == 1))
            n0 = float(np.sum(y_arr == 0))
            if n1 == 0 or n0 == 0:
                raise ValueError("Fewer than 2 classes present in hard labels.")
            s1 = n / (2.0 * n1)
            s0 = n / (2.0 * n0)
            sw_eff = np.where(y_arr == 1, s1, s0)
        else:
            sw_eff = np.asarray(sample_weight, dtype=np.float64)

    def objective_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        w = theta[:d]
        b = theta[d]

        z = X_eff @ w + b
        p = expit(z)

        # Robust log-loss:
        # For y=1: log(1 + exp(-z))
        # For y=0: log(1 + exp(z))
        # Equivalently: logaddexp(0, z) - y * z
        loss_items = np.logaddexp(0.0, z) - y_eff * z
        data_loss = float(C * np.sum(sw_eff * loss_items))

        # Prior regularization: 1/2 ||w - w0||^2
        w_diff = w - w0_arr
        reg_loss = float(0.5 * np.sum(w_diff * w_diff))
        total_loss = reg_loss + data_loss

        # Gradients
        grad_w = w_diff + C * (X_eff.T @ (sw_eff * (p - y_eff)))
        grad_b = C * np.sum(sw_eff * (p - y_eff))

        grad = np.empty(d + 1, dtype=np.float64)
        grad[:d] = grad_w
        grad[d] = grad_b

        return total_loss, grad

    # Initialize at w = w0, b = 0
    theta0 = np.zeros(d + 1, dtype=np.float64)
    theta0[:d] = w0_arr

    res = minimize(
        objective_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "gtol": gtol},
    )

    w_opt = res.x[:d]
    b_opt = float(res.x[d])

    return w_opt, b_opt
