
"""
Optimization Assignment – Sum-of-Squares Function f(x)
=====================================================

This script performs *all* tasks requested in the prompt:

1. **Analytic stationary-point analysis**
   • Finds the three stationary points of  
     f(x) = f₁(x)² + f₂(x)² and classifies each as  
     global minimum, strict local minimum, or saddle point.

2. **Numerical minimization**
   • Implements three algorithms with Armijo back-tracking  
       – Gradient (steepest-descent) method  
       – Hybrid gradient–Newton method  
       – Damped Gauss–Newton method  
   • Runs each algorithm from the three prescribed starting points.  
   • Prints a concise convergence report.

The implementation follows the notation of *Non-linear Optimization*
by Amir Beck (parameters s = 1, α = 0.5, β = 0.5; stopping criterion
‖∇f(x)‖ ≤ 1 × 10⁻⁵).

Author : Maximo Gonzalez (et al.)  
Date   : 2025-05-08
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Any, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Optional symbolic support (only for analytic part)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import sympy as sp

    _HAVE_SYMPY = True
except ImportError:  # pragma: no cover
    _HAVE_SYMPY = False


# ──────────────────────────────────────────────────────────────────────────────
# Residual vector  r(x)  and cost  f(x) = ‖r(x)‖²
# ──────────────────────────────────────────────────────────────────────────────
def residual(x: np.ndarray) -> np.ndarray:
    """Residual vector r(x) = [f₁(x), f₂(x)]ᵀ."""
    x1, x2 = x
    f1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    f2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    return np.array([f1, f2], dtype=float)


def cost(x: np.ndarray) -> float:
    """Objective function f(x) = ‖r(x)‖₂²."""
    r = residual(x)
    return float(np.dot(r, r))


# ──────────────────────────────────────────────────────────────────────────────
# Analytic gradient  ∇f  and Hessian  H
# ──────────────────────────────────────────────────────────────────────────────
def _g1(x2: float) -> float:  # ∂f₁/∂x₂
    return -3 * x2**2 + 10 * x2 - 2


def _g2(x2: float) -> float:  # ∂f₂/∂x₂
    return 3 * x2**2 + 2 * x2 - 14


def _g1p(x2: float) -> float:  # ∂²f₁/∂x₂²
    return -6 * x2 + 10


def _g2p(x2: float) -> float:  # ∂²f₂/∂x₂²
    return 6 * x2 + 2


def grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient ∇f(x)."""
    f1, f2 = residual(x)
    g1, g2 = _g1(x[1]), _g2(x[1])
    return np.array(
        [
            2.0 * (f1 + f2),                 # ∂f/∂x₁
            2.0 * (f1 * g1 + f2 * g2),       # ∂f/∂x₂
        ],
        dtype=float,
    )


def hessian(x: np.ndarray) -> np.ndarray:
    """Analytic Hessian matrix H(x)."""
    f1, f2 = residual(x)
    x2 = x[1]
    g1, g2 = _g1(x2), _g2(x2)
    g1p, g2p = _g1p(x2), _g2p(x2)

    h11 = 0.0
    h12 = 2.0 * (g1 + g2)
    h22 = 2.0 * (g1**2 + f1 * g1p + g2**2 + f2 * g2p)
    return np.array([[h11, h12], [h12, h22]], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Back-tracking line-search (Armijo condition)
# ──────────────────────────────────────────────────────────────────────────────
def backtracking(
    x: np.ndarray,
    p: np.ndarray,
    g: np.ndarray,
    s: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    max_ls: int = 50,
) -> float:
    """Armijo back-tracking line search."""
    t, fx = s, cost(x)
    for _ in range(max_ls):
        if cost(x + t * p) <= fx + alpha * t * np.dot(g, p):
            return t
        t *= beta
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Generic descent-method driver
# ──────────────────────────────────────────────────────────────────────────────
class OptResult(Dict[str, Any]):
    """Dictionary-like container holding the result of an optimization run."""


def _iterate(
    x0: np.ndarray,
    direction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    s: float,
    alpha: float,
    beta: float,
    tol: float,
    max_iter: int = 10_000,
) -> OptResult:
    """Core iterate-and-search loop shared by all methods."""
    x = x0.copy()
    for k in range(max_iter):
        g = grad(x)
        if (gn := np.linalg.norm(g)) <= tol:
            break
        p = direction(x, g)
        if np.dot(g, p) >= 0:          # Not a descent direction ⇒ fall back
            p = -g
        t = backtracking(x, p, g, s, alpha, beta)
        x = x + t * p
    return OptResult(
        x=x,
        cost=cost(x),
        grad_norm=np.linalg.norm(grad(x)),
        iterations=k,
        converged=gn <= tol,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Steepest-descent (gradient) method
# ──────────────────────────────────────────────────────────────────────────────
def gradient_method(
    x0: np.ndarray,
    *,
    s: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    tol: float = 1e-5,
) -> OptResult:
    """Classic steepest-descent with Armijo back-tracking."""
    return _iterate(
        x0,
        lambda _x, g: -g,
        s=s,
        alpha=alpha,
        beta=beta,
        tol=tol,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Hybrid gradient–Newton method
# ──────────────────────────────────────────────────────────────────────────────
def hybrid_method(
    x0: np.ndarray,
    *,
    s: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    tol: float = 1e-5,
) -> OptResult:
    """
    Hybrid method that attempts a full Newton step when the Hessian is
    symmetric-positive-definite; otherwise reverts to steepest descent.
    """

    def direction(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        H = hessian(x)
        try:
            np.linalg.cholesky(H)          # raises if H not SPD
            return -np.linalg.solve(H, g)  # Newton direction
        except np.linalg.LinAlgError:
            return -g                      # Steepest-descent fallback

    return _iterate(x0, direction, s=s, alpha=alpha, beta=beta, tol=tol)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Damped Gauss–Newton method (for least-squares structure)
# ──────────────────────────────────────────────────────────────────────────────
def jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian J(x) of the residual vector."""
    x2 = x[1]
    return np.array(
        [[1.0, _g1(x2)],
         [1.0, _g2(x2)]],
        dtype=float,
    )


def gauss_newton(
    x0: np.ndarray,
    *,
    s: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    tol: float = 1e-5,
) -> OptResult:
    """Damped Gauss–Newton with Armijo back-tracking."""
    def direction(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        J = jacobian(x)
        JTJ, JTr = J.T @ J, J.T @ residual(x)
        try:
            return -np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:       # JTJ singular ⇒ fallback
            return -g

    return _iterate(x0, direction, s=s, alpha=alpha, beta=beta, tol=tol)


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic stationary-point analysis (requires SymPy)
# ──────────────────────────────────────────────────────────────────────────────
def stationary_points() -> Optional[List[Dict[str, Any]]]:
    """Return a list of stationary points and their classification."""
    if not _HAVE_SYMPY:
        return None

    x1, x2 = sp.symbols("x1 x2", real=True)
    f1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    f2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    f = f1**2 + f2**2
    g1, g2 = sp.diff(f, x1), sp.diff(f, x2)
    critical = sp.solve((g1, g2), (x1, x2), dict=True)

    pts: List[Dict[str, Any]] = []
    for sol in critical:
        if sol[x1].is_real and sol[x2].is_real:
            pt = np.array([float(sol[x1]), float(sol[x2])])
            H = hessian(pt)
            eig = np.linalg.eigvals(H)
            if np.all(eig > 0):
                kind = "strict local minimum"
            elif np.all(eig < 0):
                kind = "strict local maximum"
            else:
                kind = "saddle point"
            pts.append(dict(point=pt, cost=cost(pt), kind=kind))

    # Mark the global minimizer (lowest cost among minima)
    if pts:
        pts[min(range(len(pts)), key=lambda i: pts[i]["cost"])]["kind"] = "global minimum"
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# Convenience routine to run *all* methods from the specified start points
# ──────────────────────────────────────────────────────────────────────────────
def run_all() -> Dict[str, List[OptResult]]:
    """Execute each method from each initial point and return a report."""
    starts = [
        np.array([-50.0, 7.0]),
        np.array([20.0, 7.0]),
        np.array([20.0, -18.0]),
    ]
    methods = {
        "Gradient":      gradient_method,
        "Hybrid GN":     hybrid_method,
        "Gauss-Newton":  gauss_newton,
    }
    report: Dict[str, List[OptResult]] = {m: [] for m in methods}
    for name, meth in methods.items():
        for x0 in starts:
            report[name].append(meth(x0))
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point – run analysis & experiments
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    # 1. Stationary-point summary
    if (pts := stationary_points()) is not None:
        print("Stationary points:")
        for p in pts:
            print(f"  x = {p['point']},  f(x) = {p['cost']:.4f},  type: {p['kind']}")
    else:
        print("SymPy unavailable – skipping analytic stationary-point analysis.")

    # 2. Numerical experiments
    results = run_all()
    print("\nOptimization results (‖∇f‖ ≤ 1e-5):")
    for mname, res_list in results.items():
        print(f"\n== {mname} ==")
        for i, res in enumerate(res_list, 1):
            print(
                f"  start #{i}: iter = {res['iterations']:>5}, "
                f"x* = {res['x']}, "
                f"‖∇f‖ = {res['grad_norm']:.1e}, "
                f"f(x*) = {res['cost']:.4f}, "
                f"converged = {res['converged']}"
            )

