r"""Continuous attractor neural network (CANN) on a ring (course Days 5–6).

``N`` rate neurons have preferred stimuli ``x`` evenly spaced on the periodic
feature space (−π, π], such as head direction or orientation. Excitation falls
off as a Gaussian of the distance between preferred stimuli, and global
divisive inhibition keeps activity bounded (Wu, Hamaguchi & Amari 2008;
Fung, Wong & Wu 2010):

.. math::
    \tau \frac{\partial u(x,t)}{\partial t} = -u(x,t) + \rho \int J(x,x') r(x',t)\,dx' + I_{ext}(x,t)

    r(x,t) = \frac{u(x,t)^2}{1 + k \rho \int u(x',t)^2\,dx'}, \qquad
    J(x,x') = \frac{J_0}{\sqrt{2\pi} a} \exp\left(-\frac{|x-x'|^2}{2a^2}\right)

``ρ = N / 2π`` is the neuron density and ``|x − x′|`` the distance on the ring.
``u`` stays non-negative for non-negative input, so ``u²`` acts as a rectified
square. Since ``J`` depends only on ``x − x′``, a bump of activity can sit
anywhere on the ring. The stationary states form a continuous family, and for
``a ≪ π`` they are Gaussian (Wu et al. 2008):

.. math::
    \tilde u(x|z) = u_0 e^{-(x-z)^2/4a^2}, \qquad \tilde r(x|z) = r_0 e^{-(x-z)^2/2a^2},

    u_0 = \frac{J_0 [1 + \sqrt{1 - k/k_c}]}{4\sqrt{\pi} a k}, \qquad
    r_0 = \frac{1 + \sqrt{1 - k/k_c}}{2\sqrt{2\pi} a k \rho}, \qquad
    k_c = \frac{\rho J_0^2}{8\sqrt{2\pi} a}.

Above the critical inhibition ``k_c`` the only stable state is silence. Stimuli
have the bump's own shape, ``I_ext = A exp(−|x − z₀|²/4a²)``, and positions are
read out with the population vector ``arg Σ_j r_j e^{i x_j}`` (Georgopoulos et
al. 1986).

References
----------
Amari, S. (1977). Dynamics of pattern formation in lateral-inhibition type
neural fields. Biological Cybernetics 27, 77–87.
Georgopoulos, A. P., Schwartz, A. B. & Kettner, R. E. (1986). Neuronal population
coding of movement direction. Science 233, 1416–1419.
Wu, S., Hamaguchi, K. & Amari, S. (2008). Dynamics and computation of continuous
attractors. Neural Computation 20, 994–1025.
Fung, C. C. A., Wong, K. Y. M. & Wu, S. (2010). A moving bump in a continuous
manifold: a comprehensive study of the tracking dynamics of continuous attractor
neural networks. Neural Computation 22, 752–792.
"""

from typing import Optional, Tuple

import brainpy as bp
import brainpy.math as bm
import numpy as np

from neuromodels.utils import n_steps


def ring_positions(num: int) -> np.ndarray:
    """``num`` preferred stimuli evenly spaced on (−π, π]."""
    return -np.pi + 2.0 * np.pi * np.arange(1, num + 1) / num


def ring_distance(x, y) -> np.ndarray:
    """Signed distance ``x − y`` on the ring, wrapped into [−π, π)."""
    return np.mod(np.asarray(x) - np.asarray(y) + np.pi, 2.0 * np.pi) - np.pi


def critical_inhibition(rho: float, J0: float = 1.0, a: float = 0.5) -> float:
    """Largest global inhibition ``k_c = ρ J0² / (8 √(2π) a)`` that still supports a bump."""
    return rho * J0 ** 2 / (8.0 * np.sqrt(2.0 * np.pi) * a)


def stationary_bump(k: float, rho: float, J0: float = 1.0, a: float = 0.5, A: float = 0.0) -> Tuple[float, float]:
    """Heights ``(u0, r0)`` of the stable bump, optionally held by a centred stimulus.

    A Gaussian ``u = U exp(−(x − z)²/4a²)`` keeps its shape under the dynamics
    (a Gaussian convolved with a Gaussian is Gaussian), and so does a stimulus of
    the same shape centred on it. The height then obeys an exact scalar ODE:

        τ dU/dt = −U + c U² / (1 + b U²) + A,   c = ρ J0 / √2,   b = k ρ √(2π) a.

    With ``A = 0`` its stable root is the closed form of Wu et al. (2008); with
    ``A > 0`` it is the largest root of the cubic ``−bU³ + (c + Ab)U² − U + A``.
    Returns NaN heights when no bump exists (``k > k_c`` without input).
    """
    b = k * rho * np.sqrt(2.0 * np.pi) * a
    if A == 0.0:
        ratio = k / critical_inhibition(rho, J0, a)
        if ratio > 1.0:
            return float("nan"), float("nan")
        u0 = J0 * (1.0 + np.sqrt(1.0 - ratio)) / (4.0 * np.sqrt(np.pi) * a * k)
    else:
        c = rho * J0 / np.sqrt(2.0)
        roots = np.roots([-b, c + A * b, -1.0, A])
        u0 = max(root.real for root in roots if abs(root.imag) < 1e-9 * abs(root) and root.real > 0)
    return float(u0), float(u0 ** 2 / (1.0 + b * u0 ** 2))


def gaussian_input(x: np.ndarray, center, amplitude, a: float = 0.5) -> np.ndarray:
    """Stimulus ``A exp(−|x − z|²/4a²)`` on the ring.

    ``center`` and ``amplitude`` are scalars or 1-D arrays over time, giving an
    array of shape ``(N,)`` or ``(T, N)`` ready to pass to :func:`neuromodels.utils.run`.
    """
    center = np.asarray(center, dtype=float)[..., None]
    amplitude = np.asarray(amplitude, dtype=float)[..., None]
    return amplitude * np.exp(-ring_distance(x, center) ** 2 / (4.0 * a ** 2))


def step_midpoints(duration: float, dt: float) -> np.ndarray:
    """Midpoints ``t_i + dt/2`` of the integration steps in ``duration``.

    The input is held constant over each step, so a moving stimulus evaluated at
    the step midpoint avoids the extra ``dt/2`` delay of sampling it at ``t_i``.
    """
    return (np.arange(n_steps(duration, dt)) + 0.5) * dt


def population_vector(activity: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Decoded position ``arg Σ_j r_j e^{i x_j}`` along the last axis (one per row)."""
    return np.angle(np.asarray(activity) @ np.exp(1j * np.asarray(x)))


def max_tracking_speed(tau: float, A: float, U: float, a: float = 0.5) -> float:
    """Fastest stimulus a bump of height ``U`` can follow, to first order in ``A``: √3 a e^{−1/2} A / (U τ)."""
    return np.sqrt(3.0) * a * np.exp(-0.5) * A / (U * tau)


def tracking_lag(velocity: float, tau: float, A: float, U: float, a: float = 0.5) -> float:
    """Steady lag ``s = z_stimulus − z_bump`` behind a stimulus moving at ``velocity``.

    Projecting the dynamics onto the adjoint translation mode ψ ∝ u ∂u/∂z (the
    left null vector of the linearised dynamics around the bump) gives, to first
    order in the stimulus strength ``A``,

        v τ = (A / U) s exp(−s² / 6a²),

    where ``U`` is the height of ``u`` while tracking. The lag grows linearly with
    speed at first, ``s ≈ vτU/A``, and the right-hand side peaks at ``s = √3 a``,
    which sets :func:`max_tracking_speed`. Returns the smaller root with the sign
    of ``velocity``, or NaN above the maximal speed.
    """
    target = abs(velocity) * tau * U / A
    s_max = np.sqrt(3.0) * a
    if target > s_max * np.exp(-0.5):
        return float("nan")
    lo, hi = 0.0, s_max
    for _ in range(80):  # bisection: s exp(−s²/6a²) increases monotonically on [0, √3 a]
        mid = 0.5 * (lo + hi)
        if mid * np.exp(-mid ** 2 / (6.0 * a ** 2)) < target:
            lo = mid
        else:
            hi = mid
    return float(np.copysign(0.5 * (lo + hi), velocity))


class CANN1D(bp.DynamicalSystem):
    r"""One-dimensional CANN with Gaussian recurrent excitation and divisive inhibition.

    The integrals over the ring are Riemann sums over the ``N`` neurons, which
    are spectrally accurate for smooth periodic integrands. With ``a/Δx ≳ 10``
    the discrete network reproduces the continuum bump and ``k_c`` to within the
    periodic wrap-around term ``e^{−π²/4a²}`` (5e-5 for a = 0.5). ``J`` is circulant,
    so rotating the state by whole grid steps commutes with the dynamics.

    Integration defaults to classical RK4: the right-hand side is smooth and not
    stiff (one time constant, bounded recurrent gain), so a fourth-order step at
    ``dt = τ/20`` is accurate to well below the effects studied here. BrainPy's
    default ``exp_auto`` takes the "linear part" from ``bm.vector_grad``, which
    for this all-to-all coupled system mixes recurrent gain into each neuron's
    leak instead of isolating ``−u/τ``.

    Parameters: ``num`` neurons, time constant ``tau`` (the unit of time in this
    chapter), global inhibition ``k`` (defaults to ``k_c / 2``), interaction range
    ``a`` (rad), and excitation strength ``J0``. ``update(I_ext)`` takes the
    external input for one step, shape ``(num,)``, and returns ``r``.
    """

    def __init__(self, num: int = 256, tau: float = 1.0, k: Optional[float] = None, a: float = 0.5,
                 J0: float = 1.0, method: str = "rk4", name: Optional[str] = None):
        super().__init__(name=name)
        self.num, self.tau, self.a, self.J0 = num, tau, a, J0
        self.x = ring_positions(num)
        self.dx = 2.0 * np.pi / num
        self.rho = num / (2.0 * np.pi)
        self.k_c = critical_inhibition(self.rho, J0, a)
        self.k = 0.5 * self.k_c if k is None else k
        distance = ring_distance(self.x[:, None], self.x[None, :])
        self.conn = bm.asarray(J0 / (np.sqrt(2.0 * np.pi) * a) * np.exp(-distance ** 2 / (2.0 * a ** 2)))
        self.u = bm.Variable(bm.zeros(num))
        self.r = bm.Variable(bm.zeros(num))
        self.integral = bp.odeint(self.du, method=method)

    def rate(self, u):
        """Firing rate ``u² / (1 + k ρ ∫ u² dx)``."""
        u2 = bm.square(u)
        return u2 / (1.0 + self.k * self.rho * bm.sum(u2) * self.dx)

    def du(self, u, t, I_ext):
        # ρ ∫ J r dx′ as a Riemann sum; ρ dx = 1, kept explicit to mirror the equation.
        recurrent = self.rho * self.dx * (self.conn @ self.rate(u))
        return (-u + recurrent + I_ext) / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.u.value = bm.zeros(self.num)
        self.r.value = bm.zeros(self.num)

    def update(self, I_ext=0.0):
        t = bp.share.load("t")
        dt = bp.share.load("dt")
        self.u.value = self.integral(self.u.value, t, I_ext, dt=dt)
        self.r.value = self.rate(self.u.value)
        return self.r.value
