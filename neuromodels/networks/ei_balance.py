"""Conductance-based E/I balanced network of LIF neurons (course Days 5–6).

The network is the COBA benchmark of Brette et al. (2007), a smaller version of
the network of Vogels & Abbott (2005): leaky integrate-and-fire neurons, 80 %
excitatory (E) and 20 % inhibitory (I), randomly connected with probability 2 %.
A presynaptic spike makes the postsynaptic conductance jump by ``w`` and decay
exponentially:

    tau_m dV/dt = (V_rest - V) + g_E (E_E - V) + g_I (E_I - V) + I_ext
    tau_E dg_E/dt = -g_E,        tau_I dg_I/dt = -g_I

Conductances are in units of the leak conductance g_L, so the weights 0.6 and
6.7 are the benchmark's 6 nS and 67 nS for g_L = 10 nS, and the drive ``I_ext``
is in mV (R·I; 20 mV is 200 pA). Brette et al. start the benchmark from random
conductances and let it run without input; started that way here (with the
initial conductances as recalled from the benchmark), activity died out within
~100 ms, so every neuron instead receives the same constant drive of 20 mV, as
in BrainPy's own version of the benchmark.

Alone, a neuron under this drive fires regularly at 53 Hz. In the network,
recurrent inhibition cancels most of the excitation: neurons fire at ~20 Hz,
irregularly (ISI CV > 1) and nearly independently. The irregularity is made by
the network itself, not by noisy input — the balanced state of van Vreeswijk &
Sompolinsky (1996) and Brunel (2000). Without inhibition the same network locks
into synchronous, regular firing at the refractory limit.

The projections use BrainPy's align-post API (``bp.dyn.FullProjAlignPostMg``):
event-driven sparse weights (``bp.dnn.EventCSRLinear`` on ``bp.conn.FixedProb``)
feed one exponential synapse per target population (``bp.dyn.Expon``), read out
as a conductance (``bp.dyn.COBA``). ``FixedProb`` draws exactly ``round(prob·N_post)``
targets per presynaptic neuron, so out-degrees are fixed and in-degrees vary.

References
----------
Vogels, T. P. & Abbott, L. F. (2005). Signal propagation and logic gating in
    networks of integrate-and-fire neurons. J. Neurosci. 25, 10786–10795.
Brette, R. et al. (2007). Simulation of networks of spiking neurons: a review of
    tools and strategies. J. Comput. Neurosci. 23, 349–398.
van Vreeswijk, C. & Sompolinsky, H. (1996). Chaos in neuronal networks with
    balanced excitatory and inhibitory activity. Science 274, 1724–1726.
Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and
    inhibitory spiking neurons. J. Comput. Neurosci. 8, 183–208.
"""

from typing import Dict, Tuple

import brainpy as bp
import brainpy.math as bm
import numpy as np

from neuromodels.utils import cv_isi, n_steps, run, spike_times

# LIF parameters of the benchmark (mV, ms).
V_REST, V_TH, V_RESET, TAU_M, TAU_REF = -60.0, -50.0, -60.0, 20.0, 5.0
E_EXC, E_INH = 0.0, -80.0  # synaptic reversal potentials (mV)
N_NEURONS, PROB = 4000, 0.02


def use_portable_event_kernels() -> None:
    """Run BrainPy's event-driven sparse kernels in pure JAX when Numba is absent.

    ``bp.dnn.EventCSRLinear`` calls ``brainevent``, whose CPU kernels default to
    Numba. Without Numba installed every event-driven projection fails to
    compile, so this switches brainevent's CPU default to its ``jax_raw``
    kernels (same results, a scatter-add over all synapses). It leaves an
    explicit user choice alone.
    """
    try:
        import numba  # noqa: F401
        return
    except ImportError:
        import brainevent
        if brainevent.config.get_backend("cpu") is None:
            brainevent.config.set_backend("cpu", "jax_raw")


class EINet(bp.DynSysGroup):
    """COBA network of ``n_exc`` excitatory and ``n_inh`` inhibitory LIF neurons.

    ``w_exc`` and ``w_inh`` are conductance jumps in units of g_L; setting
    ``w_inh=0`` removes inhibition. Every random choice (connectivity and initial
    voltages, V ~ N(-55, 2) mV) derives from ``seed``, and ``bp.reset_state``
    restores the same initial voltages, so repeated runs are identical.
    ``update(i_ext)`` takes the external drive in mV.
    """

    def __init__(self, n_exc: int = 3200, n_inh: int = 800, prob: float = PROB,
                 w_exc: float = 0.6, w_inh: float = 6.7, tau_exc: float = 5.0, tau_inh: float = 10.0,
                 seed: int = 0, method: str = "exp_auto"):
        super().__init__()
        use_portable_event_kernels()
        # LifRef holds the synaptic current at its start-of-step value; exp_auto then
        # integrates the leak exactly over the step (dt = 0.1 ms vs tau_eff ~ 0.7 ms).
        rng = np.random.default_rng(seed)
        lif = dict(V_rest=V_REST, V_th=V_TH, V_reset=V_RESET, tau=TAU_M, tau_ref=TAU_REF, method=method)
        self.E = bp.dyn.LifRef(n_exc, V_initializer=bm.asarray(rng.normal(-55.0, 2.0, n_exc)), **lif)
        self.I = bp.dyn.LifRef(n_inh, V_initializer=bm.asarray(rng.normal(-55.0, 2.0, n_inh)), **lif)
        seeds = rng.integers(0, 2**31 - 1, size=4)
        self.E2E = _projection(self.E, self.E, prob, w_exc, tau_exc, E_EXC, int(seeds[0]))
        self.E2I = _projection(self.E, self.I, prob, w_exc, tau_exc, E_EXC, int(seeds[1]))
        self.I2E = _projection(self.I, self.E, prob, w_inh, tau_inh, E_INH, int(seeds[2]))
        self.I2I = _projection(self.I, self.I, prob, w_inh, tau_inh, E_INH, int(seeds[3]))

    def update(self, i_ext=0.0):
        # Projections read the spikes of the previous step, so transmission takes one step.
        for proj in (self.E2E, self.E2I, self.I2E, self.I2I):
            proj()
        self.E(i_ext)
        self.I(i_ext)


def _projection(pre, post, prob, weight, tau, reversal, seed):
    """Sparse, event-driven projection ending in an exponential conductance synapse."""
    conn = bp.conn.FixedProb(prob, pre=pre.num, post=post.num, include_self=pre is not post, seed=seed)
    return bp.dyn.FullProjAlignPostMg(
        pre=pre, delay=None, comm=bp.dnn.EventCSRLinear(conn, weight),
        syn=bp.dyn.Expon.desc(post.num, tau=tau), out=bp.dyn.COBA.desc(E=reversal), post=post)


def scaled_network(n_neurons: int, **kwargs) -> EINet:
    """A smaller (or larger) network with the benchmark's in-degree (64 E and 16 I inputs).

    Each neuron's input mean and variance depend on how many inputs it receives
    and their weights, not on the network size, so keeping ``prob * N`` fixed
    preserves single-neuron input statistics. What changes is the fraction of
    inputs two neurons share (``prob``), which raises pairwise correlations a little.
    """
    n_exc = int(round(0.8 * n_neurons))
    return EINet(n_exc=n_exc, n_inh=n_neurons - n_exc, prob=PROB * N_NEURONS / n_neurons, **kwargs)


def simulate(net: EINet, duration: float, dt: float = 0.1, i_ext: float = 20.0,
             n_record: int = 10) -> Dict[str, np.ndarray]:
    """Run ``net`` from its initial state under constant drive ``i_ext`` (mV).

    Returns spikes of both populations, for the first ``n_record`` E neurons V
    and both conductances, and the drive. ``"ts"`` holds end-of-step times, as
    :func:`neuromodels.utils.run` returns them.
    """
    monitors = {
        "spike_E": net.E.spike,
        "spike_I": net.I.spike,
        "V": lambda: net.E.V[:n_record],
        "g_E": lambda: net.E2E.syn.g[:n_record],
        "g_I": lambda: net.I2E.syn.g[:n_record],
    }
    out = run(net, np.full(n_steps(duration, dt), i_ext), dt, monitors=monitors)
    out["i_ext"] = np.asarray(i_ext)
    return out


def synaptic_currents(V, g_E, g_I) -> Tuple[np.ndarray, np.ndarray]:
    """Excitatory and inhibitory synaptic currents g (E_rev - V), in mV (R·I)."""
    return g_E * (E_EXC - V), g_I * (E_INH - V)


def isi_cvs(spikes: np.ndarray, ts: np.ndarray, min_spikes: int = 5) -> np.ndarray:
    """ISI coefficient of variation per neuron; NaN for neurons with fewer than ``min_spikes`` spikes."""
    return np.array([cv_isi(t) if len(t) >= min_spikes else np.nan for t in spike_times(spikes, ts)])


def pairwise_correlation(spikes: np.ndarray, dt: float, bin_size: float = 10.0, n_sample: int = 200,
                         seed: int = 0) -> float:
    """Mean Pearson correlation of spike counts in ``bin_size`` ms bins over random neuron pairs.

    Neurons that never fire (zero count variance) are left out; at most
    ``n_sample`` of the rest are drawn with a seeded generator.
    """
    width = n_steps(bin_size, dt)
    n_bins = spikes.shape[0] // width
    counts = spikes[:n_bins * width].reshape(n_bins, width, -1).sum(axis=1).astype(float)
    active = np.flatnonzero(counts.std(axis=0) > 0)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(active, size=min(n_sample, active.size), replace=False)
    corr = np.corrcoef(counts[:, chosen].T)
    return float(corr[np.triu_indices_from(corr, k=1)].mean())


def population_rate(spikes: np.ndarray, dt: float, width: float = 2.0) -> np.ndarray:
    """Population firing rate in Hz, averaged over a centred ``width`` ms window (shorter at the edges)."""
    box = np.ones(max(1, n_steps(width, dt)))
    counted = np.convolve(np.ones(spikes.shape[0]), box, mode="same")
    return np.convolve(spikes.mean(axis=1) / dt * 1e3, box, mode="same") / counted


def network_statistics(out: Dict[str, np.ndarray], dt: float, transient: float = 200.0) -> Dict[str, float]:
    """Summary numbers of a :func:`simulate` result, skipping the first ``transient`` ms.

    Currents are time averages over the recorded E neurons: ``I_exc`` includes
    the external drive and ``I_net`` sums all inputs. ``net_to_exc`` is
    |net| / excitation per neuron, averaged: 1 without inhibition, near 0 when
    inhibition cancels excitation.
    """
    start = n_steps(transient, dt)
    ts = out["ts"][start:]
    spikes = np.concatenate([out["spike_E"], out["spike_I"]], axis=1)[start:]
    rates = spikes.sum(axis=0) / (spikes.shape[0] * dt) * 1e3
    n_exc = out["spike_E"].shape[1]
    cvs = isi_cvs(spikes, ts)
    I_E, I_I = synaptic_currents(out["V"][start:], out["g_E"][start:], out["g_I"][start:])
    exc = (I_E + out["i_ext"]).mean(axis=0)
    inh = I_I.mean(axis=0)
    return {
        "rate_E": float(rates[:n_exc].mean()),
        "rate_I": float(rates[n_exc:].mean()),
        "cv_mean": float(np.nanmean(cvs)),
        "cv_counted": int(np.isfinite(cvs).sum()),
        "correlation": pairwise_correlation(spikes, dt),
        "I_exc": float(exc.mean()),
        "I_inh": float(inh.mean()),
        "I_net": float((exc + inh).mean()),
        "net_to_exc": float(np.mean(np.abs(exc + inh) / exc)),
    }
