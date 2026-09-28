import brainpy as bp
import numpy as np
import pytest

from neuromodels.synapses import (
    AMPA, GABAA, NMDA, Alpha, DualExponential, Exponential, SynapticLIF, alpha_kernel, coba_current,
    cuba_current, dual_exp_peak_time, dual_exponential_kernel, exponential_kernel, markov_pulse_response,
    mg_block, nmda_current, psp_amplitude, regular_times, spike_input,
)
from neuromodels.utils import run

DT = 0.1


def response(model, times, duration, dt=DT):
    out = run(model, spike_input(times, duration, dt), dt, monitors=["g"])
    return out["ts"], out["g"][:, 0]


def peak(ts, g):
    """Time and height of the vertex of the parabola through the largest sample and its neighbours."""
    k = np.argmax(g)
    a, b, c = np.polyfit(ts[k - 1:k + 2] - ts[k], g[k - 1:k + 2], 2)
    return ts[k] - b / (2 * a), c - b * b / (4 * a)


def shifted_sum(single, times, dt=DT):
    """Superpose copies of a single-spike response (spike at ``times[0]``) at every time in ``times``."""
    total = np.zeros_like(single)
    for shift in np.rint((np.asarray(times) - times[0]) / dt).astype(int):
        total[shift:] += single[:len(single) - shift]
    return total


def test_spike_input_puts_each_spike_in_the_step_that_ends_at_it():
    np.testing.assert_array_equal(np.nonzero(spike_input([0.3, 1.0], 1.0, 0.1))[0], [2, 9])
    assert spike_input([[0.5], [0.2, 0.4]], 1.0, 0.1).shape == (10, 2)
    with pytest.raises(ValueError):
        spike_input([0.0], 1.0, 0.1)


def test_exponential_synapse_decays_with_its_time_constant():
    # Exponential Euler solves dg/dt = -g/tau exactly, so the trace is the kernel itself.
    ts, g = response(Exponential(1, tau=5.0), [10.0], 60.0)
    np.testing.assert_allclose(g, exponential_kernel(ts - 10.0, 5.0), atol=1e-12)
    decaying = (ts >= 10.0) & (ts < 50.0)
    slope = np.polyfit(ts[decaying], np.log(g[decaying]), 1)[0]
    assert -1 / slope == pytest.approx(5.0, rel=1e-9)


@pytest.mark.parametrize("tau_rise, tau_decay", [(0.5, 5.0), (1.0, 5.0), (2.0, 20.0)])
def test_dual_exponential_peaks_at_the_closed_form_time_with_unit_height(tau_rise, tau_decay):
    dt = 0.01  # fine grid so the parabolic peak estimate is accurate to ~1e-5 ms
    ts, g = response(DualExponential(1, tau_rise, tau_decay), [5.0], 5.0 + 6 * tau_decay, dt)
    t_peak, height = peak(ts, g)
    assert t_peak - 5.0 == pytest.approx(dual_exp_peak_time(tau_rise, tau_decay), abs=1e-3)
    assert height == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(g, dual_exponential_kernel(ts - 5.0, tau_rise, tau_decay) * (ts >= 5.0), atol=1e-7)


def test_alpha_synapse_peaks_at_tau_with_unit_height():
    ts, g = response(Alpha(1, tau=4.0), [5.0], 40.0, dt=0.01)
    t_peak, height = peak(ts, g)
    assert t_peak - 5.0 == pytest.approx(4.0, abs=1e-3)
    assert height == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(g, alpha_kernel(ts - 5.0, 4.0), atol=1e-7)


def test_linear_kernels_superpose_and_summate_under_a_100hz_train():
    times = regular_times(100.0, 8, 10.0)
    for model in (Exponential(1, 5.0), DualExponential(1, 1.0, 5.0), Alpha(1, 3.0)):
        _, train = response(model, times, 120.0)
        _, single = response(model, times[:1], 120.0)
        # Linear time-invariant kinetics: the train response is the sum of shifted single responses.
        np.testing.assert_allclose(train, shifted_sum(single, times), atol=1e-12)
        assert train.max() > 1.1 * single.max()
    # For the exponential synapse the value right after spike n is 1 + q + ... + q^(n-1), q = exp(-10/5).
    ts, train = response(Exponential(1, 5.0), times, 120.0)
    q = np.exp(-10.0 / 5.0)
    after_spike = train[np.rint(times / DT).astype(int) - 1]
    np.testing.assert_allclose(after_spike, (1 - q ** np.arange(1, 9)) / (1 - q), rtol=1e-12)


def test_nmda_gating_saturates_so_trains_summate_sublinearly():
    times = regular_times(100.0, 10, 10.0)
    _, train = response(NMDA(1), times, 300.0)
    _, single = response(NMDA(1), times[:1], 300.0)
    # The (1 - g) factor caps g below 1, whereas linear superposition would exceed 4.
    assert train.max() < 1.0
    assert train.max() < 0.5 * shifted_sum(single, times).max()


@pytest.mark.parametrize("receptor", [AMPA, GABAA])
def test_markov_receptor_matches_the_single_pulse_solution(receptor):
    model = receptor(1)
    alpha, beta = float(model.alpha), float(model.beta)
    ts, g = response(model, [10.0], 60.0)
    # [T] is constant within each step, so exponential Euler is exact for this linear ODE.
    exact = np.where(ts >= 10.0, markov_pulse_response(ts - 10.0, alpha, beta, model.T_max, model.T_dur), 0.0)
    np.testing.assert_allclose(g, exact, atol=1e-12)
    decaying = ts >= 10.0 + model.T_dur
    slope = np.polyfit(ts[decaying], np.log(g[decaying]), 1)[0]
    assert -slope == pytest.approx(beta, rel=1e-9)  # unbinding alone sets the decay after the pulse


def test_mg_block_follows_jahr_stevens_and_is_relieved_by_depolarization():
    V = np.linspace(-100.0, 40.0, 141)
    np.testing.assert_allclose(mg_block(V, 1.2), 1 / (1 + 1.2 / 3.57 * np.exp(-0.062 * V)), rtol=1e-12)
    assert np.all(np.diff(mg_block(V)) > 0)
    # Half the channels are free where [Mg]/3.57 exp(-0.062 V) = 1, i.e. V = ln([Mg]/3.57)/0.062 = -17.6 mV.
    assert mg_block(np.log(1.2 / 3.57) / 0.062) == pytest.approx(0.5, rel=1e-12)
    assert mg_block(-65.0) < 0.06 and mg_block(0.0) > 0.7
    np.testing.assert_allclose(mg_block(V, mg=0.0), 1.0)


def test_nmda_current_has_a_negative_slope_region():
    V = np.linspace(-90.0, -30.0, 61)
    # Below about -26 mV unblocking outpaces the loss of driving force, so the depolarizing
    # NMDA current grows with V; without Mg2+ it is ohmic and shrinks with V like AMPA's.
    assert np.all(np.diff(nmda_current(1.0, V)) > 0)
    assert np.all(np.diff(nmda_current(1.0, V, mg=0.0)) < 0)
    assert np.all(np.diff(coba_current(1.0, V, E=0.0)) < 0)


def test_coba_epsp_scales_with_driving_force_but_cuba_epsp_does_not():
    V0 = np.array([-80.0, -65.0, -50.0, -35.0, -20.0])
    inputs = spike_input([10.0], 80.0, DT)
    amplitude = {}
    for output, weight in [("coba", 0.05), ("cuba", 0.05 * 65.0)]:
        lif = SynapticLIF(Exponential(len(V0), 5.0), output, weight, E=0.0, V_th=100.0, I_ext=V0 + 65.0)
        amplitude[output] = psp_amplitude(run(lif, inputs, DT, monitors=["V"])["V"], V0)
    # The LIF is linear in V, so a current input gives the same EPSP at every baseline.
    np.testing.assert_allclose(amplitude["cuba"], amplitude["cuba"][0], rtol=1e-12)
    # With u = V - E the COBA membrane equation is linear and homogeneous in (u, V0 - E), so the
    # EPSP is exactly proportional to the driving force E - V0, shunting included.
    np.testing.assert_allclose(amplitude["coba"] / -V0, amplitude["coba"][0] / 80.0, rtol=1e-9)
    assert np.all(np.diff(amplitude["coba"]) < 0)


def test_gabaa_psp_reverses_at_its_reversal_potential():
    V0 = np.array([-90.0, -80.0, -70.0])
    lif = SynapticLIF(GABAA(3), "coba", 0.1, V_th=100.0, I_ext=V0 + 65.0)
    assert lif.E == -80.0
    amplitude = psp_amplitude(run(lif, spike_input([10.0], 60.0, DT), DT, monitors=["V"])["V"], V0)
    assert amplitude[0] > 0.1 and amplitude[2] < -0.1
    assert abs(amplitude[1]) < 1e-12


def test_nmda_epsp_grows_with_depolarization_while_ampa_epsp_shrinks():
    V0 = np.array([-80.0, -60.0, -40.0])
    inputs = spike_input([10.0], 150.0, DT)
    nmda = SynapticLIF(NMDA(3), "nmda", 0.05, V_th=100.0, I_ext=V0 + 65.0)
    ampa = SynapticLIF(AMPA(3), "coba", 0.05, V_th=100.0, I_ext=V0 + 65.0)
    out = run(nmda, inputs, DT, monitors=["V", "I_syn", "syn.g"])
    assert np.all(np.diff(psp_amplitude(out["V"], V0)) > 0)
    assert np.all(np.diff(psp_amplitude(run(ampa, inputs, DT, monitors=["V"])["V"], V0)) < 0)
    # The recorded current is the Mg2+-blocked conductance current of the recorded g and V.
    np.testing.assert_allclose(out["I_syn"], nmda_current(out["syn.g"], out["V"], 0.0, 0.05), atol=1e-12)


def test_cuba_epsp_converges_to_the_closed_form_at_first_order():
    J, tau_m, tau_s = 3.25, 20.0, 5.0
    t = np.linspace(0.0, 60.0, 60001)
    # tau_m dV/dt = -(V - V_rest) + J exp(-t/tau_s) gives a difference of exponentials.
    exact = (J * tau_s / (tau_m - tau_s) * (np.exp(-t / tau_m) - np.exp(-t / tau_s))).max()
    errors = []
    for dt in (0.1, 0.05):
        lif = SynapticLIF(Exponential(1, tau_s), "cuba", J, V_th=100.0, tau=tau_m)
        V = run(lif, spike_input([10.0], 80.0, dt), dt, monitors=["V"])["V"][:, 0]
        errors.append(abs(V.max() + 65.0 - exact) / exact)
    # The membrane sees the conductance from the start of each step (a left-endpoint rule).
    assert errors[0] < 0.015
    assert errors[1] == pytest.approx(errors[0] / 2, rel=0.05)


class BuiltinCUBANet(bp.DynSysGroup):
    """One spike through bp.dyn.Expon + bp.dyn.CUBA onto bp.dyn.LifRef."""

    def __init__(self, J, tau_s):
        super().__init__()
        self.pre = bp.dyn.SpikeTimeGroup(1, indices=[0], times=[10.0])
        self.post = bp.dyn.LifRef(1, V_rest=-65.0, V_reset=-65.0, V_th=100.0, tau=20.0,
                                  V_initializer=bp.init.Constant(-65.0))
        self.proj = bp.dyn.ProjAlignPostMg2(pre=self.pre, delay=None, comm=bp.dnn.OneToOne(1, J),
                                            syn=bp.dyn.Expon.desc(1, tau=tau_s), out=bp.dyn.CUBA.desc(),
                                            post=self.post)

    def update(self, x):
        self.pre()
        self.proj()
        self.post()


def test_cuba_epsp_matches_a_brainpy_projection_up_to_one_step_of_decay():
    J, tau_s = 3.25, 5.0
    builtin = run(BuiltinCUBANet(J, tau_s), np.zeros(800), DT, monitors=["post.V"])["post.V"][:, 0]
    lif = SynapticLIF(Exponential(1, tau_s), "cuba", J, V_th=100.0)
    mine = run(lif, spike_input([10.0], 80.0, DT), DT, monitors=["V"])["V"][:, 0]
    # The AlignPost projection lets the post neuron decay g once before using it, so with a
    # linear membrane BrainPy's EPSP is exactly exp(-dt/tau_s) times this one.
    np.testing.assert_allclose(builtin + 65.0, np.exp(-DT / tau_s) * (mine + 65.0), atol=1e-12)


@pytest.mark.parametrize("make_mine, make_builtin, scale", [
    (lambda: Exponential(1, 5.0), lambda: bp.dyn.Expon(1, tau=5.0), 1.0),
    (lambda: DualExponential(1, 1.0, 5.0), lambda: bp.dyn.DualExpon(1, tau_rise=1.0, tau_decay=5.0, method="rk4"), 1.0),
    (lambda: Alpha(1, 5.0), lambda: bp.dyn.Alpha(1, tau_decay=5.0, method="rk4"), np.e),  # bp.dyn.Alpha peaks at 1/e
    (lambda: NMDA(1), lambda: bp.dyn.NMDA(1, method="rk4"), 1.0),
], ids=["Expon", "DualExpon", "Alpha", "NMDA"])
def test_kinetics_match_brainpy_builtins(make_mine, make_builtin, scale):
    inputs = spike_input(regular_times(50.0, 4, 10.0), 150.0, DT)
    mine = run(make_mine(), inputs, DT, monitors=["g"])["g"]
    builtin = run(make_builtin(), inputs, DT, monitors=["g"])["g"]
    np.testing.assert_allclose(mine, scale * builtin, atol=1e-12)


def test_brainpy_dual_exponential_default_overshoots_the_peak():
    # bp.dyn.DualExpon defaults to exp_auto, which holds h fixed across a step (first order in dt).
    _, g = response(bp.dyn.DualExpon(1, tau_rise=1.0, tau_decay=5.0), [10.0], 40.0)
    assert g.max() > 1.03


def test_markov_receptors_match_brainpy_with_the_pulse_one_step_later():
    inputs = spike_input([10.0, 13.0, 30.0], 60.0, DT)
    pairs = [(AMPA(1), bp.dyn.AMPA(1, alpha=1.1, beta=0.19, T=1.0, T_dur=1.0)),
             (GABAA(1), bp.dyn.GABAa(1, alpha=5.0, beta=0.18, T=1.0, T_dur=1.0))]
    for mine, builtin in pairs:
        g_mine = run(mine, inputs, DT, monitors=["g"])["g"]
        # BrainPy starts the transmitter pulse in the step that receives the spike.
        g_builtin = run(builtin, np.roll(inputs, 1, axis=0), DT, monitors=["g"])["g"]
        np.testing.assert_allclose(g_mine, g_builtin, atol=1e-12)


def test_output_functions_match_brainpy_synaptic_outputs():
    g = np.array([0.0, 0.3, 1.2])
    V = np.array([-70.0, -40.0, 10.0])
    np.testing.assert_allclose(coba_current(g, V, E=0.0), np.asarray(bp.dyn.COBA(E=0.0).update(g, V)))
    np.testing.assert_allclose(cuba_current(g, 1.0), np.asarray(bp.dyn.CUBA().update(g)))
    np.testing.assert_allclose(nmda_current(g, V, E=0.0, mg=1.2),
                               np.asarray(bp.dyn.MgBlock(E=0.0, cc_Mg=1.2).update(g, V)), rtol=1e-12)
