# 07 · E/I balanced network (Days 5–6)

A randomly connected network of excitatory and inhibitory LIF neurons fires irregularly
and asynchronously even though every neuron receives the same constant, suprathreshold
drive. The irregularity comes from the network: large excitatory and inhibitory inputs
nearly cancel, and the neuron fires on the fluctuations that remain
(van Vreeswijk & Sompolinsky 1996; Brunel 2000).

## Model

Conductance-based LIF neurons (the COBA benchmark of Brette et al. 2007, after Vogels & Abbott 2005):

$$\tau_m \frac{dV}{dt} = (V_\text{rest} - V) + g_E\,(E_E - V) + g_I\,(E_I - V) + I_\text{ext}$$

$$\frac{dg_E}{dt} = -\frac{g_E}{\tau_E} + w_E \sum_{j \in \text{pre}_E(i)} \sum_k \delta(t - t_j^k), \qquad \text{same for } g_I$$

When $V \ge V_\text{th}$ the neuron spikes, $V \leftarrow V_\text{reset}$, and $V$ is clamped for $\tau_\text{ref}$.
Conductances are in units of the leak conductance $g_L$ and $I_\text{ext}$ is in mV ($R\,I$).

Two closed-form handles on the balanced state. Each spike adds a decaying exponential of area
$w\tau$, so the mean conductance is (Campbell's theorem)

$$\langle g_E \rangle = w_E\,\tau_E \sum_{j} \nu_j \approx K_E\,w_E\,\tau_E\,\nu_E = 64 \times 0.6 \times 5\,\text{ms} \times 21\,\text{Hz} \approx 4.0,
\qquad \langle g_I \rangle \approx 16 \times 6.7 \times 10\,\text{ms} \times 21\,\text{Hz} \approx 22.5 .$$

With $g_\text{tot} = 1 + g_E + g_I \approx 27.5$ the membrane relaxes with
$\tau_\text{eff} = \tau_m / g_\text{tot} \approx 0.7$ ms towards

$$V_\text{eff} = \frac{V_\text{rest} + I_\text{ext} + g_E E_E + g_I E_I}{1 + g_E + g_I} \approx -67\ \text{mV},$$

well below threshold: a high-conductance, fluctuation-driven state.

| Parameter | Value | Unit | Source |
|---|---|---|---|
| $N_E$, $N_I$ | 3200, 800 | — | Brette et al. 2007 (Vogels & Abbott used 10 000 neurons) |
| connection probability $p$ | 0.02 (in-degree 64 E + 16 I) | — | both papers |
| $\tau_m$, $\tau_\text{ref}$ | 20, 5 | ms | both papers |
| $V_\text{rest}$, $V_\text{reset}$, $V_\text{th}$ | −60, −60, −50 | mV | both papers |
| $E_E$, $E_I$ | 0, −80 | mV | both papers |
| $\tau_E$, $\tau_I$ | 5, 10 | ms | both papers |
| $w_E$, $w_I$ | 0.6, 6.7 | $g_L$ (6 nS, 67 nS for $g_L$ = 10 nS) | Brette et al. 2007 |
| $I_\text{ext}$ | 20 (= 200 pA) | mV | BrainPy's version of the benchmark, not the paper |
| $dt$ | 0.1 | ms | chosen; spikes reach targets one step later |

**Uncertain or changed.** The values above are quoted from memory of the papers, which were
not available here; the 6 nS / 67 nS increments with $g_L$ = 10 nS ($C_m$ = 200 pF) are the
least certain. Brette et al. run the benchmark without input, starting from random
conductances (recalled as $g_E \sim \mathcal N(4, 1.5^2)$, $g_I \sim \mathcal N(20, 12^2)$ in
units of $g_L$); in this implementation that activity died out within ~100 ms, so every
neuron gets the constant 20 mV drive of BrainPy's COBA example instead. `bp.conn.FixedProb`
fixes each neuron's out-degree; in-degrees vary (about 64 ± 8 E inputs).

## What the code does

- `neuromodels/networks/ei_balance.py`: `EINet` holds two `bp.dyn.LifRef` populations and four
  `bp.dyn.FullProjAlignPostMg` projections (E→E, E→I, I→E, I→I), each
  `bp.dnn.EventCSRLinear` on `bp.conn.FixedProb` → `bp.dyn.Expon` → `bp.dyn.COBA`. Align-post
  means one conductance variable per target neuron rather than per synapse. Initial voltages
  $\mathcal N(-55, 2^2)$ mV and wiring come from one seed, and reset restores them.
- `scaled_network(n)` keeps the in-degree fixed ($p = 0.02 \cdot 4000 / n$), which preserves
  each neuron's input statistics; only the shared-input fraction grows.
- `simulate` runs the network through `neuromodels.utils.run` with callable monitors (spikes of
  all neurons; V, $g_E$, $g_I$ of a few), which returns end-of-step times. `network_statistics`
  gives rates, ISI CVs (neurons with ≥ 5 spikes), spike-count correlations (10 ms bins, 200
  neurons) and mean input currents.
- `use_portable_event_kernels`: BrainPy's event-driven operators call `brainevent`, whose CPU
  kernels need Numba; without it this switches them to pure-JAX kernels.
- `scripts/07_ei_balance.py` runs the full network for 1.2 s with and without inhibition
  (float64, ~20 s) and saves `07_ei_balance_{raster,currents,statistics}.png`.

Results (full network, statistics after 200 ms):

| | balanced | inhibition removed |
|---|---|---|
| mean rate E / I | 20.9 / 21.4 Hz | 192 / 192 Hz |
| mean ISI CV | 1.70 | 0.001 |
| pairwise count correlation | 0.003 | 0.92 |
| mean excitation / inhibition / net input | 280 / −267 / 13 mV | 2146 / 0 / 2146 mV |

The CV above 1 means bursts: with only 16 inhibitory inputs of 6.7 $g_L$ each, a neuron
occasionally goes tens of ms without inhibition and fires a train of spikes (visible in the
currents figure). Rates are broad: 17 % of E neurons fire below 1 Hz, a few above 100 Hz.

## What the tests verify

`tests/test_ei_balance.py` (1000 neurons at the full in-degree, float64, ~12 s):

- Uncoupled neurons fire at the closed-form LIF period
  $\tau_\text{ref} + \tau_m \ln\frac{V_\infty - V_\text{reset}}{V_\infty - V_\text{th}} = 18.86$ ms (53 Hz), CV = 0.
- Mean $g_E$ and $g_I$ of each recorded neuron equal $w\tau\sum_j \nu_j$ over its actual
  presynaptic partners (Campbell's theorem, with the exact one-step decay factor), to 2 %.
- Balanced state: rates 2–40 Hz, mean CV > 0.7, correlation < 0.05.
- Cancellation: excitation and inhibition each exceed 10× the 10 mV rest-to-threshold gap;
  |net| / excitation < 0.2 (measured 0.05).
- Without inhibition: rate within 10 % below $1/\tau_\text{ref}$, CV < 0.05, correlations 10× higher.
- Repeated runs of the same network give identical spikes.

## Questions to think about

1. The mean voltage sits near −66 mV, below rest, although the drive alone would hold it at
   −40 mV. Use $V_\text{eff}$ to explain why. What changes if $E_I$ is raised to −60 mV
   (shunting inhibition), and why can shunting inhibition not balance the drive the same way?
2. The tests shrink the network by raising $p$. BrainPy's own examples instead keep $p$ and
   multiply the weights by 4. Compute the mean and variance of $g_I$ under both schemes. Which
   statistic of the network do you expect to move the most?
3. Replace the 16 inhibitory inputs of 6.7 $g_L$ by 64 inputs of 1.675 $g_L$ (same mean). How
   should the CV and the rate distribution change, and why?
4. The benchmark was meant to sustain its own activity without input. Why is self-sustained
   irregular activity fragile in a network of 4000 neurons, and what would you change first
   (weights, delays, size) to make it survive?
