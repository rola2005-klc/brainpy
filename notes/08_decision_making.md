# 08 · Decision making: the reduced Wong–Wang model (Days 5–6)

Two populations, each selective for one choice in a random-dot motion task, excite themselves
through slow NMDA synapses and inhibit each other. A stimulus turns the symmetric low-activity
state into a saddle; noise and a small input bias push the network into one of two attractors,
and the time this takes is the decision time. Wong & Wang (2006) reduced the spiking network
of Wang (2002) to the two NMDA gating variables used here.

## Model

$$\frac{dS_i}{dt} = -\frac{S_i}{\tau_S} + (1 - S_i)\,\gamma\,H(x_i), \qquad
H(x) = \frac{a x - b}{1 - e^{-d\,(a x - b)}}$$

$$x_1 = J_{N,11} S_1 - J_{N,12} S_2 + I_0 + I_1 + I_{\text{noise},1}, \qquad
x_2 = J_{N,22} S_2 - J_{N,21} S_1 + I_0 + I_2 + I_{\text{noise},2}$$

$$\tau_\text{AMPA}\,\frac{dI_{\text{noise},i}}{dt} = -I_{\text{noise},i} + \eta_i(t)\,\sqrt{\tau_\text{AMPA}}\,\sigma,
\qquad I_{1,2} = J_{A,\text{ext}}\,\mu_0\left(1 \pm \frac{c'}{100}\right)$$

$\eta_i$ is unit white noise, $c'$ the coherence in %. Setting $dS_i/dt = 0$ fixes the rate,
$H(x_i) = S_i / (\gamma\tau_S(1 - S_i))$; inverting $H$ gives $x_i$, which is linear in the other
$S$. Each nullcline is therefore an explicit curve, and the fixed points are its intersections.

| Parameter | Value | Unit | Source |
|---|---|---|---|
| $a$, $b$, $d$ | 270, 108, 0.154 | Hz/nA, Hz, s | Wong & Wang 2006 |
| $\gamma$, $\tau_S$ | 0.641, 100 | —, ms | Wong & Wang 2006 |
| $J_{N,11} = J_{N,22}$ | 0.2609 | nA | Wong & Wang 2006 |
| $J_{N,12} = J_{N,21}$ | 0.0497 | nA | Wong & Wang 2006 |
| $I_0$ | 0.3255 | nA | Wong & Wang 2006 |
| $J_{A,\text{ext}}$, $\mu_0$ | 0.00052, 30 | nA/Hz, Hz | Wong & Wang 2006 |
| $\tau_\text{AMPA}$, $\sigma$ | 2, 0.02 | ms, nA | Wong & Wang 2006 |
| decision threshold | 15 | Hz | Wong & Wang 2006 (reaction-time task) |
| $c'$ levels | 0, 3.2, …, 51.2 | % | Roitman & Shadlen 2002 |
| $dt$ | 0.1 | ms | chosen: $dt \ll \tau_\text{AMPA}$ |

**Uncertain or changed.** Values are quoted from memory of the papers (not available here);
they agree with common reimplementations, and the threshold is the least certain.
Read literally, the noise equation gives a stationary SD of $\sigma/\sqrt2 \approx 0.014$ nA
although the paper calls $\sigma^2$ the variance; implementations differ by this $\sqrt2$, which
changes the psychometric slope. The code follows the equation. Two choices are this
repository's: the protocol (start at $S_1 = S_2 = 0.1$, 100 ms without stimulus, then 2 s of
stimulus; no non-decision time added) and the read-out, which thresholds $H(x_i)$ *without*
the noise term. On the raw rate (noise SD ≈ 3 Hz, 2 ms correlation time) about 5 % of trials
at $c' \le 6.4$ % crossed 15 Hz near the saddle for the population that went on to lose.

## What the code does

- `neuromodels/networks/decision.py`: in `WongWang2006` (a `bp.dyn.NeuDyn`) each unit is an
  independent trial. One `bp.sdeint` (Euler–Maruyama, suited to additive noise) integrates
  $S_1, S_2, I_{\text{noise},1}, I_{\text{noise},2}$, with diffusion on the noise only.
  `update(mu)` takes the stimulus in Hz and records the first threshold crossing online, so
  thousands of trials need no monitors. `simulate_trials` seeds BrainPy's global generator.
- `summarize`: accuracy and decision times (all / correct / error, ± SEM) per coherence;
  `fit_weibull` fits $P = 1 - \tfrac12 e^{-(c/\alpha)^\beta}$ by grid-search maximum likelihood.
- Phase plane in NumPy: `nullclines` inverts $H$ by bisection; `fixed_points` finds sign
  changes of $f_2(f_1(S_1)) - S_1$ along the $S_1$-nullcline, refines them by bisection and
  classifies them by the Jacobian's eigenvalues. `bp.analysis.PhasePlane2D` was tried: its default
  tolerance lets dozens of near-duplicates through (the drift is only ~$10^{-3}$ ms$^{-1}$); a tight one, none.
- Spiking network (`Wang2002Network`): 1600 E and 400 I conductance-based `bp.dyn.LifRef` cells;
  all-to-all AMPA and GABA_A via `FullProjAlignPostMg`, NMDA (rise, saturation, Mg block) via
  `FullProjAlignPreSDMg`; $w_+ = 1.7$ within A and within B, $w_- = 0.876$ into them from other
  E cells, summed per population by a custom comm. Poisson inputs (2.4 kHz background,
  stimulus 40 Hz $(1 \pm c'/100)$) use an inverse-transform sampler, 4× faster than
  `bm.random.poisson`. Parameters from memory of Wang (2002), E/I: $C$ = 0.5/0.2 nF, $g_L$ = 25/20,
  $g_\text{ext}$ = 2.1/1.62, $g_\text{AMPA}$ = 0.05/0.04, $g_\text{NMDA}$ = 0.165/0.13, $g_\text{GABA}$ =
  1.3/1.0 nS; 0.5 ms delays. The original's fluctuating stimulus rates are left out.
- `scripts/08_decision_making.py`: 1000 trials per coherence, one spiking trial (float64, ~25 s).

Fixed points at $c' = 0$ (rates: winner ≈ 30 Hz, loser ≈ 0.9 Hz, saddle 11.5 Hz):

| stimulus | stable | saddles |
|---|---|---|
| off | (0.103, 0.103) spontaneous, (0.567, 0.032), (0.032, 0.567) | (0.314, 0.056), (0.056, 0.314) |
| $\mu_0$ = 30 Hz | (0.659, 0.052), (0.052, 0.659) | (0.424, 0.424) |

| $c'$ (%) | 0 | 3.2 | 6.4 | 12.8 | 25.6 | 51.2 |
|---|---|---|---|---|---|---|
| fraction correct | 0.47* | 0.71 | 0.86 | 0.98 | 1.00 | 1.00 |
| decision time, correct (ms) | 656* | 605 | 560 | 455 | 340 | 236 |
| decision time, error (ms) | — | 684 | 726 | 785 | — | — |

\*At $c' = 0$: fraction choosing population 1 (95 % range for a fair coin: 0.47–0.53) and all
trials. Weibull fit: $\alpha$ = 5.2 %, $\beta$ = 1.33. No trial changed its mind after crossing.
In the spiking trial ($c'$ = 25.6 %) A climbs from 1.8 to 25 Hz while B drops to 3 Hz, and A
stays near 22 Hz for the second after the stimulus: the choice is held in persistent activity.

## What the tests verify

`tests/test_decision.py` (float64, ~14 s):

- $H$: the 0/0 point equals its limit $1/d$, the large-input asymptote is $a x - b$, the JAX
  and NumPy versions agree, and `inverse_transfer` inverts it.
- Stimulus on, $c' = 0$: exactly two stable, mirror-image fixed points and one saddle on the
  diagonal (residual drift < $10^{-14}$); the saddle fires below the 15 Hz threshold, so a
  crossing means the state has passed it. Stimulus off: three stable states, two saddles.
- With $\sigma = 0$ the simulation ends on the analysis' fixed points: the population-1
  attractor at $c' = 6.4$ %, and the saddle at $c' = 0$, where the diagonal is invariant.
- The noise's stationary SD equals $\sigma/\sqrt{2 - dt/\tau_\text{AMPA}}$, exact for Euler–Maruyama.
- $c' = 0$: P(choice 1) within 3 binomial standard errors of 0.5 (800 trials). Accuracy never
  falls as coherence rises (> 0.99 at 51.2 %); mean decision time drops by more than 2 SE per
  step; the threshold winner is ahead at the end of every trial, and settled trials sit at
  an attractor (winner $S > 0.5$, loser $S < 0.15$).
- Spiking network (smoke test): without stimulus, E cells settle at 0.5–6 Hz, interneurons faster.

## Questions to think about

1. The threshold (15 Hz) sits above the saddle's rate (11.5 Hz). Predict accuracy, decision
   times and changes of mind with a 10 Hz threshold. Why does thresholding the noisy rate
   act like lowering the threshold?
2. At $\mu_0$ = 60 Hz, `fixed_points(0, 60)` finds five fixed points and the symmetric state
   (both populations near 23 Hz) is stable. What happens on a $c' = 0$ trial, and what
   behavioural failure does this predict for strong stimuli?
3. Wherever errors occur ($0 < c' \le 12.8$ %), they are slower than correct choices. Explain
   this from the phase plane: which noisy trajectories end on the wrong side of the saddle's
   stable manifold, and why do they take longer?
4. The fixed points depend on $\gamma$ and $\tau_S$ only through $\gamma\tau_S$. Divide $\tau_S$ by
   50 and multiply $\gamma$ by 50: the fixed points stay put. What happens to decision times
   and accuracy, and why does Wang (2002) argue that integration needs NMDA?
