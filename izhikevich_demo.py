"""Minimal spiking-neuron simulation for the BrainPy learning repo.

This script is intentionally dependency-light so the repository has at least
one reproducible computational-neuroscience example that runs from a fresh
Python environment. It implements the Izhikevich regular-spiking neuron model
with NumPy and saves a voltage trace plot.

Run:
    python izhikevich_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate_izhikevich(total_ms: int = 250, dt: float = 0.25):
    """Simulate a regular-spiking Izhikevich neuron.

    Returns time points, membrane voltage, recovery variable, and spike times.
    """

    # Regular-spiking cortical neuron parameters from Izhikevich 2003.
    a = 0.02
    b = 0.2
    c = -65.0
    d = 8.0

    times = np.arange(0, total_ms, dt)
    voltage = np.empty_like(times, dtype=float)
    recovery = np.empty_like(times, dtype=float)
    spikes = []

    v = -65.0
    u = b * v

    for i, t in enumerate(times):
        # Step current: quiet baseline, then stimulation.
        current = 10.0 if 50 <= t <= 200 else 0.0

        if v >= 30.0:
            voltage[i] = 30.0
            v = c
            u += d
            spikes.append(t)
        else:
            voltage[i] = v

        recovery[i] = u
        dv = 0.04 * v * v + 5 * v + 140 - u + current
        du = a * (b * v - u)
        v += dt * dv
        u += dt * du

    return times, voltage, recovery, np.array(spikes)


def main():
    times, voltage, _, spikes = simulate_izhikevich()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "izhikevich_voltage_trace.png"

    plt.figure(figsize=(9, 4))
    plt.plot(times, voltage, color="#3b82f6", linewidth=1.5)
    plt.axvspan(50, 200, color="#f59e0b", alpha=0.15, label="input current")
    plt.title("Izhikevich regular-spiking neuron")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane voltage (mV)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)

    print(f"Simulated {len(times)} time steps")
    print(f"Spike count: {len(spikes)}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
