"""Run every test in float64 on the CPU so numerical checks are reproducible.

BrainPy defaults to float32, which is fine for large simulations but can shift
a spike by one step in long runs; the tests compare against closed-form
results and independent implementations, so they need double precision.
"""

import brainpy.math as bm

bm.set_platform("cpu")
bm.enable_x64()
