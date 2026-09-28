"""Run every test in float64 on the CPU so numerical checks are reproducible.

BrainPy defaults to float32, which is fine for large simulations but can shift
a spike by one step in long runs; the tests compare against closed-form
results and independent implementations, so they need double precision.

The matrices here are small, so BLAS runs single-threaded: parallel BLAS threads
only contend with each other on a busy machine (4x slower tests under load).
These lines must run before NumPy or JAX is imported.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import brainpy.math as bm  # noqa: E402

bm.set_platform("cpu")
bm.enable_x64()
