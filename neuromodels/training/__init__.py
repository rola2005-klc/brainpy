"""Training dynamical networks with gradient descent through time.

``rnn``: a continuous-time rate RNN trained with backpropagation through time
on a perceptual decision task (Song, Yang & Wang 2016).
``snn``: a leaky integrate-and-fire network trained with surrogate gradients on
latency-coded spike patterns (Neftci, Mostafa & Zenke 2019).
"""

from . import rnn, snn

__all__ = ["rnn", "snn"]
