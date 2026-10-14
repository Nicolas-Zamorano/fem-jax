"""
A lightweight finite element library implemented with JAX.

Importing this package enables 64-bit precision in JAX (``jax_enable_x64``).
This must happen before any array is created, so it is done once at import
time. Convergence studies need float64 to observe the expected rates without
float32 round-off masking them.
"""

import jax

jax.config.update("jax_enable_x64", True)

__all__: list[str] = []
