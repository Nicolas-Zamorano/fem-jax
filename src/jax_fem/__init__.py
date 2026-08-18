"""A lightweight finite element library implemented with JAX.

See ``.claude/JAX_FEM_Library_Architecture.md`` for the full architecture
specification this package follows.

Importing this package enables 64-bit precision in JAX (``jax_enable_x64``).
This must happen before any other JAX computation, and before any array is
created anywhere in the process, so it is done once here at package import
time rather than left for users to configure. FEM convergence studies rely
on float64 to see expected a priori error rates without float32 roundoff
masking them.
"""

import jax

jax.config.update("jax_enable_x64", True)

__all__: list[str] = []
