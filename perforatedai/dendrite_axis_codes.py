# Copyright (c) 2025 Perforated AI
"""Semantic codes for the entries of a module's output-dimensions specification.

A module's ``output_dimensions`` (the global default lives in
``globals_perforatedai.PAIConfig.output_dimensions``) and the per-module
``this_output_dimensions`` buffer are lists with one entry per tensor axis. Each
entry is one of the fixed codes defined here, describing how the covariance
machinery treats that axis:

- ``REDUCE_AXIS`` (``-1``): a pure sample axis. It is summed/averaged away, so it
  does not appear in the per-node score. This is the batch axis, and by default
  the spatial height and width of a convolution.
- ``NODE_AXIS`` (``0``): the single node/channel axis. Exactly one entry must be
  ``NODE_AXIS``; its length is the number of output nodes (``out_channels``), and
  the covariance is computed to produce one score per position along it. Its
  index is cached as ``this_node_index``.
- ``NOT_REDUCE_OR_NODE_AXIS`` (``1``): an additional retained axis. It is kept
  (never reduced), but it is not the node axis; each position along it receives
  its own score, so the result carries one covariance value per combination of
  node index and retained-axis position.

These are fixed constants, not per-run configuration. This module intentionally
imports nothing so it can be imported from anywhere in either the ``perforatedai``
base package or the optional ``perforatedbp`` plugin without creating an import
cycle.
"""

__all__ = ["REDUCE_AXIS", "NODE_AXIS", "NOT_REDUCE_OR_NODE_AXIS"]

REDUCE_AXIS: int = -1
NODE_AXIS: int = 0
NOT_REDUCE_OR_NODE_AXIS: int = 1
