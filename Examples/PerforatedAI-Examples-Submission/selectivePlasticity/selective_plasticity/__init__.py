"""
SelectivePlasticity - NMDA-inspired synaptic protection for continual learning

A biologically-grounded optimizer that prevents catastrophic forgetting through
selective parameter protection based on usage patterns.
"""

from .selective_plasticity_optimizer import SelectivePlasticityOptimizer

__all__ = ['SelectivePlasticityOptimizer']
__version__ = '1.0.0'
