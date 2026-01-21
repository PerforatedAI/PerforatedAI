"""
Cost Control and Budget Management for W&B Sweeps

Provides utilities to prevent runaway compute costs:
- Budget tracking
- Automatic sweep termination
- GPU hour monitoring
"""

import os
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """Budget configuration for training."""
    max_gpu_hours: float = 10.0
    max_runs: int = 100
    max_cost_usd: float = 50.0
    warning_threshold: float = 0.8  # Warn at 80% of budget


class CostTracker:
    """Track training costs and enforce budgets."""
    
    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self.start_time = time.time()
        self.run_count = 0
        self.total_gpu_hours = 0.0
        
    def start_run(self):
        """Mark the start of a training run."""
        self.run_count += 1
        self.run_start = time.time()
        
    def end_run(self, gpu_count: int = 1):
        """Mark the end of a training run and update costs."""
        elapsed_hours = (time.time() - self.run_start) / 3600
        self.total_gpu_hours += elapsed_hours * gpu_count
        
        self._check_budget()
        
    def _check_budget(self):
        """Check if budget limits are exceeded."""
        # Check GPU hours
        if self.total_gpu_hours >= self.config.max_gpu_hours:
            logger.error(f"GPU hour budget exceeded: {self.total_gpu_hours:.2f} >= {self.config.max_gpu_hours}")
            self._terminate_sweep()
            
        elif self.total_gpu_hours >= self.config.max_gpu_hours * self.config.warning_threshold:
            logger.warning(f"GPU hour budget at {self.total_gpu_hours/self.config.max_gpu_hours*100:.0f}%")
        
        # Check run count
        if self.run_count >= self.config.max_runs:
            logger.error(f"Run count budget exceeded: {self.run_count} >= {self.config.max_runs}")
            self._terminate_sweep()
            
    def _terminate_sweep(self):
        """Terminate the current sweep."""
        logger.warning("Terminating sweep due to budget constraints")
        
        try:
            import wandb
            if wandb.run:
                wandb.run.finish(exit_code=1)
        except:
            pass
        
        # Exit process
        raise SystemExit("Budget exceeded - sweep terminated")
    
    def get_status(self) -> dict:
        """Get current cost status."""
        return {
            "run_count": self.run_count,
            "total_gpu_hours": self.total_gpu_hours,
            "budget_gpu_hours": self.config.max_gpu_hours,
            "budget_utilization_pct": self.total_gpu_hours / self.config.max_gpu_hours * 100,
            "elapsed_hours": (time.time() - self.start_time) / 3600,
        }


def estimate_sweep_cost(
    sweep_config: dict,
    avg_run_hours: float = 0.5,
    gpu_cost_per_hour: float = 0.50,
) -> dict:
    """
    Estimate total sweep cost.
    
    Args:
        sweep_config: Sweep configuration dict
        avg_run_hours: Average hours per run
        gpu_cost_per_hour: GPU cost per hour (USD)
        
    Returns:
        Cost estimation dict
    """
    params = sweep_config.get('parameters', {})
    
    # Estimate number of combinations
    total_combinations = 1
    for param_name, param_config in params.items():
        if 'values' in param_config:
            total_combinations *= len(param_config['values'])
        elif param_config.get('distribution') in ['uniform', 'log_uniform', 'log_uniform_values']:
            # For continuous params, estimate ~10 samples
            total_combinations *= 10
    
    method = sweep_config.get('method', 'random')
    if method == 'bayes':
        # Bayesian typically needs fewer runs
        estimated_runs = min(total_combinations, 50)
    elif method == 'grid':
        estimated_runs = total_combinations
    else:
        estimated_runs = min(total_combinations, 100)
    
    total_hours = estimated_runs * avg_run_hours
    total_cost = total_hours * gpu_cost_per_hour
    
    return {
        "method": method,
        "total_combinations": total_combinations,
        "estimated_runs": estimated_runs,
        "estimated_gpu_hours": total_hours,
        "estimated_cost_usd": total_cost,
    }
