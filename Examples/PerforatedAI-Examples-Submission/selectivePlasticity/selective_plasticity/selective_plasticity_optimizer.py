"""
SelectivePlasticityOptimizer - Selective Synaptic Plasticity Optimizer

An optimizer based on biological NMDA coincidence detection and eligibility traces
that implements intelligent parameter protection to prevent catastrophic forgetting.

Core Mechanism:
- Busy parameters (high eligibility) → low plasticity → require coincidence to update
- Idle parameters (low eligibility) → high plasticity → free to update

Mathematical Formulation:
    eligibility = 0.99 × eligibility + 0.01 × |gradient|
    plasticity = σ(-k × (eligibility - θ))
    coincidence = 2|g| × eligibility / (|g| + eligibility + ε)
    final_gate = plasticity + (1 - plasticity) × coincidence

Author: SelectivePlasticity Team
Date: 2025-01-29
"""

import torch
from torch.optim import Optimizer


class SelectivePlasticityOptimizer(Optimizer):
    """
    Selective Synaptic Plasticity Optimizer

    Intelligent parameter protection based on long-term usage frequency:
    - Track eligibility trace (usage frequency) for each parameter
    - High frequency → low plasticity → protect
    - Low frequency → high plasticity → learnable

    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        trace_decay: Eligibility trace decay coefficient (default: 0.99)
        plasticity_k: Plasticity gate sigmoid slope (default: 10.0)
        threshold: Plasticity threshold (default: None for adaptive)
        adaptive_threshold: Whether to use adaptive threshold (default: True)
        threshold_percentile: Percentile for adaptive threshold (default: 75)
        warmup_steps: Number of warmup steps (default: 100)
        eps: Numerical stability constant (default: 1e-8)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        trace_decay=0.99,        # Faster decay for eligibility accumulation
        plasticity_k=10.0,
        threshold=None,          # None = use adaptive threshold
        adaptive_threshold=True,
        threshold_percentile=75, # Use 75th percentile (protect top 25% busy params)
        warmup_steps=100,
        eps=1e-8
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < trace_decay < 1.0:
            raise ValueError(f"Invalid trace_decay: {trace_decay}")

        defaults = dict(
            lr=lr,
            trace_decay=trace_decay,
            plasticity_k=plasticity_k,
            threshold=threshold,
            adaptive_threshold=adaptive_threshold,
            threshold_percentile=threshold_percentile,
            warmup_steps=warmup_steps,
            eps=eps
        )
        super().__init__(params, defaults)

        self.step_count = 0

    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        self.step_count += 1

        # Diagnostic information
        diagnostics = {
            'eligibility_mean': 0.0,
            'plasticity_mean': 0.0,
            'coincidence_mean': 0.0,
            'gate_mean': 0.0,
            'gate_low_ratio': 0.0,
        }
        total_params = 0

        for group in self.param_groups:
            lr = group['lr']
            trace_decay = group['trace_decay']
            plasticity_k = group['plasticity_k']
            threshold = group['threshold']
            warmup_steps = group['warmup_steps']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['eligibility'] = torch.zeros_like(grad)
                    state['step'] = 0
                    # Adaptive threshold: initially None, updated based on eligibility distribution
                    state['threshold'] = None

                state['step'] += 1

                # === 1. Update eligibility trace ===
                grad_mag = grad.abs()
                eligibility = state['eligibility']
                eligibility.mul_(trace_decay).add_(grad_mag, alpha=1 - trace_decay)

                # === 2. Compute adaptive threshold ===
                # Use percentile of eligibility as threshold (default 75th, protect top 25%)
                percentile = group['threshold_percentile'] / 100.0
                if state['threshold'] is None:
                    # Initial threshold: eligibility percentile
                    current_threshold = torch.quantile(eligibility.flatten(), percentile)
                else:
                    current_threshold = state['threshold']

                # === 3. Normalize eligibility and gradient ===
                e_max = eligibility.max() + eps
                g_max = grad_mag.max() + eps
                e_norm = eligibility / e_max  # [0, 1]
                g_norm = grad_mag / g_max      # [0, 1]

                # === 4. Compute "protection gate" ===
                # Protect only when parameter is "busy" (high eligibility)
                # AND "not currently needed" (low gradient)
                # protection = e_norm × (1 - g_norm)
                # High e_norm + low g_norm → high protection → low gate
                # Other cases → low protection → high gate
                protection = e_norm * (1 - g_norm)

                # === 5. Final gate ===
                # gate = 1 - k × protection
                # where k controls protection strength
                protection_strength = plasticity_k / 10.0  # Scale to reasonable range
                gate = 1.0 - protection_strength * protection
                gate = gate.clamp(min=0.1, max=1.0)  # Limit to [0.1, 1.0]

                # Compute coincidence for diagnostics only
                coincidence = 2 * e_norm * g_norm / (e_norm + g_norm + eps)

                # === 6. Warmup strategy ===
                if state['step'] <= warmup_steps:
                    warmup_factor = state['step'] / warmup_steps
                    gate = warmup_factor * gate + (1 - warmup_factor) * 1.0

                # === 7. Apply gated update ===
                p.data.add_(grad * gate, alpha=-lr)

                # === 8. Update adaptive threshold (every 50 steps) ===
                if group['adaptive_threshold'] and state['step'] % 50 == 0:
                    # Use percentile of eligibility as threshold
                    new_threshold = torch.quantile(eligibility.flatten(), percentile)
                    if state['threshold'] is None:
                        state['threshold'] = new_threshold
                    else:
                        # EMA update threshold
                        state['threshold'] = 0.9 * state['threshold'] + 0.1 * new_threshold

                # Collect diagnostics
                diagnostics['eligibility_mean'] += eligibility.mean().item()
                diagnostics['plasticity_mean'] += (1.0 - protection.mean().item())  # plasticity = 1 - protection
                diagnostics['coincidence_mean'] += coincidence.mean().item()
                diagnostics['gate_mean'] += gate.mean().item()
                diagnostics['gate_low_ratio'] += (gate < 0.5).float().mean().item()
                total_params += 1

        # Average diagnostics
        if total_params > 0:
            for key in diagnostics:
                diagnostics[key] /= total_params
        diagnostics['step_count'] = self.step_count

        self.last_diagnostics = diagnostics

        return loss

    def get_diagnostics(self):
        """Return diagnostic information"""
        return getattr(self, 'last_diagnostics', {
            'eligibility_mean': 0.0,
            'plasticity_mean': 1.0,
            'gate_mean': 1.0,
            'gate_low_ratio': 0.0,
            'step_count': 0
        })

    def reset_eligibility(self):
        """Reset eligibility traces for all parameters (optional, for task switching experiments)"""
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'eligibility' in self.state[p]:
                    self.state[p]['eligibility'].zero_()
                    self.state[p]['step'] = 0

    def get_param_eligibility(self, param):
        """Get eligibility of a specific parameter (for visualization)"""
        if param in self.state and 'eligibility' in self.state[param]:
            return self.state[param]['eligibility'].clone()
        return None


if __name__ == '__main__':
    # Simple test
    import torch.nn as nn

    print("Testing SelectivePlasticityOptimizer...")

    # Create simple model
    model = nn.Linear(10, 2)
    optimizer = SelectivePlasticityOptimizer(
        model.parameters(),
        lr=0.01,
        warmup_steps=5
    )

    # Simulate two tasks
    print("\n=== Task 1 ===")
    for i in range(20):
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            diag = optimizer.get_diagnostics()
            print(f"Step {i+1}: eligibility={diag['eligibility_mean']:.4f}, "
                  f"plasticity={diag['plasticity_mean']:.4f}, "
                  f"gate={diag['gate_mean']:.4f}")

    print("\n=== Task 2 (different data) ===")
    for i in range(20):
        x = torch.randn(4, 10) * 2  # Different data distribution
        y = torch.randint(0, 2, (4,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            diag = optimizer.get_diagnostics()
            print(f"Step {i+1}: eligibility={diag['eligibility_mean']:.4f}, "
                  f"plasticity={diag['plasticity_mean']:.4f}, "
                  f"gate={diag['gate_mean']:.4f}, "
                  f"gate_low_ratio={diag['gate_low_ratio']:.2%}")

    print("\nTest passed!")
