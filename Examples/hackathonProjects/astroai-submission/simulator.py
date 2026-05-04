"""
AstroAI Light Curve Simulator
Generates synthetic exoplanet transit light curves for training and testing.
"""

import numpy as np
import torch
from astropy.modeling import models


def simulate_light_curve(period=10, radius_ratio=0.1, inclination=90, noise_level=0.01, num_points=1000):
    """
    Simulate a single exoplanet transit light curve.
    
    Args:
        period: Orbital period in days
        radius_ratio: Planet-to-star radius ratio (Rp/Rs)
        inclination: Orbital inclination in degrees
        noise_level: Standard deviation of Gaussian noise
        num_points: Number of data points
    
    Returns:
        t: Time array
        flux: Normalized flux array
    """
    # Time array
    t = np.linspace(0, period * 2, num_points)
    
    # Simple transit model using Trapezoid1D with correct parameters
    # Trapezoid1D params: amplitude, x_0, width, slope
    transit_depth = radius_ratio ** 2
    transit_width = 0.1  # 10% of phase
    transit = models.Trapezoid1D(
        amplitude=-transit_depth,
        x_0=0.0,  # Center at phase 0
        width=transit_width,
        slope=0.01  # Ingress/egress slope
    )
    phase = (t % period) / period - 0.5
    flux = 1 + transit(phase)  # Normalized flux
    
    # Add noise (Gaussian)
    flux += np.random.normal(0, noise_level, num_points)
    
    return t, flux


def simulate_transit_light_curve(
    num_points=1000,
    period=None,
    radius_ratio=None,
    inclination=None,
    noise_level=None,
    has_transit=True
):
    """
    Generate a light curve with or without a transit signal.
    
    Args:
        num_points: Number of data points
        period: Orbital period (random if None)
        radius_ratio: Planet-to-star radius ratio (random if None)
        inclination: Orbital inclination (random if None)
        noise_level: Noise level (random if None)
        has_transit: Whether to include a transit signal
    
    Returns:
        flux: Normalized flux array
        label: 1 if transit present, 0 otherwise
    """
    # Randomize parameters if not provided
    if period is None:
        period = np.random.uniform(2, 30)
    if radius_ratio is None:
        radius_ratio = np.random.uniform(0.02, 0.15)
    if inclination is None:
        inclination = np.random.uniform(85, 90)
    if noise_level is None:
        noise_level = np.random.uniform(0.001, 0.02)
    
    # Time array covering multiple periods
    t = np.linspace(0, period * 3, num_points)
    
    if has_transit:
        # Calculate transit depth based on radius ratio
        transit_depth = radius_ratio ** 2
        
        # Transit duration (simplified model)
        transit_duration = 0.05 * period  # ~5% of period
        
        # Create transit signal
        flux = np.ones(num_points)
        phase = (t % period) / period
        
        # Transit occurs around phase 0.5
        transit_center = 0.5
        half_duration = transit_duration / (2 * period)
        
        # Apply transit dip
        in_transit = np.abs(phase - transit_center) < half_duration
        
        # Smooth transit shape (limb darkening approximation)
        transit_phase = (phase[in_transit] - transit_center) / half_duration
        limb_factor = 1 - 0.3 * (1 - np.sqrt(1 - transit_phase**2))
        flux[in_transit] = 1 - transit_depth * limb_factor
        
        label = 1
    else:
        # No transit - just noise
        flux = np.ones(num_points)
        label = 0
    
    # Add realistic noise
    flux += np.random.normal(0, noise_level, num_points)
    
    # Add some systematic trends (stellar variability)
    trend = 0.001 * np.sin(2 * np.pi * t / (period * 5))
    flux += trend
    
    return flux, label


def generate_dataset(n_samples=5000, num_points=1000, train_ratio=0.7, val_ratio=0.15):
    """
    Generate a complete dataset for training, validation, and testing.
    
    Args:
        n_samples: Total number of samples
        num_points: Points per light curve
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test as torch tensors
    """
    X = []
    y = []
    
    # Generate balanced dataset
    for i in range(n_samples):
        has_transit = i < n_samples // 2  # 50% with transit, 50% without
        flux, label = simulate_transit_light_curve(
            num_points=num_points,
            has_transit=has_transit
        )
        X.append(flux)
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Split
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = torch.tensor(X[:train_end])
    y_train = torch.tensor(y[:train_end])
    X_val = torch.tensor(X[train_end:val_end])
    y_val = torch.tensor(y[train_end:val_end])
    X_test = torch.tensor(X[val_end:])
    y_test = torch.tensor(y[val_end:])
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_sample_curves(n_samples=5, save_path=None):
    """Generate sample light curves for visualization."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 6))
    
    for i in range(n_samples):
        # Transit curve
        period = np.random.uniform(5, 15)
        t, flux = simulate_light_curve(
            period=period,
            radius_ratio=np.random.uniform(0.05, 0.12),
            noise_level=np.random.uniform(0.005, 0.015)
        )
        axes[0, i].plot(t, flux, 'b-', alpha=0.7, linewidth=0.5)
        axes[0, i].set_title(f'Transit (P={period:.1f}d)')
        axes[0, i].set_xlabel('Time (days)')
        axes[0, i].set_ylabel('Normalized Flux')
        
        # Non-transit curve
        flux_no_transit, _ = simulate_transit_light_curve(has_transit=False)
        axes[1, i].plot(flux_no_transit, 'r-', alpha=0.7, linewidth=0.5)
        axes[1, i].set_title('No Transit')
        axes[1, i].set_xlabel('Data Point')
        axes[1, i].set_ylabel('Normalized Flux')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig
