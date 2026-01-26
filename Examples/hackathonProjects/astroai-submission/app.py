"""
AstroAI: Interactive Exoplanet Detection Simulator
Enhanced with Perforated AI Dendritic Optimization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulator import simulate_light_curve, simulate_transit_light_curve
from model import TransitDetector, TransitDetectorCNN
import torch
import os

st.set_page_config(
    page_title="AstroAI - Exoplanet Detection",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 AstroAI: Exoplanet Detection Simulator")
st.markdown("""
**Powered by Perforated AI Dendritic Optimization**

This interactive tool simulates exoplanet detection using synthetic telescope data and AI.
The neural network models can be enhanced with Perforated AI's dendritic optimization for improved accuracy.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Architecture",
    ["MLP (TransitDetector)", "CNN (TransitDetectorCNN)"],
    help="Select the neural network architecture"
)

# Light curve parameters
st.sidebar.subheader("Light Curve Parameters")
period = st.sidebar.slider("Orbital Period (days)", 1, 30, 10)
radius_ratio = st.sidebar.slider("Planet Radius Ratio (Rp/Rs)", 0.01, 0.2, 0.1)
inclination = st.sidebar.slider("Inclination (degrees)", 80, 90, 90)
noise_level = st.sidebar.slider("Noise Level", 0.001, 0.05, 0.01)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Light Curve Simulation")
    
    if st.button("🌟 Simulate Light Curve", type="primary"):
        t, flux = simulate_light_curve(period, radius_ratio, inclination, noise_level)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, flux, 'b-', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f'Simulated Exoplanet Transit Light Curve\n(Period={period}d, Rp/Rs={radius_ratio:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Highlight transit region
        transit_mask = flux < 0.999
        if np.any(transit_mask):
            ax.fill_between(t, flux, 1, where=transit_mask, alpha=0.3, color='red', label='Transit')
            ax.legend()
        
        st.pyplot(fig)
        
        # Save for assets
        os.makedirs('assets', exist_ok=True)
        fig.savefig('assets/sample_light_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Display statistics
        st.markdown("**Light Curve Statistics:**")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Transit Depth", f"{(1 - flux.min()) * 100:.3f}%")
        col_b.metric("SNR", f"{(1 - flux.min()) / noise_level:.1f}")
        col_c.metric("Data Points", len(flux))

with col2:
    st.subheader("🤖 AI Transit Detection")
    
    if st.button("🔍 Detect Transit", type="secondary"):
        # Generate light curve
        t, flux = simulate_light_curve(period, radius_ratio, inclination, noise_level)
        
        # Load model
        if "MLP" in model_type:
            model = TransitDetector(input_size=len(flux))
        else:
            model = TransitDetectorCNN(input_size=len(flux))
        
        model.eval()
        
        # Prepare input
        input_data = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            prob = model(input_data).item()
        
        # Display result
        st.markdown("**Detection Result:**")
        
        # Progress bar for probability
        st.progress(prob)
        
        if prob > 0.5:
            st.success(f"✅ Transit Detected! Probability: {prob:.2%}")
        else:
            st.warning(f"❌ No Transit Detected. Probability: {prob:.2%}")
        
        # Confidence visualization
        fig, ax = plt.subplots(figsize=(6, 2))
        colors = ['#e74c3c' if prob < 0.5 else '#27ae60']
        ax.barh(['Transit Probability'], [prob], color=colors)
        ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        st.info(f"Model: {model_type} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Educational Section
st.markdown("---")
st.header("📚 Learn More")

tab1, tab2, tab3 = st.tabs(["Kepler's Laws", "Transit Method", "Perforated AI"])

with tab1:
    st.markdown("""
    ### Kepler's Laws of Planetary Motion
    
    **First Law (Law of Ellipses):** Planets orbit the Sun in elliptical paths with the Sun at one focus.
    
    **Second Law (Law of Equal Areas):** A line connecting a planet to the Sun sweeps out equal areas in equal times.
    
    **Third Law (Harmonic Law):** The square of a planet's orbital period is proportional to the cube of its semi-major axis.
    
    $$P^2 \\propto a^3$$
    
    Where:
    - P = Orbital period
    - a = Semi-major axis
    """)

with tab2:
    st.markdown("""
    ### Transit Photometry
    
    When an exoplanet passes in front of its host star (a "transit"), it blocks a small fraction of the star's light.
    
    **Transit Depth:**
    $$\\delta = \\left(\\frac{R_p}{R_s}\\right)^2$$
    
    Where:
    - δ = Transit depth (fractional flux decrease)
    - Rp = Planet radius
    - Rs = Star radius
    
    **Example:** A Jupiter-sized planet transiting a Sun-like star produces a ~1% dip in brightness.
    """)

with tab3:
    st.markdown("""
    ### Perforated AI Dendritic Optimization
    
    **What are Artificial Dendrites?**
    
    In biological neurons, dendrites perform additional computation before signals reach the cell body. 
    Perforated AI adds artificial dendrites to neural networks, enabling:
    
    - 🎯 **Improved Accuracy**: Better feature representation
    - ⚡ **Efficient Learning**: Automatic architecture adaptation
    - 🔄 **Dynamic Optimization**: Dendrites added where needed during training
    
    **How It Works:**
    1. Initialize your PyTorch model with PAI
    2. Train normally - PAI handles dendrite addition
    3. Achieve better results with minimal code changes
    
    [Learn more at Perforated AI](https://www.perforatedai.com/docs)
    """)

# Batch Analysis
st.markdown("---")
st.header("📈 Batch Analysis")

n_samples = st.slider("Number of samples to analyze", 10, 100, 50)

if st.button("🚀 Run Batch Analysis"):
    with st.spinner("Generating and analyzing light curves..."):
        results = []
        
        progress_bar = st.progress(0)
        
        for i in range(n_samples):
            # Random parameters
            has_transit = np.random.random() > 0.5
            flux, label = simulate_transit_light_curve(has_transit=has_transit)
            
            # Model prediction
            model = TransitDetector(input_size=len(flux))
            model.eval()
            input_data = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                prob = model(input_data).item()
            
            predicted = 1 if prob > 0.5 else 0
            results.append({
                'actual': label,
                'predicted': predicted,
                'probability': prob,
                'correct': label == predicted
            })
            
            progress_bar.progress((i + 1) / n_samples)
        
        # Calculate metrics
        accuracy = sum(r['correct'] for r in results) / len(results)
        true_positives = sum(1 for r in results if r['actual'] == 1 and r['predicted'] == 1)
        false_positives = sum(1 for r in results if r['actual'] == 0 and r['predicted'] == 1)
        true_negatives = sum(1 for r in results if r['actual'] == 0 and r['predicted'] == 0)
        false_negatives = sum(1 for r in results if r['actual'] == 1 and r['predicted'] == 0)
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.1%}")
        col2.metric("True Positives", true_positives)
        col3.metric("False Positives", false_positives)
        col4.metric("Samples", n_samples)
        
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = [[true_negatives, false_positives], [false_negatives, true_positives]]
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Transit', 'Transit'])
        ax.set_yticklabels(['No Transit', 'Transit'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha='center', va='center', fontsize=20)
        
        plt.colorbar(im)
        st.pyplot(fig)
        plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ for the <a href='https://pytorch-dendritic-optimization.devpost.com/'>PyTorch Dendritic Optimization Hackathon</a></p>
    <p>Powered by <a href='https://www.perforatedai.com/'>Perforated AI</a></p>
</div>
""", unsafe_allow_html=True)
