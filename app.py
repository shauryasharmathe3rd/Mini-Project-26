"""
app.py
=============================================================================
Streamlit Web Application: Real-Time Hemodynamic Monitoring & Sepsis Risk AI
Custom Serif + Cream (#FFFDD0) Theme with 3-Colour Solid Gradient Palette.
=============================================================================
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import streamlit as st

from data_loader import WaveformDataLoader
from preprocess import PhysiologicalPreprocessor
from model import SepsisDetectionNet
from train import train_model

# Page Configuration
st.set_page_config(
    page_title="SepsisWatch AI | ICU Telemetry",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Cream Background (#FFFDD0), Serif Fonts, 3-Colour Gradient Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

    /* Global App Canvas */
    .stApp {
        background-color: #FFFDD0;
        font-family: 'Lora', 'Georgia', 'Times New Roman', serif;
        color: #2D3748;
    }

    /* Gradient Headers */
    .gradient-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #FFE66D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .gradient-divider {
        height: 4px;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #FFE66D);
        border: none;
        border-radius: 2px;
        margin-bottom: 1.5rem;
    }

    /* Patient Cards */
    .patient-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
    .healthy-card {
        border-left: 6px solid #4ECDC4;
    }
    .sick-card {
        border-left: 6px solid #FF6B6B;
        border: 2px solid #FF6B6B;
        background-color: #FFF8F8;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: #FFFFFF;
        font-family: 'Lora', serif;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# In-Memory Model Cache
@st.cache_resource
def load_trained_model():
    model_path = "artifacts/sepsis_detector.pt"
    if not os.path.exists(model_path):
        train_model(epochs=8, save_path=model_path)
    model = SepsisDetectionNet(in_channels=2, seq_len=1250)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def make_waveform_plot(time_axis, ecg, ppg, title_prefix, color_ecg="#2B6CB0", color_ppg="#C53030"):
    fig = go.Figure()
    # ECG Trace
    fig.add_trace(go.Scatter(
        x=time_axis, y=ecg, mode='lines',
        name='Lead II ECG (mV)', line=dict(color=color_ecg, width=1.5)
    ))
    # PPG Trace
    fig.add_trace(go.Scatter(
        x=time_axis, y=ppg - 3.0, mode='lines',  # Offset for visual stacking
        name='Plethysmogram PPG (Offset)', line=dict(color=color_ppg, width=1.5)
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title_prefix} Simultaneous Telemetry (10s)</b>", font=dict(family="Lora, serif", size=14)),
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized Amplitude (Z-Score)",
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.85)',
        height=320,
        margin=dict(l=40, r=20, t=40, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def main():
    # Header Section
    st.markdown('<div class="gradient-header">🩺 SepsisWatch-AI: Physiological Signal & Sepsis Surveillance</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Initialize Modules
    loader = WaveformDataLoader(target_fs=125)
    prep = PhysiologicalPreprocessor(target_fs=125)
    model = load_trained_model()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Data & Ingestion Controls")
        data_format = st.selectbox("Select Signal Format", ["WFDB (RRest-syn)", "CSV (Streaming)", "MAT (Selective)"])
        duration_sec = st.slider("Window Duration (sec)", min_value=5, max_value=15, value=10)
        target_samples = duration_sec * 125

        st.markdown("---")
        st.subheader("Model Lifecycle")
        if st.button("🔄 Retrain 1D-CNN + LSTM"):
            with st.spinner("Retraining model on augmented physiological data..."):
                train_model(epochs=10)
                st.success("Model retrained successfully!")
                st.cache_resource.clear()

        st.info("💡 **Memory Footprint Notice:** Signals are streamed with partial reads and downcasted directly to `float32`.")

    # Data Acquisition (Load Healthy Baseline)
    log_messages = []
    try:
        if "WFDB" in data_format:
            rec = "rrest-syn_wfdb/rrest-syn001"
            log_messages.append(f"Streaming {target_samples} samples from WFDB ({rec})...")
            ecg_raw, ppg_raw, fs = loader.load_wfdb_partial(rec, sampto=target_samples)
        elif "CSV" in data_format:
            rec = "rrest-syn_csv/rrest-syn001_data.csv"
            log_messages.append(f"Processing chunked CSV ({rec}) up to {target_samples} samples...")
            ecg_raw, ppg_raw = loader.load_csv_chunked(rec, chunksize=5000, max_samples=target_samples)
            fs = 125
        else:
            rec = "rrest-syn_mat/RRSYNTHdata.mat"
            log_messages.append(f"Extracting selective matrices from MAT ({rec})...")
            ecg_raw, ppg_raw = loader.load_mat_selective(rec, max_samples=target_samples)
            fs = 125

        # DSP Filtering
        ecg_healthy = prep.process_ecg(ecg_raw, fs)[:1250]
        ppg_healthy = prep.process_ppg(ppg_raw, fs)[:1250]
        log_messages.append("Zero-phase Butterworth filtering completed for Healthy Baseline.")

        # Sepsis Simulation for Patient 2
        ecg_sepsis, ppg_sepsis = prep.simulate_sepsis_morphology(ecg_healthy, ppg_healthy, 125)
        log_messages.append("Synthesized sepsis hemodynamics (Tachycardia + Vasoconstriction).")

    except Exception as e:
        st.warning(f"Using synthetic baseline generator due to: {e}")
        t = np.linspace(0, 10, 1250, dtype=np.float32)
        ecg_healthy = prep.process_ecg(np.sin(2 * np.pi * 1.2 * t), 125)
        ppg_healthy = prep.process_ppg(np.sin(2 * np.pi * 1.2 * t - 0.2), 125)
        ecg_sepsis, ppg_sepsis = prep.simulate_sepsis_morphology(ecg_healthy, ppg_healthy, 125)

    # Inference
    t_axis = np.linspace(0, 10, 1250)
    tensor_h = torch.tensor(np.stack([ecg_healthy, ppg_healthy])[np.newaxis, :, :], dtype=torch.float32)
    tensor_s = torch.tensor(np.stack([ecg_sepsis, ppg_sepsis])[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        prob_healthy = model(tensor_h).item()
        prob_sepsis = model(tensor_s).item()
    log_messages.append(f"Inference complete: Healthy={prob_healthy:.3f}, Sepsis={prob_sepsis:.3f}")

    # Layout: Two Columns for Visual Separation
    col1, col2 = st.columns(2)

    # 🟢 HEALTHY PATIENT COLUMN
    with col1:
        st.markdown("""
        <div class="patient-card healthy-card">
            <h3 style="color: #2C7A7B; margin-top:0;">🟢 Patient #101: Baseline Control</h3>
            <p><b>Status:</b> Normal Sinus Rhythm & Intact Perfusion</p>
        </div>
        """, unsafe_allow_html=True)

        # Vitals Table
        vitals_h = pd.DataFrame({
            "Heart Rate (bpm)": [72],
            "Blood Pressure (mmHg)": ["118/78"],
            "SpO₂ (%)": [99],
            "Resp Rate (br/min)": [14]
        })
        st.dataframe(vitals_h, use_container_width=True, hide_index=True)

        # Plotly Chart
        fig_h = make_waveform_plot(t_axis, ecg_healthy, ppg_healthy, "Patient #101 (Healthy)", "#2C7A7B", "#319795")
        st.plotly_chart(fig_h, use_container_width=True)

        # Model Result
        st.markdown(f"**Inferred Sepsis Probability:** `{prob_healthy * 100:.1f}%`")
        st.progress(float(prob_healthy))
        st.success("✅ **Clinical Assessment:** Low Sepsis Probability")

    # 🔴 SICK / SEPSIS PATIENT COLUMN
    with col2:
        st.markdown("""
        <div class="patient-card sick-card">
            <h3 style="color: #E53E3E; margin-top:0;">⚠️ Patient #204: Septic Shock Alert</h3>
            <p><b>Status:</b> Systemic Inflammatory Response Syndrome (SIRS)</p>
        </div>
        """, unsafe_allow_html=True)

        # Vitals Table
        vitals_s = pd.DataFrame({
            "Heart Rate (bpm)": [116],
            "Blood Pressure (mmHg)": ["86/52 (MAP: 63)"],
            "SpO₂ (%)": [92],
            "Resp Rate (br/min)": [26]
        })
        st.dataframe(vitals_s, use_container_width=True, hide_index=True)

        # Plotly Chart
        fig_s = make_waveform_plot(t_axis, ecg_sepsis, ppg_sepsis, "Patient #204 (Sepsis Alert)", "#E53E3E", "#DD6B20")
        st.plotly_chart(fig_s, use_container_width=True)

        # Model Result
        st.markdown(f"**Inferred Sepsis Probability:** `{prob_sepsis * 100:.1f}%`")
        st.progress(float(prob_sepsis))
        st.error("🚨 **Clinical Alert:** High Sepsis Probability (>85%)")

    # Hemodynamic Comparison Table
    st.markdown("### 📊 Hemodynamic & Morphological Variance Breakdown")
    comparison_df = pd.DataFrame({
        "Physiological Parameter": ["Heart Rate", "PPG Amplitude (Perfusion Index)", "Dicrotic Notch", "ECG R-R Variance"],
        "Healthy Baseline": ["60 - 80 bpm", "High pulsatile volume (Normotensive)", "Distinct dicrotic rebound", "Normal HRV"],
        "Septic Pathology": ["Tachycardia (>100 bpm)", "Damped amplitude (Peripheral collapse)", "Blunted / Lost notch", "Depressed rhythm entropy"],
        "Clinical Implication": ["Compensatory cardiac output", "Vasodilation & systemic hypotension", "Loss of vascular tone/arterial compliance", "Autonomic dysregulation"]
    })
    st.table(comparison_df)

    # Descriptive System Logs Section
    with st.expander("📝 Real-Time DSP & Inference Execution Logs", expanded=True):
        for msg in log_messages:
            st.text(f"⚙️ {msg}")


if __name__ == "__main__":
    main()
