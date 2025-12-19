import streamlit as st
import time
from utils.layout import render_footer, render_header

# --- Page Config ---
st.set_page_config(
    page_title="SwinIR Microscopy Super-Resolution",
    page_icon="🔬",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
body { background-color: #0e1117; color: #f5f6f7; }
h1, h2, h3, h4 { color: #ffffff; }

.section-header {
    background: linear-gradient(90deg, #1f2937, #111827);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.4);
    margin-bottom: 1.5rem;
}

.quickstart-container {
    background-color: #1a1d23;
    border-radius: 16px;
    padding: 1.8rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    transition: transform 0.2s ease;
}
.quickstart-container:hover { transform: translateY(-3px); }

/* --- NEW --- Style for the links in the Quick Start box */
.quickstart-container a {
    color: #3b82f6; /* Bright blue for links */
    text-decoration: none;
    font-weight: 600;
}
.quickstart-container a:hover {
    text-decoration: underline;
}
/* --- END NEW --- */


.footer-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #1a1d23;
    color: #f5f6f7;
    padding: 0.8rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
    border-top: 1px solid #333;
    transition: margin-left 0.3s ease;
    z-index: 999;
}

@media (max-width: 768px) {
    .footer-container {
        flex-direction: column;
        align-items: center;
        padding: 0.5rem;
        font-size: 0.8rem;
    }
    .footer-stats {
        margin-bottom: 0.5rem;
    }
}
</style>
""", unsafe_allow_html=True)


# --- Header ---
render_header()
st.markdown("---")

# --- Body Content ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="quickstart-container">
    <h2>🚀 Quick Start</h2>
    <ol style="font-size:1rem; line-height:1.7;">
        <li>Go to the <a href="process" target="_self"><b>Process</b></a> page to upload your microscopy image(s).</li>
        <li>Click <b>Run Super-Resolution</b> to start enhancement using SwinIR.</li>
        <li>View and compare your results in the <a href="results" target="_self"><b>Results</b></a> page.</li>
        <li>Check the <a href="help" target="_self"><b>Help</b></a> page for detailed usage and troubleshooting.</li>
        <li>Learn more about this project in the <a href="about" target="_self"><b>About</b></a> section.</li>
    </ol>
</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="quickstart-container">
        <h2>🧠 Model Details</h2>
        <p style="font-size:1rem; line-height:1.7;">
            <b>SwinIR</b> (Image Restoration Transformer) is designed for tasks like 
            super-resolution, denoising, and image restoration.  
            It leverages <b>window-based self-attention</b> for efficient computation while 
            maintaining exceptional visual fidelity.
        </p>
        <ul style="font-size:1rem; line-height:1.6;">
            <li><b>Architecture:</b> Swin Transformer</li>
            <li><b>Task:</b> Microscopy Image Super-Resolution (x2 & x4)</li>
            <li><b>Framework:</b> PyTorch</li>
            <li><b>Model Type:</b> Bicubic & Realistic Image Restoration</li>
            <li><b>Optimization:</b> Trained for fine biological textures</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Live Footer ---
placeholder = st.empty()
while True:
    with placeholder.container():
        render_footer()
    time.sleep(1)
