import streamlit as st
import time
from utils.layout import render_footer, render_header

# Page Config
st.set_page_config(
    page_title="Help - SwinIR Microscopy Super-Resolution",
    page_icon="❓",
    layout="wide",
)

# Custom CSS
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
.help-container {
    background-color: #1a1d23;
    border-radius: 16px;
    padding: 1.8rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    font-size: 1rem;
    line-height: 1.7;
}
.help-container ol {
    margin-left: 1.2rem;
}
.footer-container {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #1a1d23;
    color: #f5f6f7;
    padding: 0.8rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
    border-top: 1px solid #333;
}
.system-grid {
    display: grid;
    grid-template-columns: repeat(2, auto);
    grid-gap: 0.5rem 2rem;
}
</style>
""", unsafe_allow_html=True)

# Header
render_header()

# Help Content
st.markdown("""
<div class="help-container">
    <h2>❓ How to Use the Application</h2>
    <ol>
        <li><b>Select Mode:</b> Choose which model(s) to run. 
            Currently, the app loads all four (SFSR Bicubic, SFSR Realistic, MFSR Bicubic, MFSR Realistic). 
            In future updates, you’ll be able to select just one, two, or all models before inference.</li>
        <li><b>Upload Data:</b> 
            <ul>
                <li>Use <i>Browse Folder</i> to select a directory of low‑resolution burst images.</li>
                <li>Or use <i>Browse File</i> to upload a single LR image. The app will generate a synthetic burst automatically.</li>
                <li>Optionally, upload a high‑resolution ground truth image for quantitative evaluation.</li>
            </ul>
        </li>
        <li><b>Run Process:</b> Click the <i>Run Super‑Resolution Process</i> button. 
            The app will pad inputs, build tensors, and run inference on the selected models.</li>
        <li><b>View Results:</b> The app will display:
            <ul>
                <li>The input image(s)</li>
                <li>Outputs from each selected model</li>
                <li>Ground truth (if provided)</li>
                <li>Performance metrics (PSNR, SSIM, LPIPS, runtime)</li>
            </ul>
        </li>
    </ol>
</div>
""", unsafe_allow_html=True)


# Live Footer
render_footer()
