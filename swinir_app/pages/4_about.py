import streamlit as st
import time
from utils.layout import render_footer, render_header

# Page Config 
st.set_page_config(
    page_title="About - SwinIR Microscopy Super-Resolution",
    page_icon="ℹ️",
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
.about-container {
    background-color: #1a1d23;
    border-radius: 16px;
    padding: 1.8rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    font-size: 1rem;
    line-height: 1.7;
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

# About Content
st.markdown("""
<div class="about-container">
    <h2>ℹ️ About This Project</h2>
    <p>
        This application was developed as part of a research thesis:
    </p>
    <blockquote style="font-style:italic; color:#d1d5db;">
        "Peningkatan Resolusi Citra Mikroskopis Melalui Multiframe Super Resolution 
        Berbasis Model SwinIR Image Restoration (SwinIR)"
    </blockquote>
    <p>
        <b>Researcher:</b> Milson Feliciano (535220063)<br/>
        <b>Advisor:</b> Agus Budi Dharmawan, S.Kom, M.T.
    </p>
    <p>
        The system evaluates <b>Single-Frame Super-Resolution (SFSR)</b> and 
        <b>Multi-Frame Super-Resolution (MFSR)</b> methods using the powerful 
        <b>SwinIR Transformer architecture</b>, specifically adapted for microscopic imagery.  
        The goal is to enhance image clarity, reveal fine biological details, 
        and accelerate research workflows in microscopy.
    </p>
</div>
""", unsafe_allow_html=True)

# Live Footer
render_footer()
