import streamlit as st
import datetime
from utils.monitor import get_system_stats

def render_footer():
    stats = get_system_stats()
    now = datetime.datetime.now()
    formatted_date = now.strftime("%A, %d %B %Y")
    formatted_time = now.strftime("%H:%M:%S")

    # Extract numeric values for progress bars
    cpu_val = float(stats['CPU Usage'].replace("%", ""))
    ram_val = float(stats['RAM Usage'].replace("%", ""))
    disk_val = float(stats['Disk Usage'].replace("%", ""))
    gpu_val = 0.0 if stats['GPU Load'] == "N/A" else float(stats['GPU Load'].replace("%", ""))

    # Inject CSS once for responsive footer
    st.markdown("""
    <style>
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
    .system-grid {
        display: grid;
        grid-template-columns: repeat(2, auto);
        grid-gap: 0.5rem 2rem;
    }
    /* When sidebar is open, shift footer to align with main content */
    [data-testid="stSidebar"][aria-expanded="true"] ~ .main .footer-container {
        margin-left: 250px; /* default sidebar width */
    }
    </style>
    """, unsafe_allow_html=True)

    # Build footer with system stats + progress bars + date/time
    footer_html = f"""
    <div class="footer-container">
        <div class="system-grid" style="align-items:center;">
            <div>
                🧠 CPU: {stats['CPU Usage']}
                <div style="background:#333; border-radius:4px; height:6px; width:120px;">
                    <div style="background:#ef4444; width:{cpu_val}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            <div>
                💾 RAM: {stats['RAM Usage']}
                <div style="background:#333; border-radius:4px; height:6px; width:120px;">
                    <div style="background:#3b82f6; width:{ram_val}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            <div>
                💽 Disk: {stats['Disk Usage']}
                <div style="background:#333; border-radius:4px; height:6px; width:120px;">
                    <div style="background:#f59e0b; width:{disk_val}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            <div>
                🎮 GPU: {stats['GPU Load']}
                <div style="background:#333; border-radius:4px; height:6px; width:120px;">
                    <div style="background:#10b981; width:{gpu_val}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
        </div>
        <div style="text-align:right;">
            📅 {formatted_date}<br/>
            ⏰ {formatted_time}
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="section-header">
        <h1>🔬 SwinIR Multiframe Super-Resolution</h1>
        <p style="font-size:1.1rem; color:#d1d5db;">
            An AI-powered tool for enhancing microscopy images using the SwinIR Transformer architecture.  
            Improve image clarity, reveal fine details, and accelerate your research workflow.
        </p>
    </div>
    """, unsafe_allow_html=True)
