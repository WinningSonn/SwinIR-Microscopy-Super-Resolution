import streamlit as st
import time
import pandas as pd
from PIL import Image, ImageOps
import io
import zipfile
import numpy as np

# Import custom layout functions
from utils.layout import render_header, render_footer

# --- Import the baseline function from utils file ---
from utils.inference import calculate_baseline_metrics

# Page Config
st.set_page_config(
    page_title="Results - SwinIR Microscopy",
    page_icon="📊",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    body { background-color: #0e1117; color: #f5f6f7; }
    h1, h2, h3, h4 { color: #ffffff; }
    .stDataFrame { border-radius: 12px; }
    
    /* Style for the tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 24px; /* Space between tabs */
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1d23; /* Tab background */
        border-radius: 8px 8px 0 0;
        padding: 10px 15px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6; /* Active tab color */
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def convert_image_to_bytes(img_array_or_pil):
    """Converts a NumPy array or PIL Image to bytes for downloading."""
    if isinstance(img_array_or_pil, np.ndarray):
        # Convert NumPy array to PIL Image
        img = Image.fromarray(img_array_or_pil.astype(np.uint8))
    else:
        img = img_array_or_pil 
        
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def create_zip_file(results, lr_image, gt_image, scale_str): # Added scale_str
    """Creates a zip file in memory with all images and a metrics CSV."""
    zip_buffer = io.BytesIO()
    metrics_list = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        
        # 1. Add all SR (Super-Resolution) images
        for method_name, data in results.items():
            file_name = f"{method_name.replace(' ', '_')}.png"
            zip_file.writestr(file_name, convert_image_to_bytes(data['image']))
            
            # Add metrics to a list for the CSV
            metrics = data['metrics'].copy()
            metrics['Model'] = method_name
            metrics_list.append(metrics)

        # 2. Add Original LR and GT images 
        if lr_image:
            zip_file.writestr("Original_LR_Input.png", convert_image_to_bytes(lr_image))
        if gt_image:
            zip_file.writestr("Original_GT_Input.png", convert_image_to_bytes(gt_image))
        
        # 3. Add metrics.csv
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            df = df.set_index('Model') 
            zip_file.writestr("results.csv", df.to_csv())

    return zip_buffer.getvalue()

def get_image_by_name(name, results, lr_image, gt_image):
    """Helper for the comparison tool to get the image from a string name."""
    if name == "Original LR" and lr_image:
        return lr_image 
    if name == "Original GT" and gt_image:
        return gt_image 
    if name in results:
        # Convert numpy array from results to PIL Image
        return Image.fromarray(results[name]['image'])
    return None 

# Header
render_header()
st.markdown("<h2>📊 Inference Results</h2>", unsafe_allow_html=True)
st.markdown("---")

# Check for Results History
if not st.session_state.get("results_history"): # Check if history list exists and is not empty
    st.error("No processing results found. Please run a new process from the 'Process' page.")
    if st.button("Go to Process Page"):
        st.switch_page("pages/1_Process.py")
    st.stop()

# Create a list of IDs for the dropdown
result_ids = [res["id"] for res in st.session_state.results_history]

# Create the dropdown
selected_id = st.selectbox(
    "Select a past result to view:",
    result_ids
)

# Find the full result data based on the selected ID
selected_result = next(res for res in st.session_state.results_history if res["id"] == selected_id)

# Load Data from the *Selected* Result
results = selected_result["results"]
lr_image = selected_result["lr_image"]
gt_image = selected_result["gt_image"]
scale = selected_result["scale"]

st.success(f"✅ Displaying results for: **{selected_id}** ({len(results)} model(s) at **{scale}** scale)")


# "Download All" Button 
try:
    zip_bytes = create_zip_file(results, lr_image, gt_image, scale) # Pass scale to zip function
    st.download_button(
        label="📥 Download All as .zip (Images + Metrics CSV)",
        data=zip_bytes,
        file_name=f"swinir_results_{scale}.zip",
        mime="application/zip",
        use_container_width=True
    )
except Exception as e:
    st.error(f"Could not create zip file: {e}")

st.markdown("---")

# Display Input Images
st.subheader("Input Image(s)")
cols_input = st.columns(2)
with cols_input[0]:
    if lr_image:
        st.image(lr_image, caption=f"Original LR Input (1st Frame) | Size: {lr_image.width}x{lr_image.height}", use_container_width='auto')
        
        if gt_image:
            with st.spinner("Calculating baseline bicubic metrics..."):
                baseline_metrics = calculate_baseline_metrics(lr_image, gt_image)
            
            st.markdown("##### Baseline Metrics (Bicubic Upscale vs. GT)")
            df_baseline = pd.DataFrame.from_dict(baseline_metrics, orient='index', columns=['Value'])
            df_baseline['Value'] = df_baseline['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, (float, int)) else x)
            st.dataframe(df_baseline, use_container_width=True)

with cols_input[1]:
    if gt_image:
        st.image(gt_image, caption=f"Original GT Input | Size: {gt_image.width}x{gt_image.height}", use_container_width='auto')
    else:
        st.info("No Ground Truth image was provided for comparison.")

st.markdown("---")

# Main Results Display 
tab_grid, tab_compare = st.tabs(["📊 Results Grid", "↔️ Side-by-Side Comparison"])

# Results Grid (All Models)
with tab_grid:
    st.subheader(f"Generated {scale} Super-Resolution Outputs")
    
    # Create a 2-column grid layout
    cols_grid = st.columns(2)
    
    for i, (method, data) in enumerate(results.items()):
        # Fill grid left-to-right, top-to-bottom
        with cols_grid[i % 2]: 
            with st.container(border=True):
                st.subheader(method)
                
                output_img = data['image'] 
                st.image(output_img, caption=f"Enhanced Image | Size: {output_img.shape[1]}x{output_img.shape[0]}", use_container_width='auto')

                # Conditional Metrics Display
                st.markdown("##### Performance Metrics")
                
                # Create a new dict for metrics 
                metrics_to_show = {}

                #Formatting logic for time
                def format_time(time_val):
                    if isinstance(time_val, (float, int)):
                        return f"{time_val:.2f}s"
                    return time_val 
                
                if not gt_image:
                    # No GT: Only show Time and Size
                    metrics_to_show["Inference Time (s)"] = format_time(data['metrics'].get("Inference Time (s)"))
                    metrics_to_show["Output Size"] = f"{output_img.shape[1]}x{output_img.shape[0]}"
                else:
                    # GT is present: Show all metrics
                    metrics_to_show = data['metrics'].copy() # Copy all metrics
                    # Overwrite the time with the formatted version
                    metrics_to_show["Inference Time (s)"] = format_time(metrics_to_show.get("Inference Time (s)"))
                    metrics_to_show["Output Size"] = f"{output_img.shape[1]}x{output_img.shape[0]}"

                # Display as a clean dataframe
                df = pd.DataFrame.from_dict(metrics_to_show, orient='index', columns=['Value'])
                
                def format_value(val):
                    if isinstance(val, (float, int)):
                        return f"{val:.6f}" 
                    return str(val) 

                df['Value'] = df['Value'].apply(format_value)
                
                st.dataframe(df, use_container_width=True)

                # Individual Download Button
                st.download_button(
                    label=f"Download {method} Image",
                    data=convert_image_to_bytes(output_img),
                    file_name=f"{method.replace(' ', '_')}_{scale}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"download_{i}" 
                )

# Side-by-Side Comparison
with tab_compare:
    st.subheader("Compare Model Outputs")

    # Build the list of options for the dropdowns
    options = []
    if lr_image:
        options.append("Original LR")
    if gt_image:
        options.append("Original GT")
    options.extend(list(results.keys())) # Add all model names

    if len(options) < 2:
        st.warning("Not enough images to compare (e.g., only 1 image was processed).")
    else:
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            left_choice = st.selectbox("Compare Left:", options, index=0)
        with col_select2:
            default_index = 1 
            if "MFSR (Realistic)" in options:
                default_index = options.index("MFSR (Realistic)")
            elif len(results.keys()) > 0:
                default_index = options.index(list(results.keys())[0])
            
            right_choice = st.selectbox("Compare Right:", options, index=default_index)

        img_left_pil = get_image_by_name(left_choice, results, lr_image, gt_image)
        img_right_pil = get_image_by_name(right_choice, results, lr_image, gt_image)

        if img_left_pil is not None and img_right_pil is not None:
            
            is_left_lr = (left_choice == "Original LR")
            is_right_lr = (right_choice == "Original LR")
            
            if is_left_lr and not is_right_lr:
                target_size = img_right_pil.size
                img_left_pil = img_left_pil.resize(target_size, Image.Resampling.NEAREST)
                
            elif is_right_lr and not is_left_lr:
                target_size = img_left_pil.size # (width, height)
                img_right_pil = img_right_pil.resize(target_size, Image.Resampling.NEAREST)
            
            # Convert to RGB 
            img_left_final = img_left_pil.convert("RGB")
            img_right_final = img_right_pil.convert("RGB")

            st.markdown("---")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader(left_choice)
                st.image(img_left_final, use_container_width=True)

            with col_right:
                st.subheader(right_choice)
                st.image(img_right_final, use_container_width=True)
            
        else:
            st.error("Could not load one of the selected images for comparison.")


# Live Footer
placeholder = st.empty()
while True:
    with placeholder.container():
        render_footer()
    time.sleep(1)