import streamlit as st
import time
import os
import numpy as np
from PIL import Image
import psutil
from datetime import datetime  # <-- Import datetime for the history ID

# Import layout functions
from utils.layout import render_header, render_footer
from utils.inference import (
    load_model_from_checkpoint, 
    preprocess_lr_images, 
    preprocess_gt_image, 
    run_inference_on_model,
    read_image 
)

psutil.cpu_percent(interval=None) 
time.sleep(0.5)

# --- Page Config ---
st.set_page_config(
    page_title="Process & Run - SwinIR Microscopy",
    page_icon="⚙️",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    body { background-color: #0e1117; color: #f5f6f7; }
    h1, h2, h3, h4 { color: #ffffff; }
    .input-container {
        background-color: #1a1d23; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4); height: 200px;
    }
    .model-select-container {
        background-color: #1a1d23; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "processing_results" not in st.session_state:
    st.session_state.processing_results = None
if "results_history" not in st.session_state:  
    st.session_state.results_history = []       
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
    
options = ["SFSR (Bicubic)", "SFSR (Realistic)", "MFSR (Bicubic)", "MFSR (Realistic)"]
if 'selected_methods' not in st.session_state:
    st.session_state.selected_methods = [] 

# --- Pre-loading Function ---
@st.cache_resource(show_spinner=False) # Caches the result of this function
def preload_models_for_scale(scale):
    """
    Loads all 4 models for a given scale into the cache in the background.
    """
    print(f"--- Pre-loading {scale} models ---")
    scale_int = int(scale.replace('x', ''))
    models_to_load = {
        "SFSR (Bicubic)": f"models/{scale}/sfsr_bicubic.pth",
        "SFSR (Realistic)": f"models/{scale}/sfsr_realistic.pth",
        "MFSR (Bicubic)": f"models/{scale}/mfsr_bicubic.pth",
        "MFSR (Realistic)": f"models/{scale}/mfsr_realistic.pth"
    }
    
    loaded_models = {}
    for method, path in models_to_load.items():
        if os.path.exists(path):
            try:
                load_model_from_checkpoint(path, scale_int, method, hr_patch_size=192)
                loaded_models[method] = "✅ Ready"
            except Exception as e:
                print(f"Failed to pre-load {method}: {e}")
                loaded_models[method] = f"❌ Error"
        else:
            loaded_models[method] = "❌ Not Found"
    
    print(f"--- {scale} model pre-loading complete ---")
    return loaded_models

# --- Header ---
render_header()
# --- Footerr ---
render_footer()

st.markdown("<h2>⚙️ Process & Run Inference</h2>", unsafe_allow_html=True)
st.markdown("---")

# --- MAIN LAYOUT ---

# --- Row 1: Data Input ---
st.subheader("1. Input Data")
col_up_1, col_up_2 = st.columns(2)

with col_up_1:
    with st.container(border=True):
        st.markdown("##### 🖼️ Input LR Image(s)")
        st.file_uploader(
            "Upload one or more .png/.tif images for processing.",
            accept_multiple_files=True,
            key="uploaded_lr_files",
            label_visibility="collapsed"
        )
        
with col_up_2:
    with st.container(border=True):
        st.markdown("##### 🎯 Input HR Image (Ground Truth)")
        st.file_uploader(
            "Upload a single .png/.tif ground truth file (Optional).",
            accept_multiple_files=False,
            key="uploaded_gt_file",
            label_visibility="collapsed"
        )

st.markdown("---")

# --- Row 2: Model Selection ---
st.subheader("2. Select Model")

with st.container(border=True):
    col_model_1, col_model_2 = st.columns(2)
    
    with col_model_1:
        st.markdown("##### Model Scale")
        st.radio(
            "Model",
            ["x2","x4"], 
            key="model_scale",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # --- Call the pre-loading function here ---
        with st.spinner(f"Warming up {st.session_state.model_scale} models..."):
            preload_status = preload_models_for_scale(st.session_state.model_scale)


    with col_model_2:
        st.markdown("##### Method")
        
        def toggle_all_methods():
            if st.session_state.select_all_toggle:
                st.session_state.selected_methods = options
            else:
                st.session_state.selected_methods = []

        st.checkbox(
            "Select All Methods", 
            value=len(st.session_state.selected_methods) == len(options), 
            key="select_all_toggle",
            on_change=toggle_all_methods
        )
        
        st.multiselect(
            "Methods:",
            options,
            key="selected_methods",
            label_visibility="collapsed"
        )
        
        # --- Show the pre-loading status ---
        with st.expander("Model Cache Status"):
            for method, status in preload_status.items():
                st.markdown(f"- `{method}`: {status}")


st.markdown("---")

# --- Row 3: Run Button & Logic ---
if st.button("🚀 Run Super-Resolution Process", use_container_width=True, type="primary"):
    
    # --- 1. Validation Checks ---
    if not st.session_state.selected_methods:
        st.error("Please select at least one model method to run.")
        st.stop()

    if st.session_state.uploaded_lr_files:
        cleaned_lr_files = [f for f in st.session_state.uploaded_lr_files if f is not None]
    else:
        cleaned_lr_files = [] 

    if not cleaned_lr_files: 
        st.error("Please upload at least one LR input image.")
        st.stop()
    
    try:
        if cleaned_lr_files:
            # Use read_image to get a normalized 8-bit NumPy array
            lr_array = read_image(cleaned_lr_files[0])
            # Store it as a PIL Image for consistency
            st.session_state.display_lr_image = Image.fromarray(lr_array)
        else:
            st.session_state.display_lr_image = None
    
        if st.session_state.uploaded_gt_file:
            # Use read_image to get a normalized 8-bit NumPy array
            gt_array = read_image(st.session_state.uploaded_gt_file)
            # Store it as a PIL Image
            st.session_state.display_gt_image = Image.fromarray(gt_array)
        else:
            st.session_state.display_gt_image = None
    except Exception as e:
        st.error(f"Error reading input images: {e}")
        st.stop()
        
    # --- 2. Initialize Log and UI ---
    st.session_state.log_messages = ["Initializing..."]
    st.session_state.processing_results = {} 
    
    st.subheader("Processing...")
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    
    st.markdown("##### Process Log")
    log_container = st.empty()
    log_container.code("\n".join(st.session_state.log_messages), height=300)

    def log(message):
        print(message) 
        st.session_state.log_messages.append(str(message))
        log_container.code("\n".join(st.session_state.log_messages), height=300)

    # --- 3. Start Processing Loop ---
    total_steps = len(st.session_state.selected_methods)
    
    scale = int(st.session_state.model_scale.replace('x', ''))
    hr_patch_size = 192 
    
    num_lr_files = len(cleaned_lr_files) 
    is_mfsr_selected = any("MFSR" in m for m in st.session_state.selected_methods)
    is_sfsr_selected = any("SFSR" in m for m in st.session_state.selected_methods)

    if num_lr_files == 1 and is_mfsr_selected:
        log("WARNING: You uploaded 1 image. MFSR models will run using a *synthetic* burst.")
        st.warning("You uploaded 1 image. MFSR models will run using a *synthetic* burst.")
        
    if num_lr_files > 1 and is_sfsr_selected:
        first_file_name = cleaned_lr_files[0].name 
        log(f"WARNING: You uploaded {num_lr_files} images. SFSR models will *only* process the first image: {first_file_name}")
        st.warning(f"You uploaded {num_lr_files} images. SFSR models will *only* process the first image: {first_file_name}")

    try:
        log("Loading and preprocessing images...")
        gt_image_np = preprocess_gt_image(st.session_state.uploaded_gt_file)
        if gt_image_np is not None:
            log("Ground truth image loaded.")
        else:
            log("No ground truth image loaded. PSNR/SSIM/LPIPS will not be calculated.")

        preprocessed_data = {}
        
        # --- Main Model Loop ---
        for i, method in enumerate(st.session_state.selected_methods):
            status_text.info(f"Processing ({i+1}/{total_steps}): {method} ({scale}x)...")
            log(f"\n--- Processing ({i+1}/{total_steps}): {method} ({scale}x) ---")
            
            model_name_key = method.lower().replace(" ", "_").replace("(", "").replace(")", "")
            model_path = os.path.join("models", st.session_state.model_scale, f"{model_name_key}.pth")
            log(f"Model path: {model_path}")
            
            # --- Load Model (This will now be instant if pre-loading worked) ---
            log("Loading model (from cache)...")
            try:
                # This call is cached, so it will be instant.
                model = load_model_from_checkpoint(model_path, scale, method, hr_patch_size)
                log("Model loaded successfully.")
            except Exception as e:
                log(f"ERROR: Could not load model. {e}")
                st.error(f"Failed to load {method}: {e}")
                continue 

            # --- Pre-process data for this model type ---
            model_class = "SFSR" if "SFSR" in method else "MFSR"
            if model_class not in preprocessed_data:
                log(f"Preprocessing images for {model_class}...")
                lr_tensor, original_dims, padding = preprocess_lr_images(
                    cleaned_lr_files, 
                    model_class
                )
                preprocessed_data[model_class] = (lr_tensor, original_dims, padding)
                log("Image preprocessing complete.")
            else:
                log("Using cached preprocessed images.")
            
            lr_tensor, original_dims, padding = preprocessed_data[model_class]

            # --- Run Inference ---
            log("Running inference...")
            output_image, metrics = run_inference_on_model(
                model=model,
                lr_tensor=lr_tensor,
                gt_image_np=gt_image_np,
                scale=scale,
                original_dims=original_dims,
                padding=padding
            )
            log(f"Inference complete. Metrics: {metrics}")
            
            st.session_state.processing_results[method] = {
                "image": output_image,
                "metrics": metrics,
                "model_path": model_path
            }
            
            progress_bar.progress((i + 1) / total_steps, text=f"Completed: {method} ({scale}x)")

        log("\n--- All processing complete! ---")
        
        # --- HISTORY LOGIC ---
        now = datetime.now()
        result_id = now.strftime("%Y-%m-%d %H:%M:%S") # e.g., "2025-11-02 21:45:30"
        
        # Create a single "result" object to store everything
        current_result_data = {
            "id": result_id,
            "results": st.session_state.processing_results,
            "lr_image": st.session_state.display_lr_image,
            "gt_image": st.session_state.display_gt_image,
            "scale": st.session_state.model_scale
        }
        
        # Add this new result to the top of the history list
        st.session_state.results_history.insert(0, current_result_data)
        
        #  Keep only the last 10 results
        st.session_state.results_history = st.session_state.results_history[:10]

        status_text.success("All processing complete! Navigating to results...")
        time.sleep(1)
        
        st.switch_page("pages/2_Results.py") 

    except Exception as e:
        log(f"\n--- FATAL ERROR --- \n{e}")
        st.error(f"An error occurred. See log for details.")
        st.exception(e)