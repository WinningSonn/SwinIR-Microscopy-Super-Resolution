# SwinIR Multiframe Super-Resolution for Microscopy

An AI-powered tool for enhancing microscopy images using the SwinIR Transformer architecture.

This application implements a Deep Learning system to evaluate and compare **Single-Frame (SFSR) and Multi-Frame (MFSR)** super-resolution methods. It is designed to improve image clarity, reveal fine biological details, and accelerate research workflows.

### 👨‍💻 Project Information

Author: Milson Feliciano (535220093)

Advisor: Agus Budi Dharmawan, S.Kom., MT, M.Sc.

Institution: Fakultas Teknologi Informasi, Jurusan Teknik Informatika

### 📖 Overview

The system allows users to upscale microscopy images using two primary approaches:

Single-Frame Super-Resolution (SFSR): Enhances resolution using a single input image.

Multi-Frame Super-Resolution (MFSR): Enhances resolution by fusing information from a burst of multiple images.

### Key Features

SwinIR Backbone: Uses the Swin Transformer for Image Restoration.

Interactive UI: Built with Streamlit for easy uploading, processing, and visualization.

Quantitative Metrics: Automatically calculates PSNR, SSIM, and LPIPS if a Ground Truth image is provided.

Visual Comparison: Side-by-side comparison tools for analyzing results.

## 📂 Dataset for Training & Data Generation

Note: This section is only required if you plan to use the data generation scripts (generate_lr_datasets.py) or re-train the models. You do not need this for basic inference.

- Download Link (Scale x2): 👉 Kaggle Dataset (x2)

- Download Link (Scale x4): 👉 Kaggle Dataset (x4)

**Setup**: Extract the dataset content into a folder named Dataset/ in the root directory.

## ⚙️ System Requirements

- OS: Windows, macOS, or Linux.

- Python: Version 3.11 - 3.12.

- GPU: NVIDIA GPU with CUDA support is highly recommended.

    - Note: Inference can run on CPU, but it will be significantly slower.

- RAM: 8 GB minimum.

- Storage: 10 GB free space.

## 🚀 Installation & Setup

Follow these steps to set up the application locally for running inference.

### 1. Clone the Repository

Open your terminal and clone the repository:

```bash
git clone [https://github.com/WinningSonn/SwinIR-Microscopy-Super-Resolution.git](https://github.com/WinningSonn/SwinIR-Microscopy-Super-Resolution.git)
cd SwinIR-Microscopy-Super-Resolution
```

### 2. Set Up Virtual Environment (venv)

It is highly recommended to use a virtual environment to avoid library conflicts.

**For Windows:**

```bash
# Create the environment
python -m venv venv

# Activate the environment
venv\Scripts\activate
```


**For macOS / Linux:**

```bash
# Create the environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```


Once activated, your terminal prompt should show (venv).

### 3. Install Dependencies

Install the required Python libraries using pip:

`pip install -r requirements.txt`

### 4. Download Model Weights (Required for Inference)

Due to GitHub file size limits, the pre-trained .pth model weights are hosted externally.

**Download the models from Kaggle:** 👉 [SwinIR Microscopy Models](https://www.kaggle.com/models/milsonfeliciano/swinir-microscopy-models)

Extract the files and place them inside the models/ directory. Your folder structure must look exactly like this:

```
SwinIR-Microscopy-Super-Resolution/
├── models/
│   ├── x2/
│   │   ├── sfsr_bicubic.pth
│   │   ├── sfsr_realistic.pth
│   │   ├── mfsr_bicubic.pth
│   │   └── mfsr_realistic.pth
│   └── x4/
│       ├── sfsr_bicubic.pth
│       ├── ... (corresponding x4 models)
├── pages/
├── utils/
├── app.py
├── requirements.txt
└── README.md
```

## 💻 How to Run

1. Ensure your virtual environment is active ((venv) is visible).

2. Run the Streamlit application:

`streamlit run app.py`


The application will automatically open in your default web browser at http://localhost:8501.

## 📖 User Guide

1. Home Page

Provides a quick start guide and details about the SwinIR architecture and model parameters.

2. Process Page (Run Inference)

This is the main functional page where you perform Super-Resolution.

- Input Data:

    - LR Image(s): Upload Low-Resolution images.

        - For SFSR, upload 1 image.

        - For MFSR, upload a burst (multiple images).

    - Ground Truth (Optional): Upload a High-Resolution image. If provided, the system will calculate PSNR, SSIM, and LPIPS metrics.

- Select Model:

    - Scale: Choose x2 or x4.

    - Method: Select specific models (e.g., MFSR (Realistic) or SFSR (Bicubic)). You can select multiple methods to run them in sequence.

- Run: Click the Run Super-Resolution Process button to start inference.

### 3. Results Page

After processing, the app redirects here.

- History: Select past results from the dropdown menu.
- Grid View: View all generated output images side-by-side with their performance metrics.
- Comparison: Use the "Side-by-Side Comparison" tab to inspect fine details (e.g., comparing "Original LR" vs "MFSR Output").
- Download: Click "Download All as .zip" to save images and a CSV report of the metrics.

## ℹ️ Acknowledgments

This project is based on the research thesis: "Peningkatan Resolusi Citra Mikroskopis Melalui Multiframe Super Resolution Berbasis Model SwinIR Image Restoration."

It utilizes the open-source SwinIR repository:

Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). SwinIR: Image restoration using swin transformer.
