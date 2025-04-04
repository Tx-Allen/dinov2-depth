
# DINOv2 Depth Estimation Environment Setup & Usage Guide

## 📦 Installing Step One by One

```bash
# Step 1: Install Anaconda
bash Anaconda3-2024.10-1-Linux-x86_64.sh

# Step 2: Create conda environment from YAML
conda env create -f dinov2-extras.yaml

# Step 3: Activate the environment
conda activate dinov2-extras

# Step 4: Install necessary packages
pip install -U openmim
mim install mmcv-full
pip install mmsegmentation==0.30.0 xformers==0.0.18
```

### ⚠️ PyTorch CUDA Version Warning

If you get a warning saying that **PyTorch CUDA version is not matching**, do the following:

```bash
# Uninstall current PyTorch packages
pip uninstall torch torchvision torchaudio -y

# Reinstall with the correct CUDA version (11.8)
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 How to Use? (Version 0.2)

1. Download DINOv2 from GitHub and unzip it.
2. Navigate to the folder: `./dinov2` (⚠️ Not `./dinov2/dinov2`)
3. Follow the steps in **Installing Step One by One** above.
4. Activate the conda environment:
   ```bash
   conda activate dinov2-extras
   ```
5. Run the script:
   ```bash
   python run.py --input <picture folder path> --output <output path>
   ```
6. Set the desired **backbone** and **head** in the script or via command line.
7. The `run.py` script will automatically:
   - Load all `.png` images in `<picture folder path>`
   - Process them into depth maps
   - Save the results in `<output path>` with the **same format and filename**

---

Happy predicting! 😄
