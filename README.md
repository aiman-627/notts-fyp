# README for FYP Submission

## Project Structure
notts-fyp
- nnUNet/ = main codebase
- model.zip = trained model weights
- test_set = sample test dataset from FLARE2024, 10 smallest files
- data/ = placeholder folders, used during setting temp env variables
- env.ps1 = powershell script to set temporary variables
- inference_flare_task2.py = baseline optimization by Kirchhoff et. al

## What is nnU-Net?

## Disclaimers
### Modified Files
As I only made optimizations that pertain to the inference of the model,
most of the original codebase of nnU-Net remains untouched. The files listed
below are those that I have added/modified to make the optimizations are listed below:

1. NNCF_inference.py

### Hardware and Software
These are the hardware and software that I ran this on, and recommend the examiners to run it on for
better reproducibility.
OS = Windows
IDE = PyCharm
Python version = 3.14
Terminal used by PyCharm = PowerShell

### Setup
1. Open your preferred IDE (I used PyCharm for this project)
2. Click on New Project
3. Click on the project folder "notts-fyp"
4. Create a new virtual environment "python -m venv .venv"
5. Enter the environment ".venv/Scripts/activate"
6. Install dependencies (openvino, nncf, onnx) + torch first (they specify to always install torch first) 
"pip install torch torchvision"
7. Install nnU-Net "cd nnUNet" "pip install -e ."
8. Run powershell script to set temporary environment variables "./env.ps1"
4. Install provided `model.zip` by entering "nnUNetv2_install_pretrained_model_from_zip model.zip"
5. Run inference on baseline by entering python inference_flare_task1.py
6. Run inference on optimization-1 by entering
7. Run inference on optimization-2 by entering

