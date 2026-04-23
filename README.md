# README for FYP Submission

## Project Structure
notts-fyp
- `nnUNet/` = main codebase
- `data/` = folders used by nnU-Net via env. variables; i.e. `data/nnUNet_results` is used when extracting `model.zip`
- `training_set` = training images used for training `model.zip`, also used for calibration in `NNCF_inference.py`
- `test_set/` = sample test images from FLARE2024, 10 smallest files
- `ground_truths/` = ground_truths to evaluate the predictions against
- `results/` = inference results (model performance, i.e. dice, iou)
- `model.zip` = trained model weights
- `env.ps1` = PowerShell script to set temporary environment variables
- `inference_flare_task2.py` = baseline inference optimization by Kirchhoff et. al using OpenVINO
- `NNCF_inference.py` = inference optimization using OpenVINO and NNCF quantization
- 

## What is nnU-Net?

## What is my contribution?


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
1. Open PyCharm
2. `New Project >`
3. Click on the project folder "notts-fyp"
4. Create a new virtual environment by running `python -m venv .venv` in the terminal.
5. Enter the environment by running `.venv/Scripts/activate`.
6. Install dependencies by running `pip install -r fyp-dependencies.txt`.
7. Install nnU-Net `cd nnUNet` > `pip install -e .`.
8. Set temporary environment variables by running `./env.ps1`.
4. Install provided `model.zip` by running `nnUNetv2_install_pretrained_model_from_zip model.zip`.
5. Run inference on baseline by entering `python inference_flare_task1.py`.
6. Run inference on optimization-1 by entering

