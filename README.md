# README for FYP Submission

## Project Structure

- `nnUNet/` = main codebase
- `data/` = folders used by nnU-Net via env. variables; i.e. `data/nnUNet_results` is used when extracting `pretrained_model.zip`
- `training_set` = training images used for training `pretrained_model.zip`, also used for calibration in `NNCF_inference.py`
- `test_set/` = sample test images from FLARE2024, 10 smallest files
- `ground_truths/` = ground_truths to evaluate the predictions against
- `results/` = detailed inference results (model performance, i.e. dice for every class in each case)
- `pretrained_model.zip` = trained model weights
- `env.ps1` = PowerShell script to set temporary environment variables
- `inference_flare_task2.py` = reference inference optimisation by Kirchhoff et al. using OpenVINO compiling
- `NNCF_inference.py` = proposed inference optimisation using OpenVINO and NNCF Post-Training Quantisation

## What is nnU-Net?
nnU-Net is a deep-learning based framework for medical image segmentation that automatically adapts pre-defined 
training pipelines to a given dataset. Given a dataset, nnU-Net extracts specific dataset properties that dictate
how the training pipeline is configured, making the model more targeted to the specific dataset/segmentation task.
Previous approaches to medical image segmentation have been based on hand-crafted training pipelines, requiring high
levels of expertise and experience, due to varying dataset properties.

## What is my contribution?
My proposed optimisation using OpenVINO and NNCF Post-Training Quantisation (PTQ) results in an average of 35%
reduction in inference time, with DSC loss of approximately 1% to the original optimisation.

## Disclaimers
### Modified codebase files
As I only made the optimisation pertaining to the inference of the model,
most of the original codebase of nnU-Net remains untouched. The files listed
below are those that I have modified in the original codebase:

#### Modified files
- `nnUNet/nnunetv2/inference/predict_from_raw_data.py` - added time tracking for model initialisation

### Hardware and Software
These are the hardware and software that I ran this on, and recommend the examiners to run it on for
better reproducibility.
OS = Windows
IDE = PyCharm
Python Version = 3.14
Terminal used by PyCharm = PowerShell

### Setup
All the commands below are to be run in the terminal. Clicking on the Run button in PyCharm will not work.

1. Open PyCharm
2. `New Project >`
3. Click on the project folder "AIMANHAFIDZ-20626459_Software"
4. Create a new virtual environment by running `python -m venv .venv` in the terminal.
5. Enter the environment by running `.venv/Scripts/activate`.
6. Install dependencies by running `pip install -r fyp-dependencies.txt`.
7. Install nnU-Net `cd nnUNet` > `pip install -e .`.
8. Set temporary environment variables by running `./env.ps1`.
9. Install provided `model.zip` by running `nnUNetv2_install_pretrained_model_from_zip model.zip`.

### Running inference
1. Run inference on reference optimisation by running `python inference_flare_task2.py -i test_set -o 
referenceopt_inference_results -m $Env:model_weights -save_model`. 
2. Run the above command once more but omit `-save_model`.
3. Run inference on the proposed optimisation by entering `python NNCF_inference.py -i test_set -o 
NNCFopt_inference_results -m $Env:model_weights --calibration_dir training_images`.

### Evaluating predictions
1. Run `nnUNetv2_evaluate_folder -djfile $Env:model_weights/dataset.json -pfile $Env:model_weights/plans.json
ground_truths referenceopt_inference_results` to evaluate the segmentation accuracy of the reference optimisation.
2. Run `nnUNetv2_evaluate_folder -djfile $Env:model_weights/dataset.json -pfile $Env:model_weights/plans.json
ground_truths referenceopt_inference_results`

