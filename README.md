# README for FYP Submission

## Project Structure

- `data/` = folders used by nnU-Net via env. variables; i.e. `data/nnUNet_results` is used when extracting `pretrained_model.zip`
- `ground_truths/` = ground_truths to evaluate the predictions against
- `nnUNet/` = main codebase
- `results/` = detailed sample inference results on System 1 GPU and CPU (model performance, i.e. dice for every class in each case)
- `test_set/` = sample test images from FLARE2024, 10 smallest files
- `training_set` = training images used for training `pretrained_model.zip`, also used for PTQ calibration in `NNCF_inference.py`
- `env.ps1` = PowerShell script to set temporary environment variables
- `fyp-dependencies.txt` = list of dependencies required to run the code
- `inference_flare_task2.py` = reference inference optimisation by Kirchhoff et al. using OpenVINO compiling
- `NNCF_inference.py` = proposed inference optimisation using OpenVINO and NNCF Post-Training Quantisation (PTQ)
- `pretrained_model.zip` = trained model weights

## What is nnU-Net?
nnU-Net is a deep learning-based framework for medical image segmentation that automatically adapts pre-defined 
training pipelines to a given dataset. Given a dataset, nnU-Net extracts specific dataset properties that dictate
how the training pipeline is configured, making the model more targeted to the specific dataset/segmentation task.
Previous approaches to medical image segmentation have been based on hand-crafted training pipelines, requiring high
levels of expertise and experience, due to varying dataset properties.
nnU-Net's codebase is open-source and available from here: [nnU-Net GitHub Repository](https://github.com/MIC-DKFZ/nnUNet)

## What is my contribution?
My proposed optimisation using OpenVINO and NNCF Post-Training Quantisation (PTQ) results in up to an average of 38%
reduction in inference time, with an average DSC loss of 0.3% relative to the original reference optimisation.

## Disclaimers
### Modified codebase files
As I only made the optimisation pertaining to the inference of the model,
most of the original codebase of nnU-Net remains untouched. The files listed
below are those that I have modified in the original codebase:

#### Modified files
- `nnUNet/nnunetv2/inference/predict_from_raw_data.py` - added time tracking for model initialisation
- `inference_flare_task2.py` - added time tracking for model initialisation

### Hardware and Software
These are the hardware and software that I ran this on, and recommend the examiners to run it on for
better reproducibility.

OS = Windows
IDE = PyCharm
Python Version = 3.14
Terminal used by PyCharm = PowerShell (important)

### Setup
All the commands below are to be run in the terminal. Clicking on the Run button in PyCharm will not work.

1. Open PyCharm
2. Click on `Open...`
3. Navigate to the directory where you have the folder "AimanHafidz20626459_Software"
4. Click on the folder "AimanHafidz20626459_Software"
5. Open the terminal and ensure the terminal is PowerShell. If it is, it will say PS before the working directory.
6. Create a new virtual environment by running `python -m venv .venv` in the terminal.
7. Enter the virtual environment by running `.venv/Scripts/activate`.
8. Install dependencies by running `pip install -r fyp-dependencies.txt`.
9. Install nnU-Net `cd nnUNet` > `pip install -e .`.
10. Exit the nnU-Net directory by running `cd ..`. All following commands are to be run in the root working directory.
11. Set temporary environment variables by running `./env.ps1`.
12. Install provided `model.zip` by running `nnUNetv2_install_pretrained_model_from_zip pretrained_model.zip`.

### Running inference on the reference optimisation and evaluating the results
1. Run inference on reference optimisation by running `python inference_flare_task2.py -i test_set -o 
reference_inference_results -m $Env:model_weights -save_model`. 
2. Run the above command again but omit `-save_model`.
3. Run `nnUNetv2_evaluate_folder -djfile $Env:model_weights/dataset.json -pfile $Env:model_weights/plans.json
ground_truths reference_inference_results` to evaluate the segmentation accuracy of the reference optimisation.
4. The results of the evaluation can be found in `reference_inference_results/summary.json`. 

### Running inference on the proposed optimisation and evaluating the results
1. Run inference on the proposed optimisation by entering `python NNCF_inference.py -i test_set -o 
proposed_inference_results -m $Env:model_weights --calibration_dir training_images`.
2. Run `nnUNetv2_evaluate_folder -djfile $Env:model_weights/dataset.json -pfile $Env:model_weights/plans.json
ground_truths proposed_inference_results` to evaluate the segmentation accuracy of the proposed optimisation.
3. The results of the evaluation can be found in `proposed_inference_results/summary.json`. 

