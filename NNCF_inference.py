import argparse
import multiprocessing
import random
from os.path import join
from pathlib import Path
from time import time
from typing import Union, Tuple, List

import nncf
import numpy as np
import openvino as ov
import openvino.properties.hint as hints
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import nnunetv2
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

torch.set_num_threads(multiprocessing.cpu_count())


class NNCFOpenVINOPredictor(nnUNetPredictor):
    """
    Extends nnUNetPredictor with:
      1. NNCF Post-Training Quantisation (PTQ) on the trained model network
      2. Export of the quantised model to OpenVINO IR
      3. Inference via OpenVINO compiled model on CPU

    Also disables Test-Time Augmentation, similar to the reference optimisation by Kirchhoff et al., to reduce latency
    """

    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = False,  # use_mirroring is flag to enable/disable Test-Time Augmentation
                 perform_everything_on_device: bool = False,
                 device: torch.device = torch.device('cpu'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 calibration_dir: str = None,
                 num_calibration_samples: int = 16):
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )
        self.calibration_dir = calibration_dir
        self.num_calibration_samples = num_calibration_samples
        self.ov_network = None  # compiled OpenVINO model

    # Calibration dataset is built from training images; used for Post-Training Quantisation (PTQ)
    def _build_calibration_dataset(self, num_input_channels, patch_size):

        if self.calibration_dir is None:
            print("No calibration_dir provided — using random noise for PTQ calibration.")
            return [torch.randn(1, num_input_channels, patch_size[0], patch_size[1])
                    for _ in range(self.num_calibration_samples)]

        files = sorted(Path(self.calibration_dir).glob("*.nii.gz"))
        if not files:
            raise FileNotFoundError(f"No .nii.gz files found in: {self.calibration_dir}")

        print(f"Building calibration dataset from {len(files)} images in {self.calibration_dir}")
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        patches = []

        for img_path in files:
            data, _seg, _props = preprocessor.run_case(
                [str(img_path)],
                None,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json,
            )
            data = torch.from_numpy(data.astype(np.float32))

            samples_per_image = max(1, self.num_calibration_samples // len(files))

            ph, pw = patch_size
            c, z, h, w = data.shape
            for _ in range(samples_per_image):
                zi = random.randint(0, max(0, z - 1))
                hi = random.randint(0, max(0, h - ph))
                wi = random.randint(0, max(0, w - pw))
                patch = data[:, zi, hi:hi + ph, wi:wi + pw]
                if patch.shape[1:] != torch.Size([ph, pw]):
                    patch, _ = pad_nd_image(patch, [ph, pw], 'constant', {'value': 0}, True, None)
                patches.append(patch.unsqueeze(0))

            if len(patches) >= self.num_calibration_samples:
                break

        patches = patches[:self.num_calibration_samples]
        print(f"Collected {len(patches)} calibration patches.")
        return patches

    # Model initialisation; most of this is reused code from the reference optimisation
    def initialise_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(
                join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                map_location=torch.device('cpu'),
                weights_only=False,
            )
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', None)

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            'nnunetv2.training.nnUNetTrainer',
        )
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name}')

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )
        self.network = network
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        network.load_state_dict(parameters[0])

        # patch size is the same patch size determined during dataset extraction (before training)
        patch_size = configuration_manager.patch_size

        calibration_data = self._build_calibration_dataset(num_input_channels, patch_size)
        calibration_dataset = nncf.Dataset(calibration_data, lambda x: x)

        # PTQ is applied onto the model
        print("Applying NNCF Post-Training Quantisation...")
        quantized_network = nncf.quantize(network, calibration_dataset)
        quantized_network.eval()
        print("NNCF quantization complete.")

        # Convert to OpenVINO intermediate representation, to compile the model using OpenVINO backend
        example_input = torch.randn(1, num_input_channels, *patch_size)
        print("Converting quantized model to OpenVINO IR...")
        ov_model = ov.convert_model(quantized_network, example_input=example_input)

        core = ov.Core()
        core.set_property(
            "CPU", {hints.execution_mode: hints.ExecutionMode.PERFORMANCE}
        )
        config = {
            hints.performance_mode: hints.PerformanceMode.LATENCY,
            hints.enable_cpu_pinning(): True,
        }
        self.ov_network = core.compile_model(ov_model, "CPU", config=config)
        self.network = None  # PyTorch model no longer needed
        print("OpenVINO model compiled and ready.")

    # Function to return the result of a forward pass of the model, i.e. make a prediction on a patch of an image
    # Overridden to not include test-time augmentation, to reduce inference latency at the cost of reduced
    # segmentation accuracy
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        result = self.ov_network(x.numpy())[0]  # numpy in, numpy out
        return torch.from_numpy(result)

    # Function to iterate on patches of an image and make predictions
    # Overridden to not use original thread logic to feed image patches, due to it not being runnable on
    # OpenVINO backend
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = torch.device('cpu')

        try:
            data = data.to(results_device)

            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            gaussian = compute_gaussian(
                tuple(self.configuration_manager.patch_size),
                sigma_scale=1. / 8,
                value_scaling_factor=10,
                device=results_device,
            ) if self.use_gaussian else 1

            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None].float()  # (1, C, ...) — OV needs float32
                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction.half()
                n_predictions[sl[1:]] += gaussian

            torch.div(predicted_logits, n_predictions, out=predicted_logits)

            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array.')

        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            raise e

        return predicted_logits

    # Function to get "window" positions on patches of the image
    # Overridden to not use "self.network", since model is run on OpenVINO backend now instead of PyTorch
    # Also does not use CUDA autocast (improves performance of network operations on GPUs) anymore, which is only on GPUs
    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        assert isinstance(input_image, torch.Tensor)

        with dummy_context():
            assert input_image.ndim == 4, 'input_image must be 4D (c, x, y, z)'

            data, slicer_revert_padding = pad_nd_image(
                input_image, self.configuration_manager.patch_size,
                'constant', {'value': 0}, True, None,
            )
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    # Original function was to reload PyTorch model weights and average predictions across folds
    # Overridden to only use one fold, since I only trained one fold. Function effectively just calls
    # predict_sliding_window_return_logits() with no additional logic
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        prediction = self.predict_sliding_window_return_logits(data)
        if self.verbose:
            print('Prediction done')
        return prediction

# Program entry point, mostly just reused code
def predict_nncf_openvino(input_dir, output_dir, model_folder,
                          calibration_dir, num_calibration_samples):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_files = list(input_dir.glob("*.nii.gz"))
    output_files = [str(output_dir / f.name[:-12]) for f in input_files]

    model_start = time()
    # Initialise the predictor object
    predictor = NNCFOpenVINOPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        device=torch.device('cpu'),
        calibration_dir=calibration_dir,
        num_calibration_samples=num_calibration_samples,
    )
    predictor.initialise_from_trained_model_folder(model_folder, ("0",)) # I hardcoded this, since it's the only fold I trained
    rw = predictor.plans_manager.image_reader_writer_class()

    print(f"Model initialization time: {time() - model_start:.2f}s")

    for input_file, output_file in zip(input_files, output_files):
        print(f"Predicting {input_file.name}")
        start = time()
        image, props = rw.read_images([str(input_file)])
        predictor.predict_single_npy_array(image, props, None, output_file, False)
        print(f"Prediction time: {time() - start:.2f}s")

# Used argparser so running the program is more straight-forward in the CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="test_set")
    parser.add_argument("-o", "--output", default="NNCFopt_inference_results")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--calibration_dir", default=None,
                        help="Folder of .nii.gz images for PTQ calibration. "
                             "Defaults to random noise if not provided.")
    parser.add_argument("--calibration_samples", type=int, default=16)
    args = parser.parse_args()

    predict_nncf_openvino(args.input, args.output, args.model,
                          args.calibration_dir, args.calibration_samples)