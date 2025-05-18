import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

import folder_paths
import comfy
import comfy.model_management as mm

from .src.lbm.inference import get_model_from_config

LBMRelightConfig = {
    "backbone_signature": "stabilityai/stable-diffusion-xl-base-1.0",
    "vae_num_channels": 4,
    "unet_input_channels": 4,
    "timestep_sampling": "custom_timesteps",
    "selected_timesteps": [250, 500, 750, 1000],
    "prob": [0.25, 0.25, 0.25, 0.25],
    "conditioning_images_keys": [],
    "conditioning_masks_keys": [],
    "source_key": "source_image",
    "target_key": "source_image_paste",
    "bridge_noise_sigma": 0.005,
}

CONFIGS = {"lbm_relight": LBMRelightConfig}

class LBMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
            "lbm_model_name": (folder_paths.get_filename_list("diffusion_models"), ),
            "model_config": (list(CONFIGS.keys()), {"default": "lbm_relight"}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    DESCRIPTION = "Load a LBM model from a given path"
    CATEGORY = "lbm"

    def loadmodel(self, lbm_model_name, model_config):
        lbm_model_path = folder_paths.get_full_path_or_raise("diffusion_models", lbm_model_name)
        lbm_state_dict = comfy.utils.load_torch_file(lbm_model_path, safe_load=True)

        lbm_config = CONFIGS[model_config]
        model = get_model_from_config(**lbm_config)
        missing, unexpected = model.load_state_dict(lbm_state_dict, strict=True)
        print(f"Loaded LBM model missing={missing} unexpected={unexpected}")

        offload_device = mm.unet_offload_device()        

        return (model.to(offload_device),)
    
    class LBMRelight:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "input_image": ("IMAGE",),
                    "num_steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                }
            }

        RETURN_TYPES = ("IMAGE",)
        RETURN_NAMES = ("output_image",)
        FUNCTION = "relight"
        DESCRIPTION = "Apply LBM relighting to an input image"
        CATEGORY = "lbm"

        def relight(self, model, input_image, num_steps):
            output_image = evaluate(model, input_image, num_sampling_steps=num_steps)
            return (output_image,)
        
def evaluate(model, image_batch: torch.Tensor, num_sampling_steps=1):
    offload_device = mm.unet_offload_device()        
    device = mm.get_torch_device()

    # image_batch has shape [B, H, W, 3] with values in [0, 1] or [0, 255]
    # We need to run the model on a single image, so we iterate over the batch,
    # and for each [H, W, 3] tensor, convert it to [3, H, W] with values in [-1, 1]
    output_images = []
    for img in image_batch:
        # If input is numpy, convert to torch
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        # Ensure float32
        img = img.float()
        # If values are in [0, 255], scale to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        # Permute to [3, H, W]
        img = img.permute(2, 0, 1)
        # Scale to [-1, 1]
        img = img * 2.0 - 1.0
        img = img.unsqueeze(0) # Add batch dimension

        batch = {
            "source_image": img.to(device).to(torch.bfloat16),
        }

        z_source = model.vae.to(device).encode(batch[model.source_key])

        output_image = model.to(device).sample(
            z=z_source,
            num_steps=num_sampling_steps,
            conditioner_inputs=batch,
            max_samples=1,
        ).clamp(-1, 1)

        model = model.to(offload_device)

        # Remove batch dimension and permute back to [H, W, 3], scale to [0, 1]
        output_image = output_image.squeeze(0).permute(1, 2, 0)
        output_image = (output_image + 1.0) / 2.0
        output_images.append(output_image)

    # Stack outputs into a batch
    return torch.stack(output_images).to(torch.float32).cpu()

NODE_CLASS_MAPPINGS = {
    "LBMLoader": LBMLoader,
    "LBMRelight": LBMLoader.LBMRelight,
}
NODE_DISPLAY_NAMES_MAPPINGS = {
    "LBMLoader": "LBM Loader",
    "LBMRelight": "LBM Relight",
}
