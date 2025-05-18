import folder_paths
import comfy.model_management
from comfy.model_management import ModelManager as mm

from .src.lbm.inference import get_model_from_config, evaluate

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

        # return model patcher
        offload_device = mm.unet_offload_device()        

        return (model.to(offload_device),)
    
NODE_CLASS_MAPPINGS = {
    "LBMLoader": LBMLoader,
}
NODE_DISPLAY_NAMES_MAPPINGS = {
    "LBMLoader": "LBM Loader",
}
