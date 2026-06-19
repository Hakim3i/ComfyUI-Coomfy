"""Load latents uploaded to ComfyUI input/ by the Coomfy Video Lab refine flow."""

from __future__ import annotations

import folder_paths


def _input_latent_names() -> list[str]:
    input_dir = folder_paths.get_input_directory()
    files = folder_paths.filter_files_extensions(
        folder_paths.recursive_search(input_dir, ["safetensors"]),
        [".safetensors"],
    )
    return sorted(files)


class CoomfyLoadUploadedLatent:
    """Load a pass-1 latent that Coomfy uploaded to ``input/`` before queueing refine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": (
                    _input_latent_names(),
                    {
                        "tooltip": (
                            "Latent .safetensors in ComfyUI input/ "
                            "(uploaded by Coomfy Video Lab before refine)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "Coomfy/LTX"

    def load(self, latent: str):
        import safetensors.torch

        latent_path = folder_paths.get_full_path("input", latent)
        tensor_data = safetensors.torch.load_file(latent_path, device="cpu")
        multiplier = 1.0
        if "latent_format_version_0" not in tensor_data:
            multiplier = 1.0 / 0.18215
        samples = {"samples": tensor_data["tensor"].float() * multiplier}
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "CoomfyLoadUploadedLatent": CoomfyLoadUploadedLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoomfyLoadUploadedLatent": "Coomfy Load Uploaded Latent",
}
