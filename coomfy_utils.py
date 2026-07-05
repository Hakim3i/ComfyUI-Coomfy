"""Small utility nodes bundled with ComfyUI-Coomfy."""

from __future__ import annotations


class CoomfyIntToFloat:
    """Convert an integer primitive to float for nodes that expect FLOAT inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_value": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 120,
                        "tooltip": "Integer value to convert to float.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert"
    CATEGORY = "Coomfy/utils"

    def convert(self, int_value: int):
        return (float(int_value),)


NODE_CLASS_MAPPINGS = {
    "CoomfyIntToFloat": CoomfyIntToFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoomfyIntToFloat": "Coomfy Int To Float",
}
