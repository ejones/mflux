import logging

import mlx.core as mx

log = logging.getLogger(__name__)


class Config:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
            self,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
            strength: float = 0.8,
    ):
        if width % 16 != 0 or height % 16 != 0:
            log.warning("Width and height should be multiples of 16. Rounding down.")
        if not (0 <= strength <= 1):
            log.warning("Strength should be between 0.0 and 1.0. Clipping.")
        self.width = 16 * (width // 16)
        self.height = 16 * (height // 16)
        self.num_inference_steps = num_inference_steps
        self.guidance = guidance
        self.strength = min(max(0.0, strength), 1.0)


class ConfigControlnet(Config):
    def __init__(
            self,
            num_inference_steps: int = 4,
            width: int = 1024,
            height: int = 1024,
            guidance: float = 4.0,
            controlnet_strength: float = 1.0,
    ):
        super().__init__(num_inference_steps, width, height, guidance)
        self.controlnet_strength = controlnet_strength
