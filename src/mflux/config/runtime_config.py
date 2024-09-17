import mlx.core as mx
import numpy as np

from mflux.config.config import Config, ConfigControlnet
from mflux.config.model_config import ModelConfig


class RuntimeConfig:

    def __init__(self, config: Config | ConfigControlnet, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.sigmas = self._create_sigmas(config, model_config)

        # REVIEW ??
        # TODO setting for this
        if True:
          start = min(range(len(self.sigmas)), key=lambda i: abs(self.sigmas[i] - config.strength))
          self.image_sigmas = mx.concatenate([mx.array([config.strength]), self.sigmas[start + 1:]])
        else:
          start = int((1.0 - config.strength) * config.num_inference_steps)
          self.image_sigmas = self.sigmas[start:]

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def guidance(self) -> float:
        return self.config.guidance

    @property
    def num_inference_steps(self) -> int:
        return self.config.num_inference_steps

    @property
    def strength(self) -> float:
        return self.config.strength

    @property
    def precision(self) -> mx.Dtype:
        return self.config.precision

    @property
    def num_train_steps(self) -> int:
        return self.model_config.num_train_steps
    
    @property
    def controlnet_strength(self) -> float:
        if isinstance(self.config, ConfigControlnet):
            return self.config.controlnet_strength
        else:
            raise NotImplementedError("Controlnet conditioning scale is only available for ConfigControlnet")

    @staticmethod
    def _create_sigmas(config, model) -> mx.array:
        sigmas = RuntimeConfig._create_sigmas_values(config.num_inference_steps)
        if model == ModelConfig.FLUX1_DEV:
            sigmas = RuntimeConfig._shift_sigmas(sigmas, config.width, config.height)
        return sigmas

    @staticmethod
    def _create_sigmas_values(num_inference_steps: int) -> mx.array:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        return mx.concatenate([sigmas, mx.zeros(1)])

    @staticmethod
    def _shift_sigmas(sigmas: mx.array, width: int, height: int) -> mx.array:
        y1 = 0.5
        x1 = 256
        m = (1.15 - y1) / (4096 - x1)
        b = y1 - m * x1
        mu = m * width * height / 256 + b
        mu = mx.array(mu)
        shifted_sigmas = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
        shifted_sigmas[-1] = 0
        return shifted_sigmas
