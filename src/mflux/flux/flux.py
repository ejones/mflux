import mlx.core as mx
from mlx import nn
import PIL.Image
from tqdm import tqdm

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5
from mflux.tokenizer.tokenizer_handler import TokenizerHandler
from mflux.weights.model_saver import ModelSaver
from mflux.weights.weight_handler import WeightHandler


class Flux1:

    def __init__(
            self,
            model_config: ModelConfig,
            quantize: int | None = None,
            local_path: str | None = None,
            lora_paths: list[str] | None = None,
            lora_scales: list[float] | None = None,
    ):
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales
        self.model_config = model_config

        # Load and initialize the tokenizers from disk, huggingface cache, or download from huggingface
        tokenizers = TokenizerHandler(model_config.model_name, self.model_config.max_sequence_length, local_path)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        self.vae = VAE()
        self.transformer = Transformer(model_config)
        self.t5_text_encoder = T5Encoder()
        self.clip_text_encoder = CLIPEncoder()

        # Load the weights from disk, huggingface cache, or download from huggingface
        weights = WeightHandler(
            repo_id=model_config.model_name,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales
        )

        # Set the loaded weights if they are not quantized
        if weights.quantization_level is None:
            self._set_model_weights(weights)

        # Optionally quantize the model here at initialization (also required if about to load quantized weights)
        self.bits = None
        if quantize is not None or weights.quantization_level is not None:
            self.bits = weights.quantization_level if weights.quantization_level is not None else quantize
            nn.quantize(self.vae, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            nn.quantize(self.transformer, class_predicate=lambda _, m: isinstance(m, nn.Linear) and len(m.weight[1]) > 64, group_size=64, bits=self.bits)
            nn.quantize(self.t5_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)
            nn.quantize(self.clip_text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear), group_size=64, bits=self.bits)

        # If loading previously saved quantized weights, the weights must be set after modules have been quantized
        if weights.quantization_level is not None:
            self._set_model_weights(weights)

    def generate_image(
            self,
            seed: int,
            prompt: str,
            config: Config = Config(),
            image: PIL.Image.Image | None = None,
            mask: PIL.Image.Image | None = None,
    ) -> GeneratedImage:
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)
        sigmas = config.image_sigmas if image else config.sigmas
        sigmas_iter = tqdm(sigmas[:-1])

        # 1. Create the initial latents
        init_noise = mx.random.normal(
            shape=[1, (config.height // 16) * (config.width // 16), 64],
            key=mx.random.key(seed)
        )
        latents, image_latents = self._get_image_latents(image, init_noise, config) if image else (init_noise, init_noise)
        latent_mask = self._get_latent_mask(mask, config) if mask else None

        # 2. Embedd the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        for i, sigma in enumerate(sigmas_iter):
            # 3.t Predict the noise
            noise = self.transformer.predict(
                sigma=sigma,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                config=config,
            )

            # 4.t Take one denoise step
            dt = sigmas[i + 1] - sigma
            latents += noise * dt

            if mask:
                img_latents_step = Flux1._scale_noise(image_latents, sigmas[i + 1], init_noise)
                latents = (1 - latent_mask) * img_latents_step + latent_mask * latents

            # Evaluate to enable progress tracking
            mx.eval(latents)

        # 5. Decode the latent array and return the image
        latents = Flux1._unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=sigmas_iter.format_dict['elapsed'],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            config=config,
        )

    @staticmethod
    def _pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, 16, height // 16, 2, width // 16, 2))
        latents = mx.transpose(latents, (0, 2, 4, 1, 3, 5))
        latents = mx.reshape(latents, (1, (height // 16) * (width // 16), 64))
        return latents

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, height // 16, width // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, height // 16 * 2, width // 16 * 2))
        return latents

    @staticmethod
    def _scale_noise(latents: mx.array, sigma: mx.array, noise: mx.array):
        return sigma * noise + (1.0 - sigma) * latents

    @staticmethod
    def _get_latent_mask(mask: PIL.Image.Image, config: Config) -> mx.array:
        mask = mask.resize((config.width // 16 * 2, config.height // 16 * 2))
        mask = mask.convert('L')
        mask_array = ImageUtil.to_array_binary(mask)
        mask_array = mx.repeat(mask_array, 16, 1)
        return Flux1._pack_latents(mask_array, config.height, config.width)

    @staticmethod
    def from_alias(alias: str, quantize: int | None = None) -> "Flux1":
        return Flux1(
            model_config=ModelConfig.from_alias(alias),
            quantize=quantize,
        )

    def _set_model_weights(self, weights):
        self.vae.update(weights.vae)
        self.transformer.update(weights.transformer)
        self.t5_text_encoder.update(weights.t5_encoder)
        self.clip_text_encoder.update(weights.clip_encoder)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)

    def _get_image_latents(
            self,
            image: PIL.Image.Image,
            noise: mx.array,
            config: Config,
    ) -> mx.array:
        image_array = ImageUtil.to_array(image)
        encoded = self.vae.encode(image_array)
        image_latents = Flux1._pack_latents(encoded, config.height, config.width)
        latents = Flux1._scale_noise(image_latents, config.image_sigmas[0], noise)
        return latents, image_latents
