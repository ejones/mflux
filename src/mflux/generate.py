import argparse
import os
import sys
import time

import PIL.Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mflux.config.model_config import ModelConfig
from mflux.config.config import Config
from mflux.flux.flux import Flux1


def main():
    parser = argparse.ArgumentParser(description='Generate an image based on a prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='The textual description of the image to generate.')
    parser.add_argument('--output', type=str, default="image.png", help='The filename for the output image. Default is "image.png".')
    parser.add_argument('--model', "-m", type=str, required=True, choices=["dev", "schnell"], help='The model to use ("schnell" or "dev").')
    parser.add_argument('--seed', type=int, default=None, help='Entropy Seed (Default is time-based random-seed)')
    parser.add_argument('--height', type=int, default=1024, help='Image height (Default is 1024)')
    parser.add_argument('--width', type=int, default=1024, help='Image width (Default is 1024)')
    parser.add_argument('--steps', type=int, default=None, help='Inference Steps')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance Scale (Default is 3.5)')
    parser.add_argument('--quantize',  "-q", type=int, choices=[4, 8], default=None, help='Quantize the model (4 or 8, Default is None)')
    parser.add_argument('--path', type=str, default=None, help='Local path for loading a model from disk')
    parser.add_argument('--lora-paths', type=str, nargs='*', default=None, help='Local safetensors for applying LORA from disk')
    parser.add_argument('--lora-scales', type=float, nargs='*', default=None, help='Scaling factor to adjust the impact of LoRA weights on the model. A value of 1.0 applies the LoRA weights as they are.')
    parser.add_argument('--metadata', action='store_true', help='Export image metadata as a JSON file.')
    parser.add_argument('--image', type=str, help='Path to an input image, for image-to-image')
    parser.add_argument('--strength', type=float, help='Number between 0 and 1 indicating the extent to transform image for image-to-image', default=0.8)

    args = parser.parse_args()

    if args.path and args.model is None:
        parser.error("--model must be specified when using --path")

    if args.steps is None:
        args.steps = 4 if args.model == "schnell" else 14

    # Load the model
    flux = Flux1(
        model_config=ModelConfig.from_alias(args.model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales
    )

    # Generate an image
    image = flux.generate_image(
        seed=int(time.time()) if args.seed is None else args.seed,
        prompt=args.prompt,
        image=args.image and PIL.Image.open(args.image),
        config=Config(
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=args.guidance,
            strength=args.strength,
        )
    )

    # Save the image
    image.save(path=args.output, export_json_metadata=args.metadata)


if __name__ == '__main__':
    main()
