import sys
import os
from typing import Optional, List, Iterator

import cv2
import av
import numpy as np
import torch
from torch import autocast
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
)
from PIL import Image
from cog import BasePredictor, Input, Path

MODEL_CACHE = "diffusers-cache"


def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        for k, v in patch.items():
            kwargs[k] = v
        return init(self, *args, **kwargs)

    cls.__init__ = __init__


patch_conv(padding_mode="circular")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt_start: str = Input(description="Prompt to start the animation with"),
        prompt_end: str = Input(
            description="Prompt to end the animation with. You can include multiple prompts by separating the prompts with | (the 'pipe' character)"
        ),
        width: int = Input(
            description="Width of output video",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output video",
            choices=[128, 256, 512, 768],
            default=512,
        ),
        num_interpolation_steps: int = Input(
            description="Number of steps to interpolate between animation frames",
            ge=0,
            le=1000,
            default=20,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=5000, default=50
        ),
        num_animation_frames: int = Input(
            description="Number of frames to animate", default=10, ge=2, le=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        frames_per_second: int = Input(
            description="Frames per second in output video",
            default=20,
            ge=1,
            le=60,
        ),
        intermediate_output: bool = Input(
            description="Whether to display intermediate outputs during generation",
            default=False,
        ),
        seed_start: int = Input(
            description="Random seed for first prompt. Leave blank to randomize the seed",
            default=None,
        ),
        seed_end: int = Input(
            description="Random seed for last prompt. Leave blank to randomize the seed",
            default=None,
        ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        with torch.autocast("cuda"), torch.inference_mode():
            if seed_start is None:
                seed_start = int.from_bytes(os.urandom(2), "big")
            if seed_end is None:
                seed_end = int.from_bytes(os.urandom(2), "big")
            print(f"Using seeds: {seed_start}, {seed_end}")
            generator_start = torch.Generator("cuda").manual_seed(seed_start)
            generator_end = torch.Generator("cuda").manual_seed(seed_end)

            batch_size = 1

            # Generate initial latents to start to generate animation frames from
            initial_scheduler = self.pipe.scheduler = make_scheduler(
                num_inference_steps
            )
            noise_latents_start = torch.randn(
                (batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
                generator=generator_start,
                device="cuda",
            )
            noise_latents_end = torch.randn(
                (batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
                generator=generator_end,
                device="cuda",
            )
            do_classifier_free_guidance = guidance_scale > 1.0

            print("Generating first and last keyframes")
            # re-initialize scheduler
            self.pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)

            prompts = [prompt_start] + [
                p.strip() for p in prompt_end.strip().split("|")
            ]
            keyframe_text_embeddings = []

            for prompt in prompts:
                keyframe_text_embeddings.append(
                    self.pipe._encode_prompt(
                        prompt, "cuda", 1, do_classifier_free_guidance, ""
                    )
                )

            # re-initialize scheduler
            self.pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
            latents_start = self.denoise(
                latents=noise_latents_start,
                text_embeddings=keyframe_text_embeddings[0],
                t_start=0,
                t_end=None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator_start,
            )

            image_start = self.pipe.decode_latents(latents_start)
            self.pipe.run_safety_checker(
                image_start, "cuda", keyframe_text_embeddings[0].dtype
            )

            # re-initialize scheduler
            self.pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)
            latents_end = self.denoise(
                latents=noise_latents_end,
                text_embeddings=keyframe_text_embeddings[-1],
                t_start=0,
                t_end=None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator_end,
            )
            image_end = self.pipe.decode_latents(latents_end)
            self.pipe.run_safety_checker(
                image_end, "cuda", keyframe_text_embeddings[0].dtype
            )

            if intermediate_output:
                yield save_pil_image(
                    self.pipe.numpy_to_pil(image_start)[0], path="/tmp/output-0.png"
                )

            # Generate animation frames
            frames_latents = []
            for keyframe in range(len(prompts) - 1):
                for i in range(num_animation_frames):
                    if keyframe == 0 and i == 0:
                        latents = latents_start
                    else:
                        print(f"Generating frame {i} of keyframe {keyframe}")
                        text_embeddings = slerp(
                            i / num_animation_frames,
                            keyframe_text_embeddings[keyframe],
                            keyframe_text_embeddings[keyframe + 1],
                        )

                        # re-initialize scheduler
                        self.pipe.scheduler = make_scheduler(
                            num_inference_steps, initial_scheduler
                        )
                        noise_latents = slerp(
                            i / num_animation_frames,
                            noise_latents_start,
                            noise_latents_end,
                        )
                        import time

                        t = time.time()
                        latents = self.denoise(
                            latents=noise_latents,
                            text_embeddings=text_embeddings,
                            t_start=0,
                            t_end=None,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator_start,
                        )
                        print(
                            f"denoise {time.time() - t=}"
                        )  # TODO(andreas): remove debug

                    # de-noise this frame
                    frames_latents.append(latents)
                    if intermediate_output and i > 0:
                        image = self.pipe.decode_latents(latents)
                        yield save_pil_image(
                            self.pipe.numpy_to_pil(image)[0],
                            path=f"/tmp/output-{i}.png",
                        )
            frames_latents.append(latents_end)

            # for i in frames_latents: yield i
            # return

            images = self.interpolate_latents(frames_latents, num_interpolation_steps)
            # for i in images:
            #    yield i
            # return

            # images = [
            #    self.pipe.decode_latents(lat)[0].astype("float32")
            #    for lat in frames_latents
            # ]
            yield self.save_mp4(images, frames_per_second, width, height)

    def interpolate_latents(self, frames_latents, num_interpolation_steps):
        print("Interpolating images from latents")
        images = []
        with torch.inference_mode():
            for i in range(len(frames_latents) - 1):
                latents_start = frames_latents[i]
                latents_end = frames_latents[i + 1]
                for j in range(num_interpolation_steps):
                    x = j / num_interpolation_steps
                    latents = latents_start * (1 - x) + latents_end * x
                    image = self.pipe.decode_latents(latents.to(torch.float16))[
                        0
                    ].astype("float32")
                    images.append(image)
        return images

    def save_mp4(self, images, fps, width, height):
        print("Saving MP4")
        output_path = "/tmp/output.mp4"

        output = av.open(output_path, "w")
        stream = output.add_stream(
            "h264",
            rate=fps,
            options={
                "crf": "10",
                "tune": "film",
            },
        )
        # stream.bit_rate = 8000000
        # stream.bit_rate = 16000000
        stream.width = width
        stream.height = height

        for i, image in enumerate(images):
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            packet = stream.encode(frame)
            output.mux(packet)

        # flush
        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        return Path(output_path)

    def denoise(
        self,
        latents,
        text_embeddings,
        t_start,
        t_end,
        num_inference_steps,
        guidance_scale,
        generator,
    ):
        eta = 0
        timesteps = self.pipe.scheduler.timesteps
        do_classifier_free_guidance = guidance_scale > 1.0

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        with self.pipe.progress_bar(
            total=num_inference_steps
        ) as progress_bar, torch.inference_mode(), torch.no_grad():
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.pipe.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipe.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

        return latents


def make_scheduler(num_inference_steps, from_scheduler=None):
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
    scheduler.set_timesteps(num_inference_steps, device="cuda")
    if from_scheduler:
        scheduler.cur_model_output = from_scheduler.cur_model_output
        scheduler.counter = from_scheduler.counter
        scheduler.cur_sample = from_scheduler.cur_sample
        scheduler.ets = from_scheduler.ets[:]
    return scheduler


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def save_pil_image(image, path):
    image.save(path)
    return Path(path)
