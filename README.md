# TileMorph

[![Replicate](https://replicate.com/andreasjansson/tile-morph/badge)](https://replicate.com/andreasjansson/tile-morph)

TileMorph creates a tileable animation between two Stable Diffusion prompts. It uses [the circular padding trick](https://gitlab.com/-/snippets/2395088) to generate images that wrap around the edges.

The animation effect is achieved by interpolating both in CLIP embedding space and latent space.
* The number of CLIP interpolation steps is controlled by the `num_animation_frames` input. Each "animation frame" runs a full Stable Diffusion inference, which makes it slow but interesting.
* The number of latent space interpolation steps between animation frames is controlled by the `num_interpolation_steps` input. Each interpolation step only runs a VAE inference, and is fast but less interesting. You can trade off interestingness versus prediction time by tweaking `num_animation_frames` and `num_interpolation_steps`
* `num_animation_frames * num_interpolation_steps` = number of output frames
* `num_animation_frames * num_interpolation_steps / frames_per_second` = output video length in seconds

This model supports seamless transitions between different generations. Set `prompt_end` and `seed_end` to the same value of video number _n_ as `prompt_start` and `seed_start` of video number _n + 1_.

## Development

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

    cog run script/download-weights <your-hugging-face-auth-token>

Then, you can run predictions:

    cog predict -i prompt_start="colorful abstract patterns" -i seed_start=1 -i prompt_end="tropical jungle, cgsociety" -i seed_end=2

Or, build a Docker image:

    cog build

Or, [push it to Replicate](https://replicate.com/docs/guides/push-a-model):

    cog push r8.im/...
