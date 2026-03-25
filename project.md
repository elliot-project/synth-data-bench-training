# MUST READ Information for GEMINI CLI

Project title: **synthetic generation of WebDatasets to benchmark large scale training frameworks**

We need to compare large scale training frameworks for multimodal models (VLMs). Since this models can handle native resolution and aspect rations, we want to create synthetic datasets in which parameters like number of image and average image resolutions can be controlled.

Specifications:
- Websdataset creation (tar files)
- Megatron-Energon dataloader support
- Randomly generated images and text (e.g. lorem-ipsum)
- Parameters to change the type of data (e.g. avg resolution of images)
- Standarized control of the data being generated (configs via `toml` files)

Packages MUST USE:
- `megatron-energon`
- `pytorch`

Other info:
- create approx 1000 per dataset samples as initial testing
- make one dataset with standard image resolution (336x336)
- make another one with varying image resolution (max 1024x1024, min 224x224)
- simulate visual-intruct data (one image per sample)
