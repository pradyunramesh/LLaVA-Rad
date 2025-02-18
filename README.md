# 🏥 LLaVA-Rad
[![Code License](https://img.shields.io/badge/Code%20License-Microsoft%20Research-red)](LICENSE)
[![Preprint](https://img.shields.io/badge/arXiv-Preprint-blue)](ttps://arxiv.org/abs/2403.08002)

## Introduction

Official implementation of LLaVa-Rad, introduced in ["Towards a clinically accessible radiology multimodal model: open-access and lightweight, with automatic evaluation"](https://arxiv.org/abs/2403.08002).

LLaVA-Rad can take in as input a frontal chest X-ray and optionally a reason for exam and will output the corresponding findings.

**Note:** if you are interested in radiologist aligned evaluation of generated reports, we recommend you use the [CheXprompt](https://github.com/microsoft/chexprompt) codebase.

## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Train](#train)
  - [0. Preparation](#0-preparation)
  - [1. Pretrain (Alignment)](#1-pretrain-alignment)
  - [2. Fine-tuning (LoRA)](#2-fine-tuning-lora)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License and Usage Notices](#license-and-usage-notices)
- [Acknowledgements](#acknowledgements)


## Requirements

We trained and tested LLaVA-Rad using Python 3.10. For optimal inference, we recommend using a GPU environment. LLaVA-Rad has been tested on NVIDIA V100 and A100 GPUs with CUDA 11.x (or newer) drivers, on recent versions of Ubuntu.

## Installation

Follow these steps to set up LLaVA-Rad:

1. Clone the repository and navigate to the project folder:
   ```Shell
   git clone https://github.com/microsoft/LLaVA-Rad.git
   cd LLaVA-Rad
   ```
2. Create and activate a virtual environment (Python 3.10):
   ```Shell
   conda create -n llavarad python=3.10 -y
   conda activate llavarad
   ```
3. Upgrade pip and install the package:
   ```Shell
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   ```
4. [Optional] Install additional dependencies for training:
   ```bash
   pip install ninja
   pip install flash-attn --no-build-isolation
   ```

## Train

When starting from scratch, the following checkpoints are needed:
- A pre-trained LM checkpoint, e.g., [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- By default, we use a customized domain-specific ViT, BiomedCLIP-CXR. See [README.md](./llava/model/multimodal_encoder/open_clip_encoder/README.md) for details.

### 0. Preparation
Before running the commands below, you need to have the data, image folder, and the above checkpoints ready. 

**0.1 Data**

To download the data, sign the data use agreement and follow the instructions for download at [LLaVA-Rad MIMIC-CXR Annotations on PhysioNet](https://physionet.org/content/llava-rad-mimic-cxr-annotation/1.0.0/). This will include reports with extracted sections in LLaVA format, split into train/dev/test.

**0.2 Images**

You need to download the [MIMIC-CXR-JPG images from PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) by signing the data use agreement and following the instructions.

**0.3 Model weights**

You can find the pretrained model weights for BiomedCLIP-CXR and LLaVA-Rad at https://huggingface.co/microsoft/llava-rad.


**Notes before proceeding:** 
- Change the paths in the scripts below according to where you downloaded the data.
- Batch size is set for 4-GPU machines. If your machine has a difference number of GPUs, please change batch size. Training commands have been tested on a single 80GB A100 and 4x80GB H100, using torch 2.4.1 and cuda 11.8 with flash attention 2.7.2.post1.

### 1. Pretrain (Alignment)
At this stage, we only train the projection layer (which aligns the vision features with text features). The vision encoder and LLM are all frozen.

```bash
bash scripts/pretrain.sh
```

We get a pretrained projector `mm_projector.bin` after pretraining.

### 2. Fine-tuning (LoRA)
Once we have a pretrained projector, we can do fine-tuning. The command below fine-tunes the projector and LoRA of LLM:
```bash
bash scripts/finetune_lora.sh
```

## Inference

Before running the command below, you need to change the script accordingly.

```bash
bash scripts/eval.sh
```

**Note:** To reproduce the evaluation results from the manuscript on the MIMIC-CXR dataset, changing the script means uncommenting and updating the paths for `query_file` and `image_folder`.

In the manuscript, the Open-I and CheXpert chest X-ray images and reports are also used for evaluation. These datasets are available at their corresponding sources: [Open-I](https://openi.nlm.nih.gov/faq) | [CheXpert](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1).

## Evaluation

If you have run inference using multiple GPUs and have a resulting set of chunks with results, make sure you concatenate prediction chunks into a single file before running the following command:
```bash
cd llava/eval/rr_eval
python run.py ${YOUR_PREDICTION_FILE}
```

## Citation

```bibtex

@article{zambranochaves2024llavarad,
  title = {Towards a clinically accessible radiology foundation model: open-access and lightweight, with automated evaluation},
  author = {Zambrano Chaves, JM and Huang, S-C and Xu, Y and Xu, H and Usuyama, N and Zhang, S, et al.},
  journal = {arXiv preprint arXiv:2403.08002},
  year = {2024},
  url = {https://arxiv.org/pdf/2403.08002}
}

```

## License and Usage Notices

The data, code, and model checkpoints are licensed and intended for research use only. The code and model checkpoints are subject to additional restrictions as determined by the Terms of Use of LLaMA, Vicuna, and GPT-4 respectively. Code and model checkpoints may be used for research purposes and should not be used in direct clinical care or for any clinical decision making purpose.

## Acknowledgements

Our codebase heavily relies on [LLaVA](https://github.com/haotian-liu/LLaVA) v1.5. Please check out their repo for more information, and consider citing them in addition to our manuscript if you use this codebase.

```bibtex

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

```
