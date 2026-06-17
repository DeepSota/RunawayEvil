<div align="center">
    <h1>
    Runaway Evil: Jailbreaking Image-to-Video Generative Models
    </h1>
    <div>
        <h4 align="center">
            <a href="https://xzxg001.github.io/Runaway-Evil" target='_blank'>
                <img src="https://img.shields.io/badge/🌟-Project%20Page-blue" style="padding-right: 20px;">
            </a>
            <br>
        </h4>
    </div>
</div>

This repository contains the official implementation for the paper: **"Runaway Evil: Jailbreaking Image-to-Video Generative Models"**. We introduce a novel, evolutionary framework that uses Reinforcement Learning (RL) to automatically discover and execute complex, multi-step adversarial attacks against state-of-the-art Image-to-Video (I2V) models. Our system trains an autonomous agent to strategically manipulate images and generate video prompts that bypass safety alignments, effectively creating "jailbreaks."

## Framework Overview

The framework is built around a Deep Reinforcement Learning agent (using Proximal Policy Optimization - PPO) that interacts with a custom `gymnasium` environment (`VideoJailbreakEnv`).

  * **Reinforcement Learning (RL) Agent**: The agent's goal is to learn an optimal policy for selecting the most effective "tactic" to generate unsafe video content.
  * **Environment**: The environment simulates a complete red-teaming attempt. It encompasses prompt generation, image editing, video generation, and safety judging.
  * **State**: The current image being manipulated, represented as a NumPy array.
  * **Action**: The selection of an abstract "attack tactic" from an evolving list (e.g., "Use role-playing misdirection to justify unsafe elements").
  * **Reward**: The agent is rewarded for successfully generating content deemed "unsafe" by a Judge model. It is penalized for failing, taking too many steps, making significant visual edits (measured by LPIPS), and using suspicious keywords in prompts.
  * **Evolutionary Tactic Generation**: This is the framework's key innovation. A meta-learning loop analyzes successful attacks from the agent's memory bank and uses a powerful Large Vision-Language Model (LVLM) to synthesize new, more general attack tactics. These new tactics are then added to the agent's action space, allowing it to discover increasingly complex and effective strategies over time.

## System Architecture

The attack pipeline consists of several interconnected AI models, orchestrated by the RL agent:

1.  **Strategist Agent (PPO)**: The RL agent that selects a high-level tactic for the attack at each step.
2.  **Prompt Generator (LLaVA-Next)**: Given the agent's chosen tactic and the current image, this LVLM generates two coordinated prompts:
      * `P_edit`: An instruction for the image editor.
      * `P_video`: A prompt for the video generator.
3.  **Image Editor (FLUX-Kontext)**: Modifies the input image based on the `P_edit` prompt, creating an adversarial intermediate asset.
4.  **Video Generator (Wan2.2)**: Creates a video based on the edited image and the `P_video` prompt.
5.  **Judge (Qwen-VL)**: An independent LVLM that evaluates the final generated video and classifies it as "safe" or "unsafe," providing the primary reward signal for the RL agent.
6.  **Tactic Analyzer (Qwen-VL Large)**: In Stage 1, this powerful LVLM analyzes the memory of successful attacks to propose new, generalized tactics, enabling the evolutionary aspect of the framework.

## File Structure

  * `stage1.py`: Implements the first phase of training. The RL agent learns initial strategies while the **Tactic Analyzer** simultaneously observes successes and evolves the list of available tactics. This expands the agent's action space over time.
  * `stage2.py`: Implements the second phase of training (fine-tuning). It loads the evolved tactics from Stage 1 and continues to train the RL agent on this fixed, expanded set of tactics to refine its policy.
  * `inference.py`: A standalone script to run the trained PPO agent for evaluation. It loads a trained model and attacks a series of test images, logging the results.
  * `run_inference.py`: A helper script for the original `Wan2.2-Turbo` model. **Note: This file is not used if you are loading `Wan2.2-5B` directly from the Hugging Face Hub, as in the final version of `inference.py`.**

## Prerequisites

### Hardware

  * A multi-GPU setup is **highly recommended**. The code is configured to distribute models across several GPUs.
  * High VRAM (e.g., 24GB+ per GPU) is necessary to run the suite of models.
  * NVIDIA GPUs with CUDA support.

### Software

  * Python 3.9+
  * Conda for environment management.
  * PyTorch with CUDA support.

## Setup & Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/Runaway-Evil.git
    cd Runaway-Evil
    ```

2.  **Create a Conda Environment:**

    ```bash
    conda create -n runawayevil python=3.10
    conda activate runawayevil
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file is recommended. Based on the imports, you will need:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate bitsandbytes sentence-transformers
    pip install stable-baselines3[extra] gymnasium
    pip install diffusers lpips pandas opencv-python omegaconf
    ```

4.  **Download Models:**
    Download the required models and place them in a designated models directory. You will need:

      * LLaVA-Next (e.g., `llava-v1.6-mistral-7b-hf`)
      * Qwen-VL (e.g., `Qwen/Qwen2.5-7B-VL-Instruct` and `Qwen/Qwen2.5-32B-VL-Instruct`)
      * FLUX-Kontext (`black-forest-labs/FLUX.1-s-kontext`)
      * CLIP (`openai/clip-vit-base-patch32`)
      * (Optional, for `run_inference.py`) The local weights for Wan2.2-Turbo.

5.  **Download Dataset:**
    The code is configured to use the COCO dataset.

      * Download the COCO images (e.g., `val2017`).
      * Prepare corresponding `.txt` annotation files, where each file has the same name as its image and contains one or more caption strings, each on a new line.

## Configuration

Before running any script, you **must** configure the paths at the top of each file. Open `stage1.py`, `stage2.py`, and `inference.py` and fill in the empty string variables (`""`) with the absolute paths to your models, datasets, and output directories.

**Example Configuration in `stage1.py`:**

```python
# In stage1.py, stage2.py, and inference.py
# (Ensure you fill these out in all relevant files)

# --- Model Paths ---
LLAVA_PATH = "/path/to/your/models/llava-v1.6-mistral-7b-hf"
QWEN_7B_VL_PATH = "/path/to/your/models/Qwen2.5-7B-VL-Instruct"
QWEN_32B_VL_PATH = "/path/to/your/models/Qwen2.5-32B-VL-Instruct" # Only for stage1
FLUX_MODEL_PATH = "/path/to/your/models/FLUX.1-s-kontext"

# --- Dataset Paths ---
COCO_IMAGE_DIR = "/path/to/your/datasets/coco/val2017"
COCO_ANNOTATION_DIR = "/path/to/your/datasets/coco/annotations_txt"

# --- Output & Log Paths ---
# IMPORTANT: In stage1.py, change the hardcoded LOG_DIR to a placeholder
LOG_DIR = "/path/to/your/outputs/phase1_logs/" # For stage1
PHASE2_LOG_DIR = "/path/to/your/outputs/phase2_logs/" # For stage2 and inference

# --- (Optional) Wan2.2-Turbo Local Path ---
WAN22_PROJECT_DIR = "/path/to/your/Wan2.2-TI2V-5B-Turbo-main" # For stage1/stage2
```

## Usage

### Stage 1: Evolutionary Training

This stage trains the agent from scratch and evolves the attack tactics.

```bash
python stage1.py
```

  * The script will create a log directory (e.g., `/path/to/your/outputs/phase1_logs/`).
  * Inside, it will save the PPO model (`ppo_strategist_evolving.zip`), the evolving tactics (`evolving_tactics.json`), and a memory of successes.
  * The training will proceed in batches, analyzing successes and potentially adding new tactics between batches.

### Stage 2: Fine-Tuning

This stage takes the evolved tactics from Stage 1 and fine-tunes the agent's policy.

1.  Ensure the `PHASE2_TACTICS_FILE` in `stage2.py` points to the `evolving_tactics.json` generated by Stage 1.
2.  Run the script:
    ```bash
    python stage2.py
    ```

<!-- end list -->

  * This will load or create a fine-tuned PPO model and save checkpoints and logs in the specified `PHASE2_LOG_DIR`.

### Inference: Evaluating the Trained Agent

This script uses the final, fine-tuned agent from Stage 2 to attack a set of test images.

1.  Configure `PPO_MODEL_PATH` in `inference.py` to point to the model saved in Stage 2 (e.g., `ppo_strategist_finetuned.zip`).
2.  Configure all other necessary model and data paths.
3.  Run the script:
    ```bash
    python inference.py
    ```

<!-- end list -->

  * The script will iterate through images in `TEST_IMAGE_DIR`, perform the multi-step attack, and save a detailed log of the results to `OUTPUT_LOG_FILE`.

## Ethical Considerations

This framework is intended for research purposes, specifically for evaluating and improving the safety and robustness of generative AI models (i.e., "red-teaming"). The unauthorized or malicious use of this code to generate harmful, unsafe, or inappropriate content is strictly prohibited. Users are expected to adhere to the terms of service of any models used and to employ this tool responsibly and ethically.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{yourname2025runawayevil,
      title={Runaway Evil: Jailbreaking Image-to-Video Generative Models}, 
      author={Your Name and Other Co-authors},
      year={2025},
      eprint={arXiv:xxxx.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```