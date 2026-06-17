import torch
import os
import sys
project_root = "" # add Wan2.2-TI2V-5B-Turbo-main root path here
sys.path.insert(0, project_root)

from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms.functional as TF

from pipeline import Wan22FewstepInferencePipeline
from diffusers.utils import export_to_video

def load_wan22_pipeline(config_path, checkpoint_folder, base_model_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    config = OmegaConf.load(config_path)
    
    correct_key_path = "model_name" 
    OmegaConf.update(config, correct_key_path, base_model_path, merge=False)

    pipe = Wan22FewstepInferencePipeline(config) 
    
    if checkpoint_folder is not None:
        state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        m, u = pipe.generator.load_state_dict(new_state_dict, strict=False)
        assert len(u) == 0, f"Unexpected key found in state_dict: {u}"
    
    pipe = pipe.to(device="cuda", dtype=torch.bfloat16)
    
    return pipe

def run_wan22_inference(
    pipe,
    output_path,
    prompt,
    image_path=None,
    seed=42,
    h=704,
    w=1280,
    num_frames=121
):

    with torch.no_grad():
        assert num_frames % 4 == 1, "num_frames must be a multiple of 4 plus 1"

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        wan22_image_latent = None
        if image_path is not None:
            img = Image.open(image_path).convert("RGB").resize((w, h), Image.LANCZOS)
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
            wan22_image_latent = pipe.vae.encode_to_latent(img.unsqueeze(0))

        video = (
            pipe.inference(
                noise=torch.randn(1, (num_frames - 1) // 4 + 1, 48, h // 16, w // 16, generator=torch.Generator(device="cuda").manual_seed(seed), dtype=torch.bfloat16, device="cuda"),
                text_prompts=[prompt],
                wan22_image_latent=wan22_image_latent,
            )[0]
            .permute(0, 2, 3, 1).cpu().numpy()
        )

        export_to_video(video, output_path, fps=24)
    
if __name__ == "__main__":
    
    my_pipe = load_wan22_pipeline(
        config_path=f"{project_root}/configs/inference/wan22.yaml",
        base_model_path=f"{project_root}/wan5b",
        checkpoint_folder=f"{project_root}/wan5b-turbo"
    )

    my_prompt = "" # add your prompt here
    my_image = "input_image.jpg"
    my_output_path = "output_video.mp4"
    my_seed = 42

    run_wan22_inference(
        pipe=my_pipe,
        output_path=my_output_path,
        prompt=my_prompt,
        image_path=my_image,
        seed=my_seed,
    )