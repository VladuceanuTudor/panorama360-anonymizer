"""nullface_cached.py -- reimplementare a anonymize_face() din repo/anonymize_face.py,
dar cu pipeline-ul SD + IP-Adapter + InsightFace incarcate O SINGURA DATA
(load_pipeline) si reutilizate la fiecare apel (anonymize_face_cached).

Originalul reincarca tot (pipeline + adapter + insightface) la fiecare apel --
~80s doar pentru incarcare, per fata. Inacceptabil pentru procesare batch pe un
folder cu multe fete. Logica de anonimizare propriu-zisa (DDPM inversion forward
+ reverse) e identica cu a lor, doar sectiunea de incarcare e scoasa din functie.
"""
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
from torch import autocast, inference_mode

REPO_DIR = Path(__file__).resolve().parent.parent / "nullface" / "repo"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ddm_inversion.inversion_utils import (  # noqa: E402
    inversion_forward_process,
    inversion_reverse_process,
)
from ddm_inversion.utils import image_grid  # noqa: E402
from prompt_to_prompt.ptp_classes import load_512  # noqa: E402
from utils.face_embedding import FaceEmbeddingExtractor  # noqa: E402


def load_pipeline(sd_model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                   insightface_model_path: str = "~/.insightface",
                   device_num: int = 0,
                   ip_adapter_scale: float = 1.0,
                   det_thresh: float = 0.1,
                   det_size: int = 640):
    device = f"cuda:{device_num}"
    ldm_stable = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model_path, torch_dtype=torch.float16
    ).to(device)
    ldm_stable.load_ip_adapter(
        "h94/IP-Adapter-FaceID",
        subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin",
        image_encoder_folder=None,
    )
    ldm_stable.set_ip_adapter_scale(ip_adapter_scale)

    extractor = FaceEmbeddingExtractor(
        ctx_id=device_num,
        det_thresh=det_thresh,
        det_size=(det_size, det_size),
        model_path=insightface_model_path,
    )
    return ldm_stable, extractor


def anonymize_face_cached(
    ldm_stable, extractor,
    image_path: str,
    mask_image_path: str,
    sd_model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device_num: int = 0,
    guidance_scale: float = 10.0,
    num_diffusion_steps: int = 100,
    eta: float = 1.0,
    skip: int = 70,
    id_emb_scale: float = 1.0,
    output_log_file: str = "log.txt",
    seed: int = 0,
    mask_delay_steps: int = 10,
):
    device = f"cuda:{device_num}"
    dtype = ldm_stable.dtype

    with open(output_log_file, "a") as f:
        try:
            id_embs_inv, id_embs = extractor.get_face_embeddings(
                image_path=image_path,
                is_opposite=True,
                seed=seed,
                scale_factor=id_emb_scale,
                dtype=dtype,
                device=device,
            )
        except ValueError as e:
            f.write(f"{e}\n")
            return None

        ldm_stable.scheduler = DDIMScheduler.from_config(sd_model_path, subfolder="scheduler")
        ldm_stable.scheduler.set_timesteps(num_diffusion_steps)

        offsets = (0, 0, 0, 0)
        x0 = load_512(image_path, *offsets, device).to(dtype=dtype)

        if mask_image_path and Path(mask_image_path).is_file():
            mask_image = load_image(mask_image_path)
        else:
            height, width = x0.shape[-2:]
            mask_image = Image.new("RGB", (width, height), "white")

        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).to(dtype=dtype)

        wt, zs, wts = inversion_forward_process(
            ldm_stable, w0, etas=eta, prompt="", cfg_scale=guidance_scale,
            prog_bar=False, num_inference_steps=num_diffusion_steps,
            ip_adapter_image_embeds=[id_embs_inv],
        )

        generator = torch.manual_seed(seed)

        w0, _ = inversion_reverse_process(
            ldm_stable, xT=wts[num_diffusion_steps - skip], etas=eta, prompts=[""],
            cfg_scales=[guidance_scale], prog_bar=False,
            zs=zs[: (num_diffusion_steps - skip)], controller=None,
            ip_adapter_image_embeds=[id_embs], init_image=x0, mask_image=mask_image,
            generator=generator, mask_delay_steps=mask_delay_steps,
        )

        with autocast("cuda"), inference_mode():
            x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0).sample
        if x0_dec.dim() < 4:
            x0_dec = x0_dec[None, :, :, :]
        return image_grid(x0_dec)
