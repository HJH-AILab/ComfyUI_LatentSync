import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/LatentSync")

from latentsync.utils import face_detector
from latentsync.utils.face_detector import FaceDetector, INSIGHTFACE_DETECT_SIZE,cuda_to_int
face_detector.insight_face = None

# Override the FaceDetector class to use insightface from custom node input
class FaceDetectorOverride(FaceDetector):
    def __init__(self, device="cuda",):
        if face_detector.insight_face is None:
            super().__init__(device=device)
        else:
            self.app = face_detector.insight_face
            self.app.prepare(ctx_id=cuda_to_int(device),det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

face_detector.FaceDetector = FaceDetectorOverride
# End of override

import folder_paths
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
dtype = torch.float16 if is_fp16_supported else torch.float32

CONFIG_DIR = os.path.join(ROOT_DIR, "LatentSync","configs")

CONFIG = OmegaConf.load(os.path.join(CONFIG_DIR, "unet", "stage2.yaml"))

GLOBAL_CATEGORY = "HJH_LatentSync🪅"

class LipsyncPipelineNode:
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                # "vae":("VAE",),
            },
        }

    RETURN_TYPES = ("LATENTSYNCPIPELINE",)
    RETURN_NAMES = ("latentsync_pipeline",)
    FUNCTION = "load"
    CATEGORY = GLOBAL_CATEGORY

    def __init__(self):
        pass

    def load(self, ):
        models_path = folder_paths.get_folder_paths("latentsync")[0]

        vae = AutoencoderKL.from_pretrained(os.path.join(models_path, "stabilityai", "sd-vae-ft-mse"), device="cuda", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        if CONFIG.model.cross_attention_dim == 768:
            whisper_model = "small.pt"
        elif CONFIG.model.cross_attention_dim == 384:
            whisper_model = "tiny.pt"
        audio_encoder = Audio2Feature(
            model_path=os.path.join(models_path, "whisper",whisper_model),
            device="cuda",
            num_frames=CONFIG.data.num_frames,
            audio_feat_length=CONFIG.data.audio_feat_length,
        )

        denoising_unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(CONFIG.model),
            os.path.join(models_path, "latentsync_unet.pt"),
            device="cpu",
        )
        denoising_unet = denoising_unet.to(dtype=dtype)

        scheduler = DDIMScheduler.from_pretrained(CONFIG_DIR)

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        ).to("cuda")

        return pipeline,


class LatentSyncProcessingNode:
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "latentsync_pipeline":("LATENTSYNCPIPELINE",),
                "video_path":("STRING",),
                "audio_path":("STRING",),
                "output_path":("STRING",),
                "inference_steps":("INT",{"default":20}),
                "guidance_scale":("FLOAT",{"default":2.0,"min":1.0,"max":3.0,"step":0.1}),
                "seed":("INT",{"default":-1}),
            },
            "optional": {
                "insight_face":("INSIGHTFACE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    OUTPUT_NODE = True

    def __init__(self):
        pass

    def run(self, latentsync_pipeline, video_path, audio_path, output_path, inference_steps, guidance_scale, seed, insight_face=None, ):
        # pipeline(
        #     video_path=args.video_path,
        #     audio_path=args.audio_path,
        #     video_out_path=args.video_out_path,
        #     video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        #     num_frames=config.data.num_frames,
        #     num_inference_steps=args.inference_steps,
        #     guidance_scale=args.guidance_scale,
        #     weight_dtype=dtype,
        #     width=config.data.resolution,
        #     height=config.data.resolution,
        #     mask_image_path=config.data.mask_image_path,
        # )
        face_detector.insight_face = insight_face

        video_path = video_path.strip('"')
        audio_path = audio_path.strip('"')
        output_path = output_path.strip('"')

        latentsync_pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_path,
            video_mask_path=output_path.replace(".mp4", "_mask.mp4"),
            num_frames=CONFIG.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=dtype,
            width=CONFIG.data.resolution,
            height=CONFIG.data.resolution,
            mask_image_path=os.path.join(ROOT_DIR, "LatentSync", CONFIG.data.mask_image_path),
            seed=torch.seed() if seed==-1 else set_seed(seed),
        )

        return output_path,



NODE_CLASS_MAPPINGS = {
    "LipsyncPipelineNode":LipsyncPipelineNode,
    "LatentSyncProcessingNode":LatentSyncProcessingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LipsyncPipelineNode":"HJH-LatentSync Pipeline",
    "LatentSyncProcessingNode":"HJH-LatentSync Processing",
}