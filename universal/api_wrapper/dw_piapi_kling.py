import os
import time
import requests
import base64
import cv2
import torch
from io import BytesIO
import numpy as np
from PIL import Image
import folder_paths

class PiAPI_Kling_Node:
    """
    SOTA Zero-Dependency Custom Node for PiAPI Kling Integration.
    Features: Multi-Shot UI, Direct Disk I/O, and VRAM Tensor Decoding for Post-Production.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "9:16"}),
            },
            "optional": {
                "prompt_scene_2": ("STRING", {"multiline": True}),
                "duration_scene_2": ("INT", {"default": 0, "min": 0, "max": 15}),
                "prompt_scene_3": ("STRING", {"multiline": True}),
                "duration_scene_3": ("INT", {"default": 0, "min": 0, "max": 15}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("video_frames", "mp4_file_path",)
    FUNCTION = "generate_payload"
    CATEGORY = "DW/Universal/API"

    def encode_tensor_to_base64(self, tensor):
        image_array = tensor[0].cpu().numpy()
        image_array = (image_array * 255.0).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def load_video_to_tensor(self, video_path):
        """Decodes MP4 binary into ComfyUI-compatible [F, H, W, C] tensors."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame).astype(np.float32) / 255.0
            frame = torch.from_numpy(frame)
            frames.append(frame)
        cap.release()
        
        if len(frames) > 0:
            return torch.stack(frames, dim=0)
        return None

    def generate_payload(self, prompt, duration, aspect_ratio, prompt_scene_2="", duration_scene_2=0, prompt_scene_3="", duration_scene_3=0, first_frame=None, last_frame=None):
        api_key = os.getenv("PIAPI_API_KEY")
        if not api_key:
            raise ValueError("Authentication Failed: PIAPI_API_KEY environment variable is missing.")

        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        payload = {
            "model": "kling",
            "task_type": "omni_video_generation", 
            "input": {
                "version": "3.0",
                "aspect_ratio": aspect_ratio
            }
        }
        
        shots = [{"prompt": prompt, "duration": int(duration)}]
        if prompt_scene_2 and prompt_scene_2.strip() and duration_scene_2 > 0:
            shots.append({"prompt": prompt_scene_2, "duration": int(duration_scene_2)})
        if prompt_scene_3 and prompt_scene_3.strip() and duration_scene_3 > 0:
            shots.append({"prompt": prompt_scene_3, "duration": int(duration_scene_3)})
            
        if len(shots) > 1:
            total_duration = sum(s["duration"] for s in shots)
            if total_duration > 15:
                raise ValueError(f"Constraint Violation: Multi-shot total duration ({total_duration}s) exceeds 15s limit.")
            payload["input"]["multi_shots"] = shots
        else:
            payload["input"]["prompt"] = prompt
            payload["input"]["duration"] = int(duration)

        if last_frame is not None:
            payload["input"]["image_tail"] = self.encode_tensor_to_base64(last_frame)
        if first_frame is not None:
            payload["input"]["image"] = self.encode_tensor_to_base64(first_frame)

        try:
            res = requests.post("https://api.piapi.ai/api/v1/task", headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            task_id = res.json().get("data", {}).get("task_id")
            
            while True:
                time.sleep(12)
                poll_res = requests.get(f"https://api.piapi.ai/api/v1/task/{task_id}", headers=headers, timeout=15)
                poll_data = poll_res.json()
                status = poll_data.get("data", {}).get("status")
                
                if status == "completed":
                    video_url = poll_data.get("data", {}).get("output", {}).get("video_url")
                    break
                elif status in ["failed", "canceled"]:
                    raise Exception(f"Task failed: {status}")

            # I/O Persistence
            video_bytes = requests.get(video_url, timeout=120).content
            output_dir = folder_paths.get_output_directory()
            target_path = os.path.join(output_dir, f"JBoggo_Kling_{task_id}.mp4")
            
            with open(target_path, "wb") as f:
                f.write(video_bytes)
                
            # VRAM Handover for Post-Production
            video_tensor = self.load_video_to_tensor(target_path)
            
            return (video_tensor, target_path,)
            
        except Exception as e:
            raise RuntimeError(f"Interop Failure: {str(e)}")

NODE_CLASS_MAPPINGS = {"PiAPI_Kling_Node": PiAPI_Kling_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"PiAPI_Kling_Node": "PiAPI Kling 3.0 (DW)"}