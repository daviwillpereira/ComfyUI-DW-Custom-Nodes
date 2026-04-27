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
    The Ultimate SOTA Custom Node for PiAPI Kling Integration.
    Features: Kling 3.0 Multi-Shots Schema, Multi-CDN Edge Offloading, and Resilient Response Parsing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "bad quality, blurry, deformed"}),
                "version": (["3.0", "1.5"], {"default": "3.0"}),
                "mode": (["std", "pro"], {"default": "std"}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "9:16"}),
            },
            "optional": {
                "prompt_scene_2": ("STRING", {"multiline": True}),
                "duration_scene_2": ("INT", {"default": 0, "min": 0, "max": 10}),
                "prompt_scene_3": ("STRING", {"multiline": True}),
                "duration_scene_3": ("INT", {"default": 0, "min": 0, "max": 10}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "camera_zoom": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "camera_pan_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "camera_pan_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("video_frames", "mp4_file_path",)
    FUNCTION = "generate_payload"
    CATEGORY = "DW/Universal/API"

    def upload_tensor_to_cdn(self, tensor):
        image_array = tensor[0].cpu().numpy()
        image_array = (image_array * 255.0).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
            
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        img_bytes = buffer.getvalue()

        try:
            res = requests.post("https://envs.sh", files={"file": ("image.jpg", img_bytes, "image/jpeg")}, timeout=15)
            if res.status_code == 200: return res.text.strip()
        except Exception: pass 

        try:
            res = requests.post("https://uguu.se/upload.php", files={"files[]": ("image.jpg", img_bytes, "image/jpeg")}, timeout=15)
            if res.status_code == 200: return res.json()["files"][0]["url"]
        except Exception as e:
            raise RuntimeError(f"CDN Storage Cluster Failure: All ephemeral routes exhausted. {str(e)}")

        raise RuntimeError("CDN Upload Failed: Unknown routing error.")

    def load_video_to_tensor(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame).astype(np.float32) / 255.0
            frame = torch.from_numpy(frame)
            frames.append(frame)
        cap.release()
        
        if len(frames) > 0: return torch.stack(frames, dim=0)
        return None

    def generate_payload(self, prompt, negative_prompt, version, mode, duration, aspect_ratio, prompt_scene_2="", duration_scene_2=0, prompt_scene_3="", duration_scene_3=0, first_frame=None, last_frame=None, camera_zoom=0.0, camera_pan_x=0.0, camera_pan_y=0.0):
        api_key = os.getenv("PIAPI_API_KEY")
        if not api_key: raise ValueError("Authentication Failed: PIAPI_API_KEY environment variable is missing.")

        headers = {"x-api-key": api_key, "Content-Type": "application/json", "Accept": "application/json"}

        payload = {
            "model": "kling",
            "task_type": "video_generation", 
            "input": {"version": version, "mode": mode}
        }
        
        if negative_prompt.strip(): payload["input"]["negative_prompt"] = negative_prompt
            
        if any(v != 0.0 for v in [camera_zoom, camera_pan_x, camera_pan_y]):
            payload["input"]["camera_control"] = {"type": "simple", "config": {"zoom": float(camera_zoom), "pan_x": float(camera_pan_x), "pan_y": float(camera_pan_y)}}
            
        if first_frame is None and last_frame is None: payload["input"]["aspect_ratio"] = aspect_ratio

        shots = [{"prompt": prompt, "duration": int(duration)}]
        if prompt_scene_2 and prompt_scene_2.strip() and int(duration_scene_2) > 0: shots.append({"prompt": prompt_scene_2, "duration": int(duration_scene_2)})
        if prompt_scene_3 and prompt_scene_3.strip() and int(duration_scene_3) > 0: shots.append({"prompt": prompt_scene_3, "duration": int(duration_scene_3)})
            
        if len(shots) > 1:
            total_duration = sum(s["duration"] for s in shots)
            if total_duration > 15: raise ValueError(f"Constraint Violation: Multi-shot total duration ({total_duration}s) exceeds API limit.")
            payload["input"]["multi_shots"] = shots
            payload["input"]["prefer_multi_shots"] = True
        else:
            payload["input"]["prompt"] = prompt
            payload["input"]["duration"] = int(duration)

        if last_frame is not None:
            print("[DW-Node] Offloading last_frame to Edge CDN...")
            payload["input"]["image_tail"] = self.upload_tensor_to_cdn(last_frame)
            
        if first_frame is not None:
            print("[DW-Node] Offloading first_frame to Edge CDN...")
            payload["input"]["image"] = self.upload_tensor_to_cdn(first_frame)

        try:
            print("[DW-Node] Dispatching Micro-Payload to PiAPI...")
            res = requests.post("https://api.piapi.ai/api/v1/task", headers=headers, json=payload, timeout=30)
            if not res.ok: raise Exception(f"HTTP {res.status_code}: {res.text}")
                
            task_id = res.json().get("data", {}).get("task_id")
            if not task_id: raise Exception(f"Invalid API Response Structure: {res.text}")
            
            print(f"[DW-Node] Task {task_id} generated. Polling API...")
            while True:
                time.sleep(12)
                poll_res = requests.get(f"https://api.piapi.ai/api/v1/task/{task_id}", headers=headers, timeout=15)
                poll_data = poll_res.json()
                status = poll_data.get("data", {}).get("status")
                
                if status == "completed":
                    # SOTA FIX: Fallback parsing to catch both Kling 1.5 ("video_url") and Kling 3.0 ("video") payload schemas
                    output_data = poll_data.get("data", {}).get("output", {})
                    video_url = output_data.get("video") or output_data.get("video_url")
                    
                    if not video_url:
                        raise Exception(f"Parser Error: Could not locate MP4 URL in API response: {output_data}")
                        
                    print("[DW-Node] Render Complete! Downloading MP4...")
                    break
                elif status in ["failed", "canceled"]:
                    raise Exception(f"Task Failed at API Engine Level: {poll_data}")

            video_bytes = requests.get(video_url, timeout=120).content
            output_dir = folder_paths.get_output_directory()
            target_path = os.path.join(output_dir, f"JBoggo_Kling_v{version}_{task_id}.mp4")
            
            with open(target_path, "wb") as f:
                f.write(video_bytes)
                
            video_tensor = self.load_video_to_tensor(target_path)
            return (video_tensor, target_path,)
            
        except Exception as e:
            raise RuntimeError(f"Interop Failure: {str(e)}")

NODE_CLASS_MAPPINGS = {"PiAPI_Kling_Node": PiAPI_Kling_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"PiAPI_Kling_Node": "PiAPI Kling Ultimate Multi-Shot (DW)"}