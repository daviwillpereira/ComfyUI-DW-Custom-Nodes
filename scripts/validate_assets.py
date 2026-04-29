# filepath: scripts/validate_assets.py
import os
import urllib.request
from urllib.error import URLError, HTTPError
import ssl

class AssetValidator:
    """
    SOTA Zero-Dependency Pre-Flight Checker for CI/CD Pipelines.
    Validates GitHub Repositories and HuggingFace/CivitAI model endpoints
    using local .env credentials before deployment execution.
    """

    def __init__(self, env_path="../.env"):
        self.credentials = self._load_env(env_path)
        self.ssl_context = ssl._create_unverified_context()

    def _load_env(self, env_path):
        creds = {"HF_TOKEN": "", "CIVITAI_API_KEY": "", "GITHUB_TOKEN": ""}
        if not os.path.exists(env_path):
            print(f"[!] Warning: .env file not found at {env_path}")
            return creds

        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key in creds:
                        creds[key] = value.strip(' "\'')
        return creds

    def check_endpoint(self, url, asset_type="Model"):
        req = urllib.request.Request(url, method="HEAD")
        
        if "huggingface.co" in url and self.credentials["HF_TOKEN"]:
            req.add_header("Authorization", f"Bearer {self.credentials['HF_TOKEN']}")
        elif "github.com" in url and self.credentials["GITHUB_TOKEN"]:
            req.add_header("Authorization", f"Bearer {self.credentials['GITHUB_TOKEN']}")
        elif "civitai.com" in url and self.credentials["CIVITAI_API_KEY"]:
            req.add_header("Authorization", f"Bearer {self.credentials['CIVITAI_API_KEY']}")

        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        try:
            response = urllib.request.urlopen(req, context=self.ssl_context, timeout=10)
            status = response.getcode()
            print(f"[✅] HTTP {status} | {asset_type} Validated: {url}")
            return True
        except HTTPError as e:
            if e.code == 405 or e.code == 403:
                try:
                    fallback_req = urllib.request.Request(url, method="GET")
                    for k, v in req.headers.items():
                        fallback_req.add_header(k, v)
                    fallback_res = urllib.request.urlopen(fallback_req, context=self.ssl_context, timeout=10)
                    print(f"[✅] HTTP {fallback_res.getcode()} (GET Fallback) | {asset_type} Validated: {url}")
                    return True
                except HTTPError as fallback_err:
                    print(f"[❌] HTTP {fallback_err.code} | {asset_type} Failed: {url}")
                    return False
            print(f"[❌] HTTP {e.code} | {asset_type} Failed: {url}")
            return False
        except URLError as e:
            print(f"[❌] Network Error | {asset_type} Failed: {url} -> {e.reason}")
            return False

def execute_pre_flight():
    print("="*60)
    print("🚀 INITIATING ASSET VALIDATION PRE-FLIGHT CHECK")
    print("="*60)

    validator = AssetValidator()
    
    custom_nodes = [
        "https://github.com/rgthree/rgthree-comfy",
        "https://github.com/ltdrdata/ComfyUI-Manager",
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "https://github.com/1038lab/ComfyUI-RMBG",
        "https://github.com/kijai/ComfyUI-KJNodes",
        "https://github.com/chrisgoringe/cg-use-everywhere",
        "https://github.com/city96/ComfyUI-GGUF",
        "https://github.com/daviwillpereira/ComfyUI-DW-Custom-Nodes",
        "https://github.com/kijai/ComfyUI-HunyuanVideoWrapper",
        "https://github.com/kijai/ComfyUI-SUPIR",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
        # Observability & UX Tools
        "https://github.com/crystian/ComfyUI-Crystools",
        "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    ]

    models = [
        "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_I2V_720_fixed_fp8_e4m3fn.safetensors",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors",
        "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0Q_fp16.safetensors",
        "https://civitai.com/api/download/models/133832?type=Model&format=SafeTensor&size=full&fp=fp16" # Crystal Clear XL
    ]

    print("\n--- Validating Custom Nodes GitHub Pointers ---")
    all_nodes_valid = all([validator.check_endpoint(url, "Node") for url in custom_nodes])

    print("\n--- Validating Model Weights & Tensor Pointers ---")
    all_models_valid = all([validator.check_endpoint(url, "Tensor") for url in models])

    print("\n" + "="*60)
    if all_nodes_valid and all_models_valid:
        print("[SUCCESS] All CI/CD pointers verified. Architecture is clear for deployment.")
        exit(0)
    else:
        print("[FATAL] Dead links or Authentication failures detected. Aborting setup preparation.")
        exit(1)

if __name__ == "__main__":
    execute_pre_flight()