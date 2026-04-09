import torch
import numpy as np
import cv2
from PIL import Image
import gc
import json
import re
import comfy.model_management as mm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class DW_QwenBatchExtractor:
    """
    Enterprise-grade LVLM batch processor with Split-Inference Fusion.
    Encapsulates internal NLP prompts and executes micro-domain passes 
    to prevent Attention Dilution. 
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.current_quant = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "extraction_mode": (["characters", "background"],),
                "model": (["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"],),
                "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": 1675, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "detail_images_batch": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BATCH_VQA_OUTPUT",)
    FUNCTION = "process_batch"
    CATEGORY = "DW_Nodes/Vision"

    def load_model(self, model_name: str, quant: str):
        repo_id = f"Qwen/{model_name}"
        if self.model is not None and self.current_model_name == repo_id and self.current_quant == quant:
            return 

        if self.model is not None:
            del self.model
            del self.processor
            gc.collect()
            mm.soft_empty_cache()

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = None

        if quant == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype)
        elif quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo_id, quantization_config=quantization_config, device_map="auto", torch_dtype=compute_dtype
        )
        self.current_model_name = repo_id
        self.current_quant = quant

    def _extract_mathematical_skin_tone(self, pil_image: Image.Image) -> str:
        img_np = np.array(pil_image.convert('RGB'))
        h, w, _ = img_np.shape
        cy, cx = h // 2, w // 2
        crop_h, crop_w = int(h * 0.4), int(w * 0.4)
        face_crop = img_np[cy - crop_h // 2 : cy + crop_h // 2, cx - crop_w // 2 : cx + crop_w // 2]
        
        pixels = np.float32(face_crop.reshape(-1, 3))
        n_colors = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        _, counts = np.unique(labels, return_counts=True)
        dominant_rgb = palette[np.argmax(counts)]
        
        fitzpatrick = {
            "pale/fair skin": np.array([248, 217, 206]), "light skin": np.array([243, 195, 179]),
            "medium skin": np.array([221, 162, 131]), "olive skin": np.array([197, 139, 102]),
            "medium-dark skin": np.array([138, 91, 68]), "dark skin": np.array([91, 58, 41]),
            "very dark skin": np.array([46, 29, 22])
        }
        
        closest_tone, min_dist = "medium skin", float('inf')
        for tone, ref_rgb in fitzpatrick.items():
            dist = np.linalg.norm(dominant_rgb - ref_rgb)
            if dist < min_dist:
                min_dist = dist; closest_tone = tone
        return closest_tone

    def _extract_json_from_image(self, pil_image: Image.Image, question: str, temperature: float, max_new_tokens: int) -> dict:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[pil_image], padding=True, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature if temperature > 0.0 else 1.0, do_sample=(temperature > 0.0))
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        clean_str = re.sub(r'\s*`{3}(?:json)?$', '', output_text, flags=re.IGNORECASE).strip()
        start_idx, end_idx = clean_str.find('{'), clean_str.rfind('}')
        if start_idx != -1: clean_str = clean_str[start_idx:end_idx+1] if (end_idx != -1 and end_idx > start_idx) else clean_str[start_idx:] + '}'
        clean_str = re.sub(r',\s*([\]}])', r'\1', clean_str)
        try: return json.loads(clean_str)
        except json.JSONDecodeError: return {}

    def process_batch(self, images: torch.Tensor, extraction_mode: str, model: str, quantization: str, keep_model_loaded: bool, temperature: float, max_new_tokens: int, seed: int, detail_images_batch: torch.Tensor = None) -> tuple[str]:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        self.load_model(model, quantization)
        batch_size = images.shape[0]
        results = []

        PROMPT_BODY = 'Analyze this full-body image. Output STRICTLY JSON. NEVER output "null" or "not visible".\nKeys:\n"gender"("male"|"female"), "age_group", "exact_age", "physical_build"("slim"|"regular"|"heavy"), "exact_build", "outfit_upper", "outfit_lower"(MUST be lower-body clothes like jeans/pants/skirt. NEVER output a shirt here), "outfit_footwear"(e.g., "sneakers")'
        
        PROMPT_BIO = 'Analyze this face. Output STRICTLY JSON.\nKeys:\n"eyes"(e.g. "brown eyes"), "beard_style_and_color"(Specify density/length: "light stubble", "clean-shaven", "massive long beard", "goatee"), "glasses", "distinctive_features"("none" or specify VISIBLE anomalies/traits like "vitiligo", "heavy freckles", "birthmark", "facial scars")'
        
        PROMPT_HAIR = 'Analyze ONLY the hair geometry. Output STRICTLY JSON.\nKeys:\n"hair_length", "hair_volume"("flat", "thin", "regular", "thick"), "hair_color"(Handle complex cases: "two-tone black and blonde", "salt and pepper", "pink dyed", or exact shades), "hair_texture"("straight", "wavy", "curly", "coily", "dreadlocks", "micro braids", "buzz cut", "bald"), "hair_style"("worn down", "ponytail", "man bun", "fade")'
        
        PROMPT_BG_SEMANTIC = 'Analyze background STRICTLY based on VISUAL EVIDENCE. If it is daytime, say day. Output STRICTLY JSON.\nKeys:\n"location_type"("famous_landmark"|"generic_location"), "location_name", "architecture_style", "ground_material", "lighting_conditions", "atmosphere"'
        
        PROMPT_BG_SPATIAL = 'Classify camera angle. Output STRICTLY JSON.\nKeys:\n"camera_angle" (MUST be EXACTLY ONE: "low_angle" if pointing UP at a tall structure/sky, "high_angle" ONLY if pointing DOWN at the floor from above, "eye_level" if straight ahead).'
        
        for i in range(batch_size):
            img_np = (np.clip(images[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            pil_base = Image.fromarray(img_np).convert("RGB")
            merged_data = {}

            if extraction_mode == "characters":
                body_data = self._extract_json_from_image(pil_base, PROMPT_BODY, temperature, max_new_tokens)
                
                face_data = {}
                if detail_images_batch is not None and i < detail_images_batch.shape[0]:
                    pil_detail = Image.fromarray((np.clip(detail_images_batch[i].cpu().numpy(), 0, 1) * 255).astype(np.uint8)).convert("RGB")
                    
                    # Split Inference: Biological Pass + Hair Geometry Pass
                    bio_data = self._extract_json_from_image(pil_detail, PROMPT_BIO, temperature, max_new_tokens)
                    hair_data = self._extract_json_from_image(pil_detail, PROMPT_HAIR, temperature, max_new_tokens)
                    math_skin = {"skin_tone": self._extract_mathematical_skin_tone(pil_detail)}
                    
                    face_data = {**bio_data, **hair_data, **math_skin}
                
                merged_data = {**body_data, **face_data}

            elif extraction_mode == "background":
                # Split Inference: Semantics + Spatial Engine
                semantic_data = self._extract_json_from_image(pil_base, PROMPT_BG_SEMANTIC, temperature, max_new_tokens)
                spatial_data = self._extract_json_from_image(pil_base, PROMPT_BG_SPATIAL, temperature, max_new_tokens)
                merged_data = {**semantic_data, **spatial_data}

            results.append(merged_data)

        if not keep_model_loaded:
            del self.model, self.processor
            self.model = self.processor = self.current_model_name = self.current_quant = None
            gc.collect(); mm.soft_empty_cache(); torch.cuda.empty_cache()

        return (json.dumps(results),)

NODE_CLASS_MAPPINGS = {"DW_QwenBatchExtractor": DW_QwenBatchExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_QwenBatchExtractor": "DW Qwen Batch Extractor"}