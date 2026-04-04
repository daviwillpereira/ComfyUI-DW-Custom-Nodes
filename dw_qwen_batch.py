import torch
import numpy as np
from PIL import Image
import gc
import json
import re
import comfy.model_management as mm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class DW_QwenBatchExtractor:
    """
    Enterprise-grade LVLM batch processor with Dual-Stream Multi-Image support.
    Injects both full-body and facial crops into the same multimodal prompt to 
    guarantee accurate macro and micro taxonomy extraction.
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
                "question": ("STRING", {
                    "multiline": True, 
                    "default": "Analyze the images and return a JSON object."
                }),
                "model": (["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"],),
                "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.1}),
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

        quantization_config = None
        if quant == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.processor = AutoProcessor.from_pretrained(repo_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.current_model_name = repo_id
        self.current_quant = quant

    def process_batch(self, images: torch.Tensor, question: str, model: str, quantization: str, keep_model_loaded: bool, temperature: float, max_new_tokens: int, seed: int, detail_images_batch: torch.Tensor = None) -> tuple[str]:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.load_model(model, quantization)
        batch_size = images.shape[0]
        results = []

        for i in range(batch_size):
            img_tensor = images[i]
            img_np = (np.clip(img_tensor.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            pil_base_image = Image.fromarray(img_np).convert("RGB")
            
            content_payload = [{"type": "image"}]
            pil_images = [pil_base_image]

            # Dual-Stream Injection: Add face crop to the context if available
            if detail_images_batch is not None and i < detail_images_batch.shape[0]:
                detail_tensor = detail_images_batch[i]
                detail_np = (np.clip(detail_tensor.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                pil_detail_image = Image.fromarray(detail_np).convert("RGB")
                content_payload.append({"type": "image"})
                pil_images.append(pil_detail_image)

            content_payload.append({"type": "text", "text": question})

            messages = [
                {
                    "role": "user",
                    "content": content_payload
                }
            ]
            
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=pil_images, padding=True, return_tensors="pt")
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0)
                )
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            clean_str = output_text
            clean_str = re.sub(r'\s*```$', '', clean_str).strip()
            start_idx = clean_str.find('{')
            end_idx = clean_str.rfind('}')
            
            if start_idx != -1:
                if end_idx != -1 and end_idx > start_idx:
                    clean_str = clean_str[start_idx:end_idx+1]
                else:
                    clean_str = clean_str[start_idx:] + '}'
            
            clean_str = re.sub(r',\s*([\]}])', r'\1', clean_str)
            
            try:
                results.append(json.loads(clean_str))
            except json.JSONDecodeError:
                results.append({"error": "parser_failure", "raw_content": output_text})

        aggregated_json_string = json.dumps(results)

        if not keep_model_loaded:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_name = None
            self.current_quant = None
            gc.collect()
            mm.soft_empty_cache()
            torch.cuda.empty_cache()

        return (aggregated_json_string,)

NODE_CLASS_MAPPINGS = {"DW_QwenBatchExtractor": DW_QwenBatchExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_QwenBatchExtractor": "DW Qwen Batch Extractor"}