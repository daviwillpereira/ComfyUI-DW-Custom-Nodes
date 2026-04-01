import torch
import numpy as np
from PIL import Image
import gc
import comfy.model_management as mm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class DW_QwenBatchExtractor:
    """
    Enterprise-grade LVLM batch processor with self-contained model lifecycle.
    Manages its own VRAM footprint, quantization, and batch tensor iteration.
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
                    "default": "Analyze the person and output strictly a comma-separated list answering these parameters in order:\n1. Gender\n2. Age group (choose: baby, child, teenager, adult, elder)\n3. Body build (choose: slim, regular, heavy, muscular)\n4. Skin tone\n5. Hair color and style\n6. Eye color\n7. Beard style and color (or 'no beard')\n8. Wearing glasses (choose: 'no', 'prescription', 'sunglasses')\n\nOutput format example: male, adult, heavy, dark skin, black dreadlocks, brown eyes, full black beard, no\nDo not include any other text."
                }),
                "model": (["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"],),
                "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "seed": ("INT", {"default": 1675, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BATCH_VQA_OUTPUT",)
    FUNCTION = "process_batch"
    CATEGORY = "DW_Nodes/Vision"

    def load_model(self, model_name: str, quant: str):
        repo_id = f"Qwen/{model_name}"
        
        if self.model is not None and self.current_model_name == repo_id and self.current_quant == quant:
            return # Model already loaded with correct params

        print(f"[DW_INFO] Loading {repo_id} with {quant} quantization...")
        
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
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
        print("[DW_INFO] Qwen-VL Model loaded successfully.")

    def process_batch(self, images: torch.Tensor, question: str, model: str, quantization: str, keep_model_loaded: bool, temperature: float, max_new_tokens: int, seed: int) -> tuple[str]:
        
        torch.manual_seed(seed)
        self.load_model(model, quantization)

        batch_size = images.shape[0]
        results = []

        print(f"[DW_INFO] Starting Qwen-VL Batch Inference for {batch_size} images...")

        for i in range(batch_size):
            img_tensor = images[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=[pil_image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0)
                )
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            print(f"[DW_INFO] Image {i+1}/{batch_size} Extracted: {output_text}")
            results.append(output_text)

        aggregated_string = " | ".join(results)

        # Lifecycle Management: VRAM Flush
        if not keep_model_loaded:
            print("[DW_INFO] keep_model_loaded is False. Offloading model to free VRAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_name = None
            self.current_quant = None
            gc.collect()
            mm.soft_empty_cache()
            torch.cuda.empty_cache()

        return (aggregated_string,)

NODE_CLASS_MAPPINGS = {"DW_QwenBatchExtractor": DW_QwenBatchExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_QwenBatchExtractor": "DW Qwen Batch Extractor"}
