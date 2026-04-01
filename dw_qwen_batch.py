import torch
import numpy as np
from PIL import Image
from typing import Any

class DW_QwenBatchExtractor:
    """
    Enterprise-grade LVLM batch processor.
    Iterates over an image tensor [B, H, W, C], executes Qwen-VL inference for each,
    and returns a concatenated string delimited by ' | ' for the DW Composer.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("*",), # Wildcard to accept third-party loaders (dict or obj)
                "images": ("IMAGE",), # Tensor shape [Batch, Height, Width, Channels]
                "question": ("STRING", {
                    "multiline": True, 
                    "default": "Analyze the person and output strictly a comma-separated list answering these parameters in order:\n1. Gender\n2. Age group (choose: baby, child, teenager, adult, elder)\n3. Body build (choose: slim, regular, heavy, muscular)\n4. Skin tone\n5. Hair color and style\n6. Eye color\n7. Beard style and color (or 'no beard')\n8. Wearing glasses (choose: 'no', 'prescription', 'sunglasses')\n\nOutput format example: male, adult, heavy, dark skin, black dreadlocks, brown eyes, full black beard, no\nDo not include any other text."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("BATCH_VQA_OUTPUT",)
    FUNCTION = "process_batch"
    CATEGORY = "DW_Nodes/Vision"

    def process_batch(self, qwen_model: Any, images: torch.Tensor, question: str) -> tuple[str]:
        # 1. Safely extract model and processor from third-party wrappers
        model = None
        processor = None
        
        if isinstance(qwen_model, dict):
            model = qwen_model.get("model")
            processor = qwen_model.get("processor")
        else:
            model = getattr(qwen_model, "model", qwen_model)
            processor = getattr(qwen_model, "processor", None)

        if model is None or processor is None:
            raise RuntimeError("[DW_FATAL] Qwen-VL Model or Processor not found. Check the loader connection.")

        device = model.device
        dtype = next(model.parameters()).dtype
        batch_size = images.shape[0]
        results = []

        print(f"[DW_INFO] Starting Qwen-VL Batch Inference for {batch_size} images...")

        # 2. Iterate sequentially over the batch to prevent VRAM OOM spikes
        for i in range(batch_size):
            img_tensor = images[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np).convert("RGB")

            # Qwen2.5-VL strict chat template structure
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[pil_image], padding=True, return_tensors="pt")
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                
            # Trim the prompt from the output IDs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            print(f"[DW_INFO] Image {i+1}/{batch_size} Extracted: {output_text}")
            results.append(output_text)

        # 3. Aggregate outputs with the exact delimiter expected by DW_DynamicPoseComposer
        aggregated_string = " | ".join(results)
        return (aggregated_string,)
NODE_CLASS_MAPPINGS = {"DW_QwenBatchExtractor": DW_QwenBatchExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_QwenBatchExtractor": "DW Qwen Batch Extractor"}
