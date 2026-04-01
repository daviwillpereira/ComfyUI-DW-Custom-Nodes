import torch
import numpy as np
from PIL import Image

class DW_PureVQAExtractor:
    """
    Executes the native <MORE_DETAILED_CAPTION> task to bypass VQA limitations.
    Provides dense phenotypic descriptions for downstream Regex parsing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "florence2_model": ("*",), 
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DENSE_CAPTION",)
    FUNCTION = "extract_caption"
    CATEGORY = "DW_Nodes/Vision"

    def extract_caption(self, florence2_model, image: torch.Tensor):
        model = None
        processor = None
        
        if isinstance(florence2_model, dict):
            model = florence2_model.get("model")
            processor = florence2_model.get("processor")
            dtype = florence2_model.get("dtype", torch.float16)
        else:
            model = getattr(florence2_model, "model", florence2_model)
            processor = getattr(florence2_model, "processor", None)
            dtype = getattr(florence2_model, "dtype", torch.float16)

        if model is None or processor is None:
            raise RuntimeError("[DW_FATAL] Invalid Florence-2 model loaded.")

        device = model.device
        img_tensor = image[0]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np).convert("RGB")

        # Architectural Pivot: Using native dense captioning instead of forced VQA
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task_prompt, images=pil_image, return_tensors="pt").to(device, dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height)
        )

        final_text = parsed_answer.get(task_prompt, str(parsed_answer)).strip()

        return (final_text,)

NODE_CLASS_MAPPINGS = {"DW_PureVQAExtractor": DW_PureVQAExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_PureVQAExtractor": "DW Dense Captioner"}
