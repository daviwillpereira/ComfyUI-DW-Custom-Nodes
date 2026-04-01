import torch
import numpy as np
from PIL import Image

class DW_PureVQAExtractor:
    """
    Bypasses third-party wrapper limitations to execute strict <vqa> tasks natively.
    Relaxes type hinting to "*" to support Kijai's model dictionary.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "florence2_model": ("*",), 
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "multiline": True, 
                    "default": "Analyze the person and output strictly a comma-separated list answering these parameters in order: 1. Gender, 2. Age group (choose: baby, child, teenager, adult, elder), 3. Body build (choose: slim, regular, heavy, muscular), 4. Skin tone, 5. Hair color and style, 6. Eye color, 7. Beard style and color (or 'no beard'), 8. Wearing glasses (choose: 'no', 'prescription', 'sunglasses')."
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("VQA_OUTPUT",)
    FUNCTION = "extract_vqa"
    CATEGORY = "DW_Nodes/Vision"

    def extract_vqa(self, florence2_model, image: torch.Tensor, question: str):
        model = None
        processor = None
        
        # Robust extraction from Kijai's dictionary wrapper
        if isinstance(florence2_model, dict):
            model = florence2_model.get("model")
            processor = florence2_model.get("processor")
            dtype = florence2_model.get("dtype", torch.float16)
        else:
            model = getattr(florence2_model, "model", florence2_model)
            processor = getattr(florence2_model, "processor", None)
            dtype = getattr(florence2_model, "dtype", torch.float16)

        if model is None or processor is None:
            raise RuntimeError("[DW_FATAL] Invalid Florence-2 model. Ensure connection to a valid loader.")

        device = model.device

        # Process first image in batch
        img_tensor = image[0]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np).convert("RGB")

        task_prompt = f"<vqa> {question}"
        inputs = processor(text=task_prompt, images=pil_image, return_tensors="pt").to(device, dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height)
        )

        return (parsed_answer.get("<vqa>", str(parsed_answer)).strip(),)

NODE_CLASS_MAPPINGS = {"DW_PureVQAExtractor": DW_PureVQAExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_PureVQAExtractor": "DW Pure VQA Extractor"}
