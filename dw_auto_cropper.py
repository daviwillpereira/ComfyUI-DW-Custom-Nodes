import torch
import torch.nn.functional as F
import numpy as np
import nodes

class DW_SemanticIsolationEngine:
    """
    State-of-the-Art Semantic Isolation Engine.
    Uses Dependency Injection to interface with GroundingDINO and SAM HQ.
    Outputs mathematically pure semantic masks and isolated RGB tensors
    to prevent visual cross-contamination in Latent Space operations.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sam_model": ("SAM_MODEL",),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL",),
                "prompt": ("STRING", {"default": "head, face, hair"}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_dilation": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("ISOLATED_IMAGE_BATCH", "SEMANTIC_MASK_BATCH", "BBOX_METRICS")
    FUNCTION = "isolate_semantics"
    CATEGORY = "DW_Nodes/Vision"

    def _dilate_mask(self, mask: torch.Tensor, dilation: int) -> torch.Tensor:
        if dilation <= 0: return mask
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        kernel_size = dilation * 2 + 1
        dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=dilation)
        return dilated.squeeze(0).squeeze(0)

    def isolate_semantics(self, images, sam_model, grounding_dino_model, prompt, threshold, mask_dilation):
        
        # 1. Dependency Injection: Fetching ComfyUI-Segment-Anything native nodes
        DinoNode = nodes.NODE_CLASS_MAPPINGS.get("GroundingDinoSAMSegment (segment anything)")
        if not DinoNode:
            raise RuntimeError("[DW] Critical: 'comfyui_segment_anything' is not installed. DINO/SAM pipeline aborted.")

        segment_node = DinoNode()
        segment_func = getattr(segment_node, segment_node.FUNCTION)

        batch_size = images.shape[0]
        isolated_images = []
        semantic_masks = []
        bbox_metrics = []

        for i in range(batch_size):
            img_tensor = images[i].unsqueeze(0)
            
            try:
                # 2. Execute Zero-Shot Detection & Segmentation
                res = segment_func(
                    sam_model=sam_model,
                    grounding_dino_model=grounding_dino_model,
                    image=img_tensor,
                    prompt=prompt,
                    threshold=threshold
                )
                
                # SAM returns (IMAGE, MASK). We extract the mask.
                raw_mask = res[1].squeeze(0) # Shape: [H, W]
                
                # 3. Morphological Dilation to capture loose hair strands
                dilated_mask = self._dilate_mask(raw_mask, mask_dilation)
                
                # 4. RGB Isolation (Matrix Multiplication)
                # Multiply the original RGB image by the 0.0-1.0 mask to annihilate the background
                mask_rgb = dilated_mask.unsqueeze(-1).repeat(1, 1, 3)
                isolated_img = img_tensor.squeeze(0) * mask_rgb
                
                # 5. Extract BBOX Coordinates from the mask for Phase 4 Telemetry
                active_pixels = torch.nonzero(dilated_mask > 0.5)
                if active_pixels.nelement() == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = torch.min(active_pixels, dim=0)[0].tolist()
                    y_max, x_max = torch.max(active_pixels, dim=0)[0].tolist()
                    bbox = [x_min, y_min, x_max, y_max]

                isolated_images.append(isolated_img)
                semantic_masks.append(dilated_mask)
                bbox_metrics.append({"subject_index": i, "bbox": bbox})

            except Exception as e:
                raise RuntimeError(f"[DW] Semantic Isolation failed on Subject {i}. Details: {e}")

        final_images = torch.stack(isolated_images)
        final_masks = torch.stack(semantic_masks)
        
        import json
        telemetry = json.dumps(bbox_metrics, indent=2)

        return (final_images, final_masks, telemetry)

# --- REGISTRATION ---
NODE_CLASS_MAPPINGS = {
    "DW_SemanticIsolationEngine": DW_SemanticIsolationEngine
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DW_SemanticIsolationEngine": "DW Semantic Isolation Engine"
}