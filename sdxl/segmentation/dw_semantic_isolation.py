import torch
import torch.nn.functional as F
import nodes
import json

class DW_SemanticIsolationEngine:
    """
    Semantic Isolation Engine.
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
                "mask_dilation": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1})
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
        
        # Dependency Injection: Fetching ComfyUI-Segment-Anything native node
        DinoNode = nodes.NODE_CLASS_MAPPINGS.get("GroundingDinoSAMSegment (segment anything)")
        if not DinoNode:
            raise RuntimeError("[DW] Critical: 'comfyui_segment_anything' is not installed. Pipeline aborted.")

        segment_node = DinoNode()
        segment_func = getattr(segment_node, segment_node.FUNCTION)

        batch_size = images.shape[0]
        isolated_images = []
        semantic_masks = []
        bbox_metrics = []

        for i in range(batch_size):
            img_tensor = images[i].unsqueeze(0)
            
            try:
                # Zero-Shot Detection & Segmentation
                res = segment_func(
                    sam_model=sam_model,
                    grounding_dino_model=grounding_dino_model,
                    image=img_tensor,
                    prompt=prompt,
                    threshold=threshold
                )
                
                raw_mask = res[1].squeeze(0) 
                
                # Morphological Dilation for loose hair strands
                dilated_mask = self._dilate_mask(raw_mask, mask_dilation)
                
                # RGB Isolation (Matrix Annihilation of Background)
                mask_rgb = dilated_mask.unsqueeze(-1).repeat(1, 1, 3)
                isolated_img = img_tensor.squeeze(0) * mask_rgb
                
                # Extract BBOX Coordinates for Telemetry/Phase 4
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
        telemetry = json.dumps(bbox_metrics, indent=2)

        return (final_images, final_masks, telemetry)

# --- REGISTRATION ---
NODE_CLASS_MAPPINGS = {"DW_SemanticIsolationEngine": DW_SemanticIsolationEngine}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_SemanticIsolationEngine": "DW Semantic Isolation Engine"}