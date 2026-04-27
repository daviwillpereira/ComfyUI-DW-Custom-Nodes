import torch
import torch.nn.functional as F
import nodes

class DW_IdentityMultiplexer:
    """
    Identity Injection Engine (Color & Skin Anchor).
    Consumes mathematically pure semantic masks and isolated RGB tensors from Phase 1.
    Delegates structural geometry to Phase 4, keeping Phase 2 strictly for Chromatic locking
    via IP-Adapter with Temporal Easing to preserve global lighting.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "isolated_image_batch": ("IMAGE",), 
                "semantic_mask_batch": ("MASK",), 
                "base_positive": ("CONDITIONING",),
                "base_negative": ("CONDITIONING",),
                "ipadapter_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 3.0, "step": 0.01}),
                "ipadapter_end_at": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", "TELEMETRY_REPORT")
    FUNCTION = "multiplex_pipeline"
    CATEGORY = "DW_Nodes/Identity"

    def _pad_to_512(self, image_tensor):
        h, w, _ = image_tensor.shape
        scale = 512 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_c = image_tensor.permute(2, 0, 1).unsqueeze(0)
        img_resized = F.interpolate(img_c, size=(new_h, new_w), mode='bicubic', align_corners=False)
        pad_h, pad_w = 512 - new_h, 512 - new_w
        return F.pad(img_resized, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0).squeeze(0).permute(1, 2, 0)

    def multiplex_pipeline(self, model, ipadapter, isolated_image_batch, semantic_mask_batch, base_positive, base_negative, ipadapter_weight, ipadapter_end_at, clip_vision=None):
        
        telemetry = ["# 🧬 CHROMATIC ANCHOR REPORT (Phase 2)", "---"]
        
        IPAdapterAdvanced = nodes.NODE_CLASS_MAPPINGS.get("IPAdapterAdvanced")
        if not IPAdapterAdvanced:
            raise RuntimeError("[DW] Critical: IPAdapterAdvanced node not found.")
            
        ip_node = IPAdapterAdvanced()
        final_model = model

        batch_size = isolated_image_batch.shape[0]
        mask_count = semantic_mask_batch.shape[0]
        processed_isolated_images = torch.stack([self._pad_to_512(img) for img in isolated_image_batch])

        for i in range(batch_size):
            ref_tensor = processed_isolated_images[i].unsqueeze(0)
            char_mask = semantic_mask_batch[i if i < mask_count else mask_count - 1].unsqueeze(0)
            log_entry = f"- **Subject {i}**: "

            try:
                final_model = ip_node.apply_ipadapter(
                    final_model, 
                    ipadapter, 
                    image=ref_tensor, 
                    weight=ipadapter_weight, 
                    weight_type="linear", 
                    combine_embeds="concat", 
                    start_at=0.0, 
                    end_at=ipadapter_end_at, 
                    embeds_scaling="V only", 
                    attn_mask=char_mask, 
                    clip_vision=clip_vision
                )[0]
                
                log_entry += f"✅ IPA Anchored (Weight: {ipadapter_weight} | Stops at: {int(ipadapter_end_at*100)}%)"
            except Exception as e: 
                log_entry += f"❌ IPA ERROR ({e})"

            telemetry.append(log_entry)

        telemetry.append("---")
        telemetry.append("⚠️ *InstantID bypassed intentionally. Structural geometry delegated to Phase 4.*")

        return (final_model, base_positive, base_negative, "\n".join(telemetry))

NODE_CLASS_MAPPINGS = {"DW_IdentityMultiplexer": DW_IdentityMultiplexer}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_IdentityMultiplexer": "DW Chromatic Multiplexer (P2)"}