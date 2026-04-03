import torch
import torch.nn.functional as F
import nodes
import re

class DW_IdentityMultiplexer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "instantid": ("INSTANTID",),
                "insightface": ("FACEANALYSIS",),
                "control_net": ("CONTROL_NET",),
                
                "reference_image_batch": ("IMAGE",), 
                "p1_base_image": ("IMAGE",),
                "segm_detector": ("SEGM_DETECTOR",),
                
                "clip": ("CLIP",),
                "base_positive": ("CONDITIONING",),
                "base_negative": ("CONDITIONING",),
                "phenotypes_text": ("STRING", {"multiline": True, "forceInput": True}),
                
                # Parameters
                "segm_drop_size": ("INT", {"default": 150, "min": 1, "max": 1000, "step": 1}),
                "ipadapter_weight": ("FLOAT", {"default": 0.40, "min": -1.0, "max": 3.0, "step": 0.01}),
                "instantid_ip_weight": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 3.0, "step": 0.01}),
                "instantid_cn_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "MASK", "STRING")
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", "DEBUG_MASKS", "TELEMETRY_REPORT")
    FUNCTION = "multiplex_pipeline"
    CATEGORY = "DW_Nodes/Identity"

    def multiplex_pipeline(self, model, ipadapter, instantid, insightface, control_net, reference_image_batch, p1_base_image, segm_detector, clip, base_positive, base_negative, phenotypes_text, segm_drop_size, ipadapter_weight, instantid_ip_weight, instantid_cn_strength, clip_vision=None):
        
        telemetry = ["# 🧬 DUAL-INJECTION MULTIPLEXER REPORT", "---"]
        
        # 1. Dependency Check
        ImpactSegm = nodes.NODE_CLASS_MAPPINGS.get("SegmDetectorSEGS")
        ImpactSegmToMask = nodes.NODE_CLASS_MAPPINGS.get("ImpactSEGSToMaskBatch")
        IPAdapterAdvanced = nodes.NODE_CLASS_MAPPINGS.get("IPAdapterAdvanced")
        ApplyInstantIDAdvanced = nodes.NODE_CLASS_MAPPINGS.get("ApplyInstantIDAdvanced")
        CondSetMask = nodes.NODE_CLASS_MAPPINGS.get("ConditioningSetMask")
        CondCombine = nodes.NODE_CLASS_MAPPINGS.get("ConditioningCombine")
        ClipTextEncode = nodes.NODE_CLASS_MAPPINGS.get("CLIPTextEncode")

        if not all([ImpactSegm, ImpactSegmToMask, IPAdapterAdvanced, ApplyInstantIDAdvanced]):
            raise RuntimeError("ERROR: Missing required dependencies (Impact Pack, IPAdapter-Plus, or InstantID).")

        # 2. Internal SEGM Detection & Conversion
        try:
            segm_node = ImpactSegm()
            segs_func = getattr(segm_node, segm_node.FUNCTION)
            # threshold=0.5, dilation=1, crop_factor=2.5, drop_size=var, labels="person"
            segs_out = segs_func(segm_detector, p1_base_image, 0.5, 1, 2.5, segm_drop_size, "person")[0]
            
            mask_node = ImpactSegmToMask()
            mask_func = getattr(mask_node, mask_node.FUNCTION)
            semantic_masks_batch = mask_func(segs_out)[0] # Tensor [N, H, W]
        except Exception as e:
            raise RuntimeError(f"ERROR: Internal Segmentation Failed. Details: {e}")

        mask_count = semantic_masks_batch.shape[0]
        if mask_count == 0:
            raise ValueError("ERROR: YOLO detected 0 people. Lower the 'segm_drop_size'.")

        # 3. Auto Spatial Sorting (Left-to-Right)
        mask_centers = []
        for m in semantic_masks_batch:
            cols = torch.any(m > 0, dim=0)
            if not torch.any(cols):
                mask_centers.append(0)
            else:
                mask_centers.append(torch.nonzero(cols).float().mean().item())
        
        sorted_idx = sorted(range(len(mask_centers)), key=lambda k: mask_centers[k])
        semantic_masks_batch = semantic_masks_batch[sorted_idx]

        # 4. Internal Image Processing (Center Pad to 512x512)
        processed_faces = []
        target_size = 512
        for img in reference_image_batch: # [H, W, C]
            h, w, _ = img.shape
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            img_c = img.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
            img_resized = F.interpolate(img_c, size=(new_h, new_w), mode='bicubic', align_corners=False)
            
            pad_h, pad_w = target_size - new_h, target_size - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            processed_faces.append(img_padded.squeeze(0).permute(1, 2, 0))

        processed_image_batch = torch.stack(processed_faces) # [N, 512, 512, 3]

        # 5. Phenotype Initialization
        raw_phenotypes = [str(p) for p in phenotypes_text] if isinstance(phenotypes_text, list) else phenotypes_text.split('\n')
        phenotypes = [re.sub(r'<loc_\d+>', ' ', p).strip() for p in raw_phenotypes if p.strip()]

        telemetry.extend([
            f"**Faces Processed (Padded 512x512):** `{processed_image_batch.shape[0]}`",
            f"**Sorted Masks Extracted:** `{mask_count}`",
            "---",
            "### 🔄 INJECTION LOG"
        ])

        # 6. Core Instantiation
        ip_node = IPAdapterAdvanced()
        iid_node = ApplyInstantIDAdvanced()
        set_mask_node = CondSetMask()
        combine_node = CondCombine()
        encoder_node = ClipTextEncode()

        final_model = model
        accumulated_positive = []
        accumulated_negative = []

        # 7. Daisy-Chain Loop
        batch_size = processed_image_batch.shape[0]
        for i in range(batch_size):
            face_tensor = processed_image_batch[i].unsqueeze(0)
            char_mask = semantic_masks_batch[i if i < mask_count else mask_count - 1].unsqueeze(0)
            phenotype_str = phenotypes[i] if i < len(phenotypes) else ""
            
            log_entry = f"- **Subject {i}**: "

            try:
                ip_res = ip_node.apply_ipadapter(
                    model=final_model, ipadapter=ipadapter, image=face_tensor, weight=ipadapter_weight,
                    weight_type="linear", combine_embeds="concat", start_at=0.0, end_at=1.0,
                    embeds_scaling="V only", attn_mask=char_mask, clip_vision=clip_vision
                )
                final_model = ip_res[0]
                log_entry += "✅ IPA | "
            except Exception as e: log_entry += f"❌ IPA ERROR ({e}) | "

            char_pos, char_neg = base_positive, base_negative
            if phenotype_str:
                try:
                    encoded_pos, = encoder_node.encode(clip, phenotype_str)
                    char_pos, = set_mask_node.append(encoded_pos, char_mask, "default", 1.0)
                    log_entry += "🧬 Cond | "
                except Exception as e: log_entry += f"❌ COND ERROR ({e}) | "

            try:
                iid_res = iid_node.apply_instantid(
                    instantid=instantid, insightface=insightface, control_net=control_net,
                    image=face_tensor, model=final_model, positive=char_pos, negative=char_neg,
                    ip_weight=instantid_ip_weight, cn_strength=instantid_cn_strength, start_at=0.0,
                    end_at=1.0, noise=0.0, combine_embeds="average", mask=char_mask
                )
                final_model = iid_res[0]
                accumulated_positive.append(iid_res[1])
                accumulated_negative.append(iid_res[2])
                log_entry += "✅ IID"
            except Exception as e: log_entry += f"❌ IID ERROR ({e})"

            telemetry.append(log_entry)

        # 8. Parallel Fusion
        final_positive, final_negative = base_positive, base_negative
        for pos in accumulated_positive: final_positive, = combine_node.combine(final_positive, pos)
        for neg in accumulated_negative: final_negative, = combine_node.combine(final_negative, neg)

        return (final_model, final_positive, final_negative, semantic_masks_batch, "\n".join(telemetry))

# --- REGISTRATION ---
NODE_CLASS_MAPPINGS = {
    "DW_IdentityMultiplexer": DW_IdentityMultiplexer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DW_IdentityMultiplexer": "DW Identity Multiplexer"
}