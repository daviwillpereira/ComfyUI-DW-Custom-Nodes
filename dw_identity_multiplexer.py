import torch
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
                
                "face_image_batch": ("IMAGE",),
                "semantic_masks_batch": ("MASK",),
                "clip": ("CLIP",),
                
                "base_positive": ("CONDITIONING",),
                "base_negative": ("CONDITIONING",),
                "phenotypes_text": ("STRING", {"multiline": True, "forceInput": True}),
                
                "ipadapter_weight": ("FLOAT", {"default": 0.40, "min": -1.0, "max": 3.0, "step": 0.01}),
                "instantid_ip_weight": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 3.0, "step": 0.01}),
                "instantid_cn_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", "TELEMETRY_REPORT")
    FUNCTION = "multiplex_dual_injection"
    CATEGORY = "DW_Nodes/Identity"

    def multiplex_dual_injection(self, model, ipadapter, instantid, insightface, control_net, face_image_batch, semantic_masks_batch, clip, base_positive, base_negative, phenotypes_text, ipadapter_weight, instantid_ip_weight, instantid_cn_strength, clip_vision=None):
        
        # 1. Dependency Injection (Fetching ComfyUI Native Nodes)
        IPAdapterAdvanced = nodes.NODE_CLASS_MAPPINGS.get("IPAdapterAdvanced")
        ApplyInstantIDAdvanced = nodes.NODE_CLASS_MAPPINGS.get("ApplyInstantIDAdvanced")
        CondSetMask = nodes.NODE_CLASS_MAPPINGS.get("ConditioningSetMask")
        CondCombine = nodes.NODE_CLASS_MAPPINGS.get("ConditioningCombine")
        ClipTextEncode = nodes.NODE_CLASS_MAPPINGS.get("CLIPTextEncode")

        if not all([IPAdapterAdvanced, ApplyInstantIDAdvanced, CondSetMask, CondCombine, ClipTextEncode]):
            raise RuntimeError("ERROR: Missing required custom nodes. Ensure IPAdapter-Plus and ComfyUI-InstantID are installed.")

        ip_node = IPAdapterAdvanced()
        iid_node = ApplyInstantIDAdvanced()
        set_mask_node = CondSetMask()
        combine_node = CondCombine()
        encoder_node = ClipTextEncode()

        # 2. Phenotype Parsing & Sanitization (Florence-2 artifacts)
        if isinstance(phenotypes_text, list):
            raw_phenotypes = [str(p) for p in phenotypes_text]
        elif isinstance(phenotypes_text, str):
            raw_phenotypes = phenotypes_text.split('\n')
        else:
            raw_phenotypes = []

        phenotypes = [re.sub(r'<loc_\d+>', ' ', p).strip() for p in raw_phenotypes if p.strip()]

        batch_size = face_image_batch.shape[0]
        mask_batch_size = semantic_masks_batch.shape[0]

        final_model = model
        accumulated_positive = []
        accumulated_negative = []

        telemetry = [
            "# 🧬 DUAL-INJECTION MULTIPLEXER REPORT",
            "---",
            f"**Faces Detected:** `{batch_size}`",
            f"**Semantic Masks Received:** `{mask_batch_size}`",
            f"**Phenotypes Parsed:** `{len(phenotypes)}`",
            "---",
            "### 🔄 MULTIPLEXING LOG"
        ]

        # 3. The Daisy-Chain & Parallel Conditioning Loop O(N)
        for i in range(batch_size):
            face_tensor = face_image_batch[i].unsqueeze(0)
            
            mask_idx = i if i < mask_batch_size else mask_batch_size - 1
            char_mask = semantic_masks_batch[mask_idx].unsqueeze(0)

            phenotype_str = phenotypes[i] if i < len(phenotypes) else ""
            log_entry = f"- **Subject {i}**: "

            # A. IP-Adapter Application (Style & Skin Pass)
            try:
                ip_res = ip_node.apply_ipadapter(
                    model=final_model,
                    ipadapter=ipadapter,
                    image=face_tensor,
                    weight=ipadapter_weight,
                    weight_type="linear",
                    combine_embeds="concat",
                    start_at=0.0,
                    end_at=1.0,
                    embeds_scaling="V only",
                    attn_mask=char_mask,
                    clip_vision=clip_vision
                )
                final_model = ip_res[0]
                log_entry += "✅ IPA | "
            except Exception as e:
                log_entry += f"❌ IPA ERROR ({str(e)}) | "

            # B. Semantic Text Encoding & Masking
            char_pos = base_positive
            char_neg = base_negative
            if phenotype_str:
                try:
                    encoded_pos, = encoder_node.encode(clip, phenotype_str)
                    char_pos, = set_mask_node.append(encoded_pos, char_mask, "default", 1.0)
                    log_entry += "🧬 Phenotype Encoded | "
                except Exception as e:
                    log_entry += f"❌ CLIP ERROR ({str(e)}) | "

            # C. InstantID Application (Biometric Lock & ControlNet)
            try:
                iid_res = iid_node.apply_instantid(
                    instantid=instantid,
                    insightface=insightface,
                    control_net=control_net,
                    image=face_tensor,
                    model=final_model,
                    positive=char_pos,
                    negative=char_neg,
                    ip_weight=instantid_ip_weight,
                    cn_strength=instantid_cn_strength,
                    start_at=0.0,
                    end_at=1.0,
                    noise=0.0,
                    combine_embeds="average",
                    mask=char_mask
                )
                final_model = iid_res[0]
                accumulated_positive.append(iid_res[1])
                accumulated_negative.append(iid_res[2])
                log_entry += "✅ IID"
            except Exception as e:
                log_entry += f"❌ IID ERROR ({str(e)})"

            telemetry.append(log_entry)

        # 4. Parallel Conditioning Fusion
        final_positive = base_positive
        final_negative = base_negative

        for pos in accumulated_positive:
            final_positive, = combine_node.combine(final_positive, pos)
        
        for neg in accumulated_negative:
            final_negative, = combine_node.combine(final_negative, neg)

        telemetry.append("---")
        telemetry.append("✅ Parallel Conditioning Combined Successfully")

        return (final_model, final_positive, final_negative, "\n".join(telemetry))

# --- REGISTRATION ---
NODE_CLASS_MAPPINGS = {
    "DW_IdentityMultiplexer": DW_IdentityMultiplexer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DW_IdentityMultiplexer": "DW Identity Multiplexer"
}