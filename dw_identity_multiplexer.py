import torch
import torch.nn.functional as F
import nodes

class DW_IdentityMultiplexer:
    """
    O(N) Identity Injection Engine with Strict Face BBOX Isolation.
    Generates face-only dilated masks directly from SEGS to prevent 
    IP-Adapter semantic bleeding onto the body (clothing/background colors).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "instantid": ("INSTANTID",),
                "insightface": ("FACEANALYSIS",),
                "control_net": ("CONTROL_NET",),
                
                "body_image_batch": ("IMAGE",), 
                "face_image_batch": ("IMAGE",), 
                "p1_base_image": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR",), 
                
                "clip": ("CLIP",),
                "base_positive": ("CONDITIONING",),
                "base_negative": ("CONDITIONING",),
                "phenotypes_text": ("STRING", {"multiline": True, "forceInput": True}),
                
                "mask_dilation": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "ipadapter_weight": ("FLOAT", {"default": 0.80, "min": -1.0, "max": 3.0, "step": 0.01}),
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

    def _pad_to_512(self, image_tensor):
        h, w, _ = image_tensor.shape
        scale = 512 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_c = image_tensor.permute(2, 0, 1).unsqueeze(0)
        img_resized = F.interpolate(img_c, size=(new_h, new_w), mode='bicubic', align_corners=False)
        pad_h, pad_w = 512 - new_h, 512 - new_w
        return F.pad(img_resized, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='constant', value=0).squeeze(0).permute(1, 2, 0)

    def multiplex_pipeline(self, model, ipadapter, instantid, insightface, control_net, body_image_batch, face_image_batch, p1_base_image, bbox_detector, clip, base_positive, base_negative, phenotypes_text, mask_dilation, ipadapter_weight, instantid_ip_weight, instantid_cn_strength, clip_vision=None):
        
        telemetry = ["# 🧬 DUAL-STREAM MULTIPLEXER REPORT", "---"]
        
        ImpactBbox = nodes.NODE_CLASS_MAPPINGS.get("BboxDetectorSEGS")
        IPAdapterAdvanced = nodes.NODE_CLASS_MAPPINGS.get("IPAdapterAdvanced")
        ApplyInstantIDAdvanced = nodes.NODE_CLASS_MAPPINGS.get("ApplyInstantIDAdvanced")
        CondSetMask = nodes.NODE_CLASS_MAPPINGS.get("ConditioningSetMask")
        CondCombine = nodes.NODE_CLASS_MAPPINGS.get("ConditioningCombine")
        ClipTextEncode = nodes.NODE_CLASS_MAPPINGS.get("CLIPTextEncode")

        if not ImpactBbox:
            raise RuntimeError("ERROR: Impact Pack 'BboxDetectorSEGS' not found in ComfyUI.")

        # SOTA FIX: Extract Face SEGS natively and build masks in Python to avoid NoneType crashes
        try:
            bbox_node = ImpactBbox()
            segs_out = getattr(bbox_node, bbox_node.FUNCTION)(
                bbox_detector, p1_base_image, threshold=0.3, dilation=mask_dilation, crop_factor=1.0, drop_size=10, labels="face"
            )[0]
            
            canvas_shape = segs_out[0]
            seg_list = segs_out[1]
            
            if len(seg_list) == 0: 
                raise ValueError("ERROR: YOLO detected 0 faces.")
            
            h, w = canvas_shape[0], canvas_shape[1]
            face_masks = []
            mask_centers = []
            
            for seg in seg_list:
                crop_mask = getattr(seg, 'cropped_mask', None)
                x1, y1, x2, y2 = getattr(seg, 'crop_region', (0, 0, 0, 0))
                canvas_mask = torch.zeros((h, w), dtype=torch.float32)
                
                if crop_mask is not None:
                    if crop_mask.dim() == 3: crop_mask = crop_mask.squeeze(0)
                    ch, cw = crop_mask.shape
                    y2_actual, x2_actual = min(y1+ch, h), min(x1+cw, w)
                    canvas_mask[y1:y2_actual, x1:x2_actual] = crop_mask[:(y2_actual-y1), :(x2_actual-x1)]
                else:
                    canvas_mask[y1:y2, x1:x2] = 1.0 # Fallback solid box
                    
                face_masks.append(canvas_mask)
                mask_centers.append((x1 + x2) / 2.0)
                
            # Sort masks left-to-right to maintain index synchronization
            face_masks_batch = torch.stack([m for _, m in sorted(zip(mask_centers, face_masks), key=lambda k: k[0])])

        except Exception as e:
            raise RuntimeError(f"ERROR: Mask Building Failed. Details: {e}")

        mask_count = face_masks_batch.shape[0]
        processed_bodies = torch.stack([self._pad_to_512(img) for img in body_image_batch])
        processed_faces = torch.stack([self._pad_to_512(img) for img in face_image_batch])

        raw_phenotypes = [str(p) for p in phenotypes_text] if isinstance(phenotypes_text, list) else phenotypes_text.split('\n')
        phenotypes = [p.strip() for p in raw_phenotypes if p.strip()]

        ip_node, iid_node, set_mask_node, combine_node, encoder_node = IPAdapterAdvanced(), ApplyInstantIDAdvanced(), CondSetMask(), CondCombine(), ClipTextEncode()
        final_model, accumulated_positive, accumulated_negative = model, [], []

        batch_size = processed_bodies.shape[0]
        for i in range(batch_size):
            body_tensor = processed_bodies[i].unsqueeze(0)
            face_tensor = processed_faces[i if i < processed_faces.shape[0] else processed_faces.shape[0]-1].unsqueeze(0)
            char_mask = face_masks_batch[i if i < mask_count else mask_count - 1].unsqueeze(0)
            
            log_entry = f"- **Subject {i}**: "

            # 1. IPAdapter gets the BODY Tensor but injects ONLY into the FACE MASK
            try:
                final_model = ip_node.apply_ipadapter(final_model, ipadapter, image=body_tensor, weight=ipadapter_weight, weight_type="linear", combine_embeds="concat", start_at=0.0, end_at=1.0, embeds_scaling="V only", attn_mask=char_mask, clip_vision=clip_vision)[0]
                log_entry += "✅ IPA (Strict Face Mask) | "
            except Exception as e: log_entry += f"❌ IPA ERROR ({e}) | "

            char_pos, char_neg = base_positive, base_negative
            if i < len(phenotypes) and phenotypes[i]:
                encoded_pos, = encoder_node.encode(clip, phenotypes[i])
                char_pos, = set_mask_node.append(encoded_pos, char_mask, "default", 1.0)

            # 2. InstantID gets the HEADSHOT
            try:
                iid_res = iid_node.apply_instantid(instantid, insightface, control_net, image=face_tensor, model=final_model, positive=char_pos, negative=char_neg, ip_weight=instantid_ip_weight, cn_strength=instantid_cn_strength, start_at=0.0, end_at=1.0, noise=0.0, combine_embeds="average", mask=char_mask)
                final_model, pos_out, neg_out = iid_res[0], iid_res[1], iid_res[2]
                accumulated_positive.append(pos_out)
                accumulated_negative.append(neg_out)
                log_entry += "✅ IID (Face)"
            except Exception as e: log_entry += f"❌ IID ERROR ({e})"

            telemetry.append(log_entry)

        final_positive, final_negative = base_positive, base_negative
        for pos in accumulated_positive: final_positive, = combine_node.combine(final_positive, pos)
        for neg in accumulated_negative: final_negative, = combine_node.combine(final_negative, neg)

        return (final_model, final_positive, final_negative, face_masks_batch, "\n".join(telemetry))

# --- REGISTRATION ---
NODE_CLASS_MAPPINGS = {"DW_IdentityMultiplexer": DW_IdentityMultiplexer}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_IdentityMultiplexer": "DW Identity Multiplexer"}