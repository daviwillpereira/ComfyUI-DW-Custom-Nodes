import torch
import nodes
import re # V5 SOTA FIX: Regex para o Sanitizador do Florence-2

class DW_IdentityMultiplexer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "face_image_batch": ("IMAGE",), # Lote de N Rostos
                "z_buffer_masks": ("MASK",),    # Lote de N Máscaras (Vindo da P2)
                "clip": ("CLIP",),
                "base_positive": ("CONDITIONING",), # O POSITIVE que saiu da P2
                
                # O Florence-2 pode devolver String ou Lista
                "phenotypes_text": ("STRING", {"multiline": True, "forceInput": True}),
                
                # ==================================
                # V4 SOTA FIX: Exposição Universal
                # ==================================
                "ipadapter_mode": (["FaceID", "Standard"],),
                "weight": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 3.0, "step": 0.01}),
                "weight_type": (["linear", "ease in", "ease out", "ease in-out", "reverse in-out", "weak penalty", "strong penalty", "style transfer", "composition", "strong style transfer"],),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],),
                
                # Exclusivo do FaceID
                "weight_faceidv2": ("FLOAT", {"default": 0.85, "min": -1.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "insightface": ("INSIGHTFACE",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("MODEL", "POSITIVE", "TELEMETRY_REPORT")
    FUNCTION = "multiplex_identities"
    CATEGORY = "DW_Nodes/Identity"

    def multiplex_identities(self, model, ipadapter, face_image_batch, z_buffer_masks, clip, base_positive, phenotypes_text, ipadapter_mode, weight, weight_type, combine_embeds, start_at, end_at, embeds_scaling, weight_faceidv2, insightface=None, clip_vision=None):
        
        # 1. Roteamento Dinâmico de Classes do IPAdapter_Plus
        if ipadapter_mode == "FaceID":
            IPAdapterApply = nodes.NODE_CLASS_MAPPINGS.get("IPAdapterFaceID")
            if not insightface:
                raise Exception("SOTA ERROR: Modo 'FaceID' selecionado, mas o modelo 'insightface' não foi conectado na porta amarela!")
        else:
            IPAdapterApply = nodes.NODE_CLASS_MAPPINGS.get("IPAdapterAdvanced")
            
        if not IPAdapterApply:
            raise Exception(f"SOTA ERROR: O nó '{'IPAdapterFaceID' if ipadapter_mode == 'FaceID' else 'IPAdapterAdvanced'}' não foi encontrado no sistema.")

        clip_encoder = nodes.CLIPTextEncode()
        cond_set_mask = nodes.ConditioningSetMask()
        cond_combine = nodes.ConditioningCombine()

        ipadapter_node = IPAdapterApply()

        # ==========================================
        # V5 SOTA FIX: Florence-2 Sanitizer (Regex)
        # ==========================================
        if isinstance(phenotypes_text, list):
            raw_phenotypes = [str(p) for p in phenotypes_text]
        elif isinstance(phenotypes_text, str):
            raw_phenotypes = phenotypes_text.split('\n')
        else:
            raw_phenotypes = []

        phenotypes = []
        for p in raw_phenotypes:
            # Arranca qualquer tag de coordenada alienígena <loc_XXX>
            clean_p = re.sub(r'<loc_\d+>', ' ', p)
            # Limpa espaços duplos deixados para trás
            clean_p = ' '.join(clean_p.split()).strip()
            if clean_p:
                phenotypes.append(clean_p)

        batch_size = face_image_batch.shape[0]
        mask_batch_size = z_buffer_masks.shape[0]

        final_model = model
        final_positive = base_positive

        telemetry_lines = [
            "# 🧬 SOTA IDENTITY MULTIPLEXER REPORT",
            "---",
            f"**Mode Selected:** `{ipadapter_mode}`",
            f"**Faces Detected in Batch:** `{batch_size}`",
            f"**Z-Buffer Masks Received:** `{mask_batch_size}`",
            f"**Phenotypes Parsed:** `{len(phenotypes)}`",
            "---",
            "### 🔄 MULTIPLEXING LOG"
        ]

        # 3. O Loop Multiplexador O(N)
        for i in range(batch_size):
            face_tensor = face_image_batch[i].unsqueeze(0)
            
            mask_idx = i if i < mask_batch_size else mask_batch_size - 1
            char_mask = z_buffer_masks[mask_idx].unsqueeze(0)

            # V5 SOTA FIX: Telemetria detalhada de Parâmetros do IP-Adapter
            log_entry = f"- **Identity {i+1}**: "
            log_entry += f"⚙️ `[W:{weight} | {weight_type} | {start_at}-{end_at}]` ➔ "

            # ==========================================
            # A. TRANCA VISUAL (Injeção de FaceID)
            # ==========================================
            try:
                if ipadapter_mode == "FaceID":
                    ipadapter_result = ipadapter_node.apply_ipadapter(
                        model=final_model,
                        ipadapter=ipadapter,
                        image=face_tensor,
                        weight=weight,
                        weight_faceidv2=weight_faceidv2,
                        weight_type=weight_type,
                        combine_embeds=combine_embeds,
                        start_at=start_at,
                        end_at=end_at,
                        embeds_scaling=embeds_scaling,
                        image_negative=None,
                        attn_mask=char_mask, 
                        clip_vision=clip_vision,
                        insightface=insightface
                    )
                else:
                    ipadapter_result = ipadapter_node.apply_ipadapter(
                        model=final_model,
                        ipadapter=ipadapter,
                        image=face_tensor,
                        weight=weight,
                        weight_type=weight_type,
                        combine_embeds=combine_embeds,
                        start_at=start_at,
                        end_at=end_at,
                        embeds_scaling=embeds_scaling,
                        image_negative=None,
                        attn_mask=char_mask, 
                        clip_vision=clip_vision
                    )
                final_model = ipadapter_result[0]
                log_entry += "✅ IP-Adapter Applied | "
            except Exception as e:
                log_entry += f"❌ IP-Adapter FAILED ({str(e)}) | "

            # ==========================================
            # B. TRANCA SEMÂNTICA (Injeção de Fenótipo)
            # ==========================================
            if i < len(phenotypes):
                phenotype_text = phenotypes[i]
                reg_cond, = clip_encoder.encode(clip, phenotype_text)
                masked_reg_cond, = cond_set_mask.append(reg_cond, char_mask, "default", 1.0)
                final_positive, = cond_combine.combine(final_positive, masked_reg_cond)
                log_entry += f"🧬 Phenotype: `{phenotype_text}`"
            else:
                log_entry += "⚠️ No Phenotype Provided"

            telemetry_lines.append(log_entry)

        telemetry_lines.append("---")
        telemetry_lines.append("### 🧠 FLORENCE-2 SANITIZED TEXT")
        telemetry_lines.append("```text")
        telemetry_lines.append(str(phenotypes))
        telemetry_lines.append("```")

        telemetry_report_str = "\n".join(telemetry_lines)

        return (final_model, final_positive, telemetry_report_str)

# --- REGISTRO DO NÓ ---
NODE_CLASS_MAPPINGS = {
    "DW_IdentityMultiplexer": DW_IdentityMultiplexer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DW_IdentityMultiplexer": "DW Identity Multiplexer SOTA"
}