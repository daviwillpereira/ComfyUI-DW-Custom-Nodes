import json
import torch
import numpy as np
import cv2
import random
import nodes # V16  FIX: Importa o motor Core do ComfyUI
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ==========================================
# 1. Data Transfer Objects (DTOs) & Constants
# ==========================================
@dataclass
class SceneDTO:
    width: int
    height: int
    floor_y_percent: float
    global_scale: float
    seed: Optional[int] = None
    original_characters: Optional[List['CharacterDTO']] = None # V28  FIX (Tipagem Correta)

@dataclass
class CharacterDTO:
    char_id: str
    gender: str
    age_group: str
    build: str
    z_index: int = 1
    parent_id: Optional[str] = None
    is_holding_baby: bool = False
    keypoints: Optional[Dict[int, Tuple[int, int]]] = None

# Official OpenPose RGB Color Table (18 keypoints)
POSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), 
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), 
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), 
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), 
    (255, 0, 170), (255, 0, 85)
]

# Bone Connections (Pairs)
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

# ==========================================
# 2. Biometric Engine & Factory (V7 )
# ==========================================
class BiometricFactory:
    """Factory Pattern to generate parametric bone structures with Ground Truth kinetics."""
    
    @staticmethod
    def get_metrics(char: CharacterDTO, base_height: int) -> Dict[str, float]:
        female_mod = 1.05 if char.gender == "female" else 1.0

        height_map = {"elder": 0.95, "adult": 1.0, "teenager": 0.85, "child": 0.6, "baby": 0.35}
        head_map = {"elder": 8.0, "adult": 8.0, "teenager": 7.0, "child": 6.0, "baby": 4.0 * female_mod}
        width_map = {"slim": 0.85, "regular": 1.0, "muscular": 1.15, "heavy": 1.25}
        
        rel_h = height_map.get(char.age_group, 1.0)
        h_ratio = head_map.get(char.age_group, 8.0)
        w_mod = width_map.get(char.build, 1.0)
        
        total_pixels = int(base_height * rel_h)
        head_pixels = int(total_pixels / h_ratio)
        
        # V13  FIX: Shoulders 1.55x, Hips 0.7x of shoulders (anatomical base)
        shoulder_width = int(head_pixels * 1.55 * w_mod)
        hip_width = int(shoulder_width * 0.70)
        
        arm_len = int(total_pixels * 0.4)
        leg_len = int(total_pixels * 0.5)

        # V13  FIX: Baby specific proportions (short legs, narrow shoulders)
        if char.age_group == "baby":
            shoulder_width = int(head_pixels * 1.15 * w_mod)
            hip_width = int(shoulder_width * 0.80)
            arm_len = int(total_pixels * 0.35)
            leg_len = int(total_pixels * 0.35)
        
        return {
            "total_h": total_pixels,
            "head_h": head_pixels,
            "shoulder_w": shoulder_width,
            "hip_w": hip_width,
            "arm_l": arm_len,
            "leg_l": leg_len
        }

    @staticmethod
    def build_skeleton(char: CharacterDTO, anchor_x: int, anchor_y: int, metrics: Dict[str, float], rng_context: random.Random, parent_kps: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict[int, Tuple[int, int]]:
        kps = {}
        th, hh, sw, hw = metrics["total_h"], metrics["head_h"], metrics["shoulder_w"], metrics["hip_w"]
        arm_l, leg_l = metrics["arm_l"], metrics["leg_l"]
        
        hip_tilt = 0.0
        shoulder_tilt = 0.0
        head_tilt = 0.0
        spine_curvature_offset = 0.0 
        
        if not char.parent_id and char.age_group != "baby":
            pelvis_tilt_offset = int(hh * 0.06) 
            hip_tilt = pelvis_tilt_offset
            shoulder_tilt = rng_context.uniform(-0.10, 0.10) * hh
            head_tilt = -shoulder_tilt * 0.8 + rng_context.uniform(-0.05, 0.05) * hh
            spine_curvature_offset = rng_context.uniform(-0.15, 0.15) * hh
            
        # --- Stage 1: Core & Curved Spine ---
        if char.parent_id and parent_kps:
            # V23  FIX: Perfil Estrito com Amputação Paramétrica Limpa
            is_left_hip = getattr(char, "_is_left_hip", True)

            hip_idx = 11 if is_left_hip else 8
            adult_hip = parent_kps.get(hip_idx, (anchor_x, anchor_y))
            adult_neck = parent_kps.get(1, (anchor_x, anchor_y - int(th*0.8)))
            
            waist_y = adult_hip[1] - int((adult_hip[1] - adult_neck[1]) * 0.3)
            baby_torso_len = int(th * 0.39)
            
            # Inclina na direção do adulto
            lean_direction = -1 if is_left_hip else 1 
            lean_offset = int(hw * 0.3) * lean_direction
            
            kps[1] = (adult_hip[0] + lean_offset, waist_y - baby_torso_len)
            kps[0] = (kps[1][0], kps[1][1] - hh)
            
            # Amputação do quadril oculto (Só desenhamos o quadril visível)
            if is_left_hip:
                kps[11] = (adult_hip[0], waist_y)
            else:
                kps[8] = (adult_hip[0], waist_y)
                
        elif char.age_group == "baby":
            kps[8] = (anchor_x - hw//2, anchor_y - int(leg_l * 0.4)) 
            kps[11] = (anchor_x + hw//2, anchor_y - int(leg_l * 0.4)) 
            kps[1] = (anchor_x, anchor_y - int(leg_l * 0.55)) 
            kps[0] = (anchor_x, kps[1][1] - int(hh * 0.8)) 
        else:
            r_hip_base = anchor_y - leg_l
            l_hip_base = anchor_y - leg_l
            kps[8] = (anchor_x - hw//2, int(r_hip_base + hip_tilt))  
            kps[11] = (anchor_x + hw//2, int(l_hip_base - hip_tilt)) 
            neck_y = anchor_y - leg_l - int(th * 0.3)
            kps[1] = (int(anchor_x + shoulder_tilt - spine_curvature_offset*0.5), neck_y) 
            kps[0] = (int(kps[1][0] + head_tilt), kps[1][1] - hh)
        
        # --- Stage 2: Face ---
        face_offset_x = int(hh * 0.2)
        face_offset_y = int(hh * 0.2)
        kps[14] = (kps[0][0] - face_offset_x, int(kps[0][1] - face_offset_y - head_tilt))
        kps[15] = (kps[0][0] + face_offset_x, int(kps[0][1] - face_offset_y + head_tilt))
        ear_offset_x = int(hh * 0.4)
        kps[16] = (kps[0][0] - ear_offset_x, int(kps[0][1] - head_tilt))
        kps[17] = (kps[0][0] + ear_offset_x, int(kps[0][1] + head_tilt))
        
        # --- Shoulders ---
        shoulder_drop = int(hh * 0.05) 
        if char.parent_id:
            # V23  FIX: Amputação do ombro oculto
            is_left_hip = getattr(char, "_is_left_hip", True)
            if is_left_hip:
                kps[5] = (kps[1][0], kps[1][1] + shoulder_drop) 
            else:
                kps[2] = (kps[1][0], kps[1][1] + shoulder_drop) 
        else:
            kps[2] = (kps[1][0] - sw//2, kps[1][1] + shoulder_drop + int(shoulder_tilt)) 
            kps[5] = (kps[1][0] + sw//2, kps[1][1] + shoulder_drop - int(shoulder_tilt))
        
        # --- Arms ---
        if char.parent_id:
            # V23  FIX: O braço visível abraça o adulto, o outro é amputado
            is_left_hip = getattr(char, "_is_left_hip", True)
            wrap_direction = -1 if is_left_hip else 1
            arm_reach_x = int(arm_l * 0.4) * wrap_direction
            
            if is_left_hip:
                # Omitimos o direito, desenhamos o esquerdo abraçando
                kps[6] = (kps[5][0] + arm_reach_x, kps[5][1] + int(arm_l*0.2))
                kps[7] = (kps[6][0] + int(arm_l*0.2)*wrap_direction, kps[6][1] - int(arm_l*0.1)) 
            else:
                # Omitimos o esquerdo, desenhamos o direito abraçando
                kps[3] = (kps[2][0] + arm_reach_x, kps[2][1] + int(arm_l*0.2))
                kps[4] = (kps[3][0] + int(arm_l*0.2)*wrap_direction, kps[3][1] - int(arm_l*0.1)) 
        elif char.age_group == "baby":
            elbow_drop = int(arm_l * 0.5)
            arm_spread = int(hw * 0.8)
            kps[3] = (kps[2][0] - arm_spread, kps[2][1] + elbow_drop)
            kps[4] = (kps[3][0], anchor_y)
            kps[6] = (kps[5][0] + arm_spread, kps[5][1] + elbow_drop)
            kps[7] = (kps[6][0], anchor_y)
        elif char.is_holding_baby:
            holding_left = getattr(char, "_holding_baby_on_left", True)
            
            # V25  FIX: Braço relaxado rente ao corpo (Evita cruzar Bounding Box)
            arm_swing_relax = rng_context.uniform(0.05, 0.08) * arm_l
            
            elbow_drop_hold = int(arm_l * 0.70)
            wrist_cross_x = int(hw * 0.6) 
            
            if holding_left:
                kps[6] = (kps[5][0] + int(arm_l*0.05), kps[5][1] + elbow_drop_hold)
                kps[7] = (kps[6][0] - wrist_cross_x, kps[6][1] - int(arm_l*0.05)) 
                kps[3] = (kps[2][0] - int(arm_swing_relax), kps[2][1] + int(arm_l*0.5))
                kps[4] = (kps[3][0] - int(arm_l*0.05), kps[3][1] + int(arm_l*0.5))
            else:
                kps[3] = (kps[2][0] - int(arm_l*0.05), kps[2][1] + elbow_drop_hold)
                kps[4] = (kps[3][0] + wrist_cross_x, kps[3][1] - int(arm_l*0.05)) 
                kps[6] = (kps[5][0] + int(arm_swing_relax), kps[5][1] + int(arm_l*0.5))
                kps[7] = (kps[6][0] + int(arm_l*0.05), kps[6][1] + int(arm_l*0.5))
        else:
            # V25  FIX: Braços rentes ao corpo no cluster
            arm_swing_r = rng_context.uniform(0.05, 0.08) * arm_l
            kps[3] = (kps[2][0] - int(arm_swing_r), kps[2][1] + int(arm_l*0.5))
            kps[4] = (kps[3][0] - int(arm_l*0.05), kps[3][1] + int(arm_l*0.5))
            arm_swing_l = rng_context.uniform(0.05, 0.08) * arm_l
            kps[6] = (kps[5][0] + int(arm_swing_l), kps[5][1] + int(arm_l*0.5))
            kps[7] = (kps[6][0] + int(arm_l*0.05), kps[6][1] + int(arm_l*0.5))
        
        # --- Legs ---
        if char.parent_id:
            # V23  FIX: A perna visível "abraça" a cintura, a outra é amputada
            is_left_hip = getattr(char, "_is_left_hip", True)
            knee_drop = int(leg_l * 0.3)
            ankle_drop = int(leg_l * 0.5)
            
            wrap_direction = -1 if is_left_hip else 1
            knee_wrap = int(hw * 0.8) * wrap_direction

            if is_left_hip:
                kps[12] = (kps[11][0] + knee_wrap, kps[11][1] + knee_drop)
                kps[13] = (kps[12][0], kps[12][1] + ankle_drop)
            else:
                kps[9] = (kps[8][0] + knee_wrap, kps[8][1] + knee_drop)
                kps[10] = (kps[9][0], kps[9][1] + ankle_drop)
        elif char.age_group == "baby":
            knee_spread = int(hw * 1.2)
            kps[9] = (kps[8][0] - knee_spread, anchor_y)
            kps[10] = (kps[9][0] + int(hw*0.5), kps[9][1] - int(leg_l*0.3)) 
            kps[12] = (kps[11][0] + knee_spread, anchor_y)
            kps[13] = (kps[12][0] - int(hw*0.5), kps[12][1] - int(leg_l*0.3))
        else:
            leg_bend_r = rng_context.uniform(0.40, 0.45) * leg_l
            knee_offset_r = int(hw * 0.1) 
            kps[9] = (kps[8][0] + knee_offset_r, kps[8][1] + int(leg_bend_r))
            kps[10] = (kps[8][0], kps[9][1] + int(leg_l - leg_bend_r)) 
            leg_bend_l = rng_context.uniform(0.48, 0.52) * leg_l
            knee_offset_l = int(hw * 0.05) 
            kps[12] = (kps[11][0] - knee_offset_l, kps[11][1] + int(leg_bend_l))
            kps[13] = (kps[11][0], kps[12][1] + int(leg_l - leg_bend_l)) 
        
        return kps

# ==========================================
# 3. Graphics & Z-Buffer Engine
# ==========================================
class OpenPoseRenderer:
    """Handles spatial drawing and Volumetric Boolean subtraction for Masks."""
    
    def __init__(self, scene: SceneDTO):
        self.scene = scene
        self.blur_kernel = (11, 11)

    def draw_pose_and_masks(self, characters: List[CharacterDTO]) -> Tuple[np.ndarray, torch.Tensor]:
        canvas = np.zeros((self.scene.height, self.scene.width, 3), dtype=np.uint8)
        global_z_buffer = np.zeros((self.scene.height, self.scene.width), dtype=np.uint8)
        masks = {}

        # STAGE 1: VOLUMETRIC MASK CALCULATION (Front-to-Back)
        # We need this Stage to handle Boolean subtraction BEFORE we blur,
        # otherwise we get semantic bleeding
        sorted_chars_masks = sorted(characters, key=lambda c: c.z_index, reverse=True)
        
        for char in sorted_chars_masks:
            char_mask = np.zeros((self.scene.height, self.scene.width), dtype=np.uint8)
            
            # V27  FIX: Mask Dilation Otimizada (Corpo magro, Cabeça gorda)
            # Reduzimos a espessura dos membros drasticamente para evitar Semantic Bleeding
            base_thickness = 50 if char.build in ["heavy", "muscular"] else 35
            if char.age_group in ["child", "baby"]:
                base_thickness = int(base_thickness * 0.7)

            #  FIX: Draw Torso Volume (Solid Polygon)
            if all(k in char.keypoints for k in [1, 2, 5, 8, 11]):
                torso_pts = np.array([
                    char.keypoints[2],  # RShoulder
                    char.keypoints[5],  # LShoulder
                    char.keypoints[11], # LHip
                    char.keypoints[8],  # RHip
                ], np.int32)
                cv2.fillPoly(char_mask, [torso_pts], 255)
            
            # Draw Thick Limbs
            for pair in POSE_PAIRS:
                if pair[0] in char.keypoints and pair[1] in char.keypoints:
                    pt1 = char.keypoints[pair[0]]
                    pt2 = char.keypoints[pair[1]]
                    cv2.line(char_mask, pt1, pt2, 255, thickness=base_thickness)
            
            # V15  FIX: Expanded Head Volume for Hair Generation
            if 0 in char.keypoints:
                head_radius = 85 if char.gender == "female" else 75
                if char.age_group in ["child", "baby"]:
                    head_radius = int(head_radius * 0.7)
                cv2.circle(char_mask, char.keypoints[0], head_radius, 255, thickness=-1)

            # Apply Gaussian Blur BEFORE Z-Buffer subtraction to ensure clean edge preservation
            blurred_char_mask = cv2.GaussianBlur(char_mask, self.blur_kernel, 0)
            char_mask_exclusive = cv2.bitwise_and(blurred_char_mask, cv2.bitwise_not(global_z_buffer))
            masks[char.char_id] = char_mask_exclusive
            
            # Update the Global Z-Buffer with the HARD mask (no blur) to reserve space
            global_z_buffer = cv2.bitwise_or(global_z_buffer, char_mask)

        # STAGE 2: OPENPOSE CANVAS DRAWING (Back-to-Front)
        # Prevents visual overwrite by painting lines over background figures
        sorted_chars_canvas = sorted(characters, key=lambda c: c.z_index, reverse=False)
        
        for char in sorted_chars_canvas:
            # Draw the colored OpenPose Skeleton
            for pair in POSE_PAIRS:
                if pair[0] in char.keypoints and pair[1] in char.keypoints:
                    pt1 = char.keypoints[pair[0]]
                    pt2 = char.keypoints[pair[1]]
                    color = POSE_COLORS[pair[1]]
                    cv2.line(canvas, pt1, pt2, color, thickness=4, lineType=cv2.LINE_AA)
            
            for k, pt in char.keypoints.items():
                color = POSE_COLORS[k]
                cv2.circle(canvas, pt, 4, color, thickness=-1, lineType=cv2.LINE_AA)

        # V28  FIX: Preservação Estrita do Índice Original (Garante que Máscara 0 = Char 1)
        # Em vez de pegar a ordem bagunçada pelo Z-Buffer, iteramos o scene characters original
        mask_list = [torch.from_numpy(masks[c.char_id]).float() / 255.0 for c in self.scene.original_characters]
        masks_tensor = torch.stack(mask_list) if mask_list else torch.empty((0, self.scene.height, self.scene.width))
        
        bg_mask = cv2.bitwise_not(global_z_buffer)
        bg_mask_tensor = torch.from_numpy(bg_mask).float() / 255.0
        
        return canvas, masks_tensor, bg_mask_tensor

# ==========================================
# 3.5. Qwen-VL NLP Parser
# ==========================================
import json
import re

def parse_qwen_phenotypes(qwen_string: str) -> dict:
    data = {
        "gender": "male", "age_group": "adult", "exact_age": "30 years old",
        "build_cat": "regular", "exact_build": "average build",
        "skin": "natural skin", "hair": "simple hair", "eyes": "brown eyes",
        "beard": "no beard", "glasses": "no glasses", "outfit": "modern stylish casual clothes"
    }
    try:
        clean_str = re.sub(r'\s*```$', '', clean_str).strip()
        parsed = json.loads(clean_str)
        
        # Resilient mapping with fallbacks for hallucinated keys
        data["gender"] = parsed.get("gender", data["gender"]).lower()
        
        age_group = parsed.get("age_group", "").lower()
        data["age_group"] = age_group if age_group in ["baby", "child", "teenager", "adult", "elder"] else data["age_group"]
        
        data["exact_age"] = parsed.get("exact_age", data["exact_age"]).lower()
        
        # Fallback for "physical_build" hallucination
        build_cat = parsed.get("build_cat", parsed.get("physical_build", "")).lower()
        data["build_cat"] = build_cat if build_cat in ["slim", "regular", "heavy", "muscular"] else data["build_cat"]
        
        data["exact_build"] = parsed.get("exact_build", data["exact_build"]).lower()
        
        # Fallback for "skin_tone" hallucination
        data["skin"] = parsed.get("skin", parsed.get("skin_tone", data["skin"])).lower()
        
        # Fallback for "hair_style_and_color" hallucination
        data["hair"] = parsed.get("hair", parsed.get("hair_style_and_color", data["hair"])).lower()
        
        data["eyes"] = parsed.get("eyes", data["eyes"]).lower()
        
        # Fallback for "beard_style_and_color" hallucination
        data["beard"] = parsed.get("beard", parsed.get("beard_style_and_color", data["beard"])).lower()
        
        data["glasses"] = parsed.get("glasses", data["glasses"]).lower()
        data["outfit"] = parsed.get("outfit", data["outfit"]).lower()

    except json.JSONDecodeError as e:
        print(f"[DW_WARN] JSON Parser failed: {e}. Faulty String: {qwen_string}")
    except Exception as e:
        print(f"[DW_WARN] Unexpected Error parsing Phenotypes: {e}")
        
    return data

# ==========================================
# 4. Main ComfyUI Node
# ==========================================
class DW_DynamicPoseComposer:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 8}),
                "floor_y_percent": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),
                "global_scale": ("FLOAT", {"default": 0.70, "min": 0.3, "max": 2.0, "step": 0.01}),
                "vision_context": ("STRING", {"forceInput": True, "multiline": True}),
                "clip": ("CLIP",),
                "global_positive": ("STRING", {"multiline": True, "default": "RAW photo, 8k uhd, dslr"}),
                "global_negative": ("STRING", {"multiline": True, "default": "deformed, bad anatomy"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("POSE_CANVAS", "Z_BUFFER_MASKS", "TELEMETRY_REPORT", "POSITIVE", "NEGATIVE", "PHENOTYPES")
    FUNCTION = "generate_rigs"
    CATEGORY = "DW_Nodes/Pose"

    def generate_rigs(self, width: int, height: int, floor_y_percent: float, global_scale: float, vision_context: str, clip, global_positive: str, global_negative: str):
        
        vision_strings = [s.strip() for s in vision_context.split('|||') if s.strip()]
        num_characters = len(vision_strings)
        
        if num_characters == 0:
            raise ValueError("[DW] Critical Error: vision_context is empty. Connect the DW Qwen Batch Extractor.")

        rng_seed = random.randint(0, 9999999)
        rng_context = random.Random(rng_seed)

        scene = SceneDTO(
            width=width, height=height,
            floor_y_percent=floor_y_percent,
            global_scale=global_scale,
            seed=rng_seed
        )

        characters = []
        phenotypes_list = []
        
        available_adults = []
        for i in range(num_characters):
            v_data = parse_qwen_phenotypes(vision_strings[i])
            if v_data["age_group"] in ["adult", "elder"]:
                available_adults.append(f"person_{i}")
                
        used_adults = set()
        
        for i in range(num_characters):
            v_data = parse_qwen_phenotypes(vision_strings[i])
            
            char = CharacterDTO(
                char_id=f"person_{i}", 
                gender=v_data["gender"],
                age_group=v_data["age_group"], 
                build=v_data["build_cat"]
            )
            
            glasses = ", wearing glasses" if "no" not in v_data["glasses"] else ""
            beard = f", {v_data['beard']}" if "no" not in v_data['beard'] else ""
            traits = f"{v_data['skin']}, {v_data['hair']}, {v_data['eyes']}{beard}{glasses}"
                
            phenotype_line = f"{char.age_group} {char.gender}, {traits}"
            phenotypes_list.append(phenotype_line)
            
            if char.age_group == "baby":
                free_adults = [a for a in available_adults if a not in used_adults]
                if free_adults:
                    char.z_index = 2
                    char.parent_id = free_adults[0]
                    used_adults.add(free_adults[0])
                else:
                    char.parent_id = None
                    char.z_index = 1
                
            characters.append(char)
            
        phenotypes_output = "\n".join(phenotypes_list)

        for char in characters:
            if char.parent_id:
                parent_char = next((c for c in characters if c.char_id == char.parent_id), None)
                if parent_char:
                    parent_char.is_holding_baby = True
                    # V22  FIX: Decide de qual lado o adulto vai segurar o bebê ANTES de desenhar
                    parent_char._holding_baby_on_left = rng_context.choice([True, False])
                    char._is_left_hip = parent_char._holding_baby_on_left

        scene.original_characters = list(characters) # V28  FIX: Trava a ordem original

        # Process independent adults first, then dependent babies
        processing_order = sorted(characters, key=lambda c: c.parent_id is not None)

        # Cinematic base height lock
        base_h = int(height * 0.7 * scene.global_scale)
        # Floor anchoring lock
        floor_y = int(height * scene.floor_y_percent)
        
        # V25  FIX: Dynamic Z-Index Oclusion Engine (Strict Hierarchy)
        # Garante sincronia perfeita entre Máscaras e Canvas
        for char in characters:
            metrics = BiometricFactory.get_metrics(char, base_h)
            if char.parent_id:
                char.z_index = 99999 # Top priority (Bebês no colo)
            elif char.is_holding_baby:
                char.z_index = 50000 # Adulto com bebê dá um passo à frente
            else:
                # Adultos/Crianças livres ficam no fundo
                char.z_index = int(10000 / metrics["total_h"])

        # V26  FIX: Dynamic Bounding Box with Anti-Fusion Padding
        independent_chars = [c for c in processing_order if not c.parent_id]
        dependent_chars = [c for c in processing_order if c.parent_id]
        
        char_metrics = []
        cluster_total_width = 0
        
        # Pré-cálculo das larguras com padding (O(N) Sweep)
        for char in independent_chars:
            metrics = BiometricFactory.get_metrics(char, base_h)
            
            # V27  FIX: Aumento do Distanciamento Paramétrico
            # Adultos recebem 75% de folga extra; crianças 40%
            spacing_factor = 0.75 if char.age_group in ["adult", "elder", "teenager"] else 0.40
            personal_space = int(metrics["shoulder_w"] * spacing_factor)
            
            padded_width = metrics["shoulder_w"] + personal_space
            char_metrics.append((char, metrics, padded_width))
            cluster_total_width += padded_width
            
        # Ponto de partida X para o cluster ficar perfeitamente centrado no Canvas
        start_x = (width - cluster_total_width) // 2 if cluster_total_width < width else 50
        x_current = start_x
        
        adult_anchors = {} 
        
        # 1. Process Independent Adults/Children (1D Layout)
        for i, (char, metrics, padded_width) in enumerate(char_metrics):
            char_center_x = x_current + (padded_width // 2)
            
            # Z-Depth Jitter (micro passo no eixo Y para evitar clipping perfeito nos ombros)
            y_jitter = int(metrics["head_h"] * 0.08) if i % 2 == 0 else 0
            
            char.keypoints = BiometricFactory.build_skeleton(char, char_center_x, floor_y + y_jitter, metrics, rng_context)
            adult_anchors[char.char_id] = char.keypoints 
            
            x_current += padded_width

        # 2. Process Dependent Babies (Dependency Injection)
        for char in dependent_chars:
            metrics = BiometricFactory.get_metrics(char, base_h)
            parent_kps = adult_anchors.get(char.parent_id, None)
            char.keypoints = BiometricFactory.build_skeleton(char, 0, 0, metrics, rng_context, parent_kps=parent_kps)

        # Render stage
        renderer = OpenPoseRenderer(scene)
        pose_canvas_np, masks_tensor, bg_mask_tensor = renderer.draw_pose_and_masks(characters)
        
        # Convert RGB Numpy canvas to ComfyUI Tensor [Batch, H, W, C]
        pose_canvas_tensor = torch.from_numpy(pose_canvas_np).float() / 255.0
        pose_canvas_tensor = pose_canvas_tensor.unsqueeze(0)

        people = []
        for char in characters:
            pose_2d = []
            for i in range(18):
                if i in char.keypoints:
                    pose_2d.extend([float(char.keypoints[i][0]), float(char.keypoints[i][1]), 1.0])
                else:
                    pose_2d.extend([0.0, 0.0, 0.0])
            people.append({
                "pose_keypoints_2d": pose_2d,
                "face_keypoints_2d": None,
                "hand_left_keypoints_2d": None,
                "hand_right_keypoints_2d": None
            })
            
        pose_keypoint_str = json.dumps([{"people": people, "canvas_height": height, "canvas_width": width}])

        # =========================================================
        # V18 FIX: Telemetry, Wardrobe & BG Masking
        # =========================================================
        clip_encoder = nodes.CLIPTextEncode()
        cond_set_mask = nodes.ConditioningSetMask()
        cond_combine = nodes.ConditioningCombine()

        global_pos_cond_raw, = clip_encoder.encode(clip, global_positive)
        global_neg_cond, = clip_encoder.encode(clip, global_negative)
        
        bg_mask_tensor_unsqueeze = bg_mask_tensor.unsqueeze(0)
        final_pos_cond, = cond_set_mask.append(global_pos_cond_raw, bg_mask_tensor_unsqueeze, "default", 1.0)

        telemetry_lines = [
            "# 📊  TELEMETRY REPORT",
            f"**Seed:** `{rng_seed}`",
            "---",
            f"### 🌍 GLOBAL PROMPTS",
            f"**Positive:** `{global_positive}`",
            f"**Negative:** `{global_negative}`",
            "---",
            "### 👤 REGIONAL PROMPTS (Multiplexed)"
        ]

        for i, char in enumerate(characters):
            noun = "person"
            if char.gender == "male":
                noun = "man" if char.age_group in ["adult", "elder"] else "boy"
            else:
                noun = "woman" if char.age_group in ["adult", "elder"] else "girl"
            
            if char.age_group == "baby":
                noun = "baby"
                
            action_context = "being carried in arms" if char.parent_id else ("crawling on the floor" if char.age_group == "baby" else "standing, dynamic cinematic pose")
            
            # Resgatamos os dados diretos do Qwen parseados anteriormente
            v_data = parse_qwen_phenotypes(vision_strings[i]) if i < len(vision_strings) else None
            
            if v_data:
                glasses = ", wearing glasses" if "no" not in v_data["glasses"] else ""
                beard = f", {v_data['beard']}" if "no" not in v_data['beard'] else ""
                traits = f"{v_data['skin']}, {v_data['hair']}, {v_data['eyes']}{beard}{glasses}"
                exact_age = v_data["exact_age"]
                exact_build = v_data["exact_build"]
                outfit = v_data["outfit"]
            else:
                traits, exact_age, exact_build, outfit = "detailed face", "adult", "regular build", "stylish casual clothes"
            
            # Remove "wearing" if Qwen included it to prevent duplication
            clean_outfit = outfit.replace("wearing ", "").strip()
            
            regional_text = f"A photorealistic {exact_age} {exact_build} {noun}, {traits}, wearing {clean_outfit}, {action_context}, cinematic lighting"
            telemetry_lines.append(f"- **{char.char_id}**: `{regional_text}`")
            
            reg_cond, = clip_encoder.encode(clip, regional_text)
            char_mask = masks_tensor[i].unsqueeze(0)
            masked_reg_cond, = cond_set_mask.append(reg_cond, char_mask, "default", 1.0)
            final_pos_cond, = cond_combine.combine(final_pos_cond, masked_reg_cond)

        telemetry_lines.append("---")
        telemetry_lines.append("### 🦴 RAW POSE JSON")
        telemetry_lines.append("```json")
        telemetry_lines.append(pose_keypoint_str)
        telemetry_lines.append("```")
        
        telemetry_report_str = "\n".join(telemetry_lines)

        return (pose_canvas_tensor, masks_tensor, telemetry_report_str, final_pos_cond, global_neg_cond, phenotypes_output)
    
# Registration
NODE_CLASS_MAPPINGS = {
    "DW_DynamicPoseComposer": DW_DynamicPoseComposer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DW_DynamicPoseComposer": "DW Dynamic Pose Composer"
}