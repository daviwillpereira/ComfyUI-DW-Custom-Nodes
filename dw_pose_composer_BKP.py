import json
import torch
import numpy as np
import cv2
import random
import re
import nodes 
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
    original_characters: Optional[List['CharacterDTO']] = None 

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
# 2. Context & Domain Managers (SRP Enforcement)
# ==========================================
class VQAContextParser:
    """Parses raw text from VQA Vision nodes to extract explicit biometrics."""
    
    @staticmethod
    def parse_biometrics(vision_text: str) -> Dict[str, str]:
        if not vision_text:
            return {}
            
        text = vision_text.lower()
        build = "regular"
        
        if re.search(r'\b(heavy|chubby|fat|large|overweight|plump)\b', text):
            build = "heavy"
        elif re.search(r'\b(slim|skinny|thin|slender|petite)\b', text):
            build = "slim"
        elif re.search(r'\b(muscular|athletic|fit|jacked)\b', text):
            build = "muscular"
            
        return {"build": build}

class WardrobeManager:
    """Handles dynamic outfit allocation to prevent cloning within the same batch."""
    
    def __init__(self, rng_context: random.Random):
        outfits = {
            "man": ["wearing a burgundy knit sweater and dark trousers", "wearing a mustard yellow polo shirt and chinos", "wearing a forest green casual button-down shirt and jeans", "wearing a rust colored t-shirt and a light jacket"],
            "woman": ["wearing a crimson chic blouse and denim pants", "wearing an emerald green elegant midi dress", "wearing a burnt orange sleeveless top and wide-leg trousers", "wearing a bright yellow casual cardigan and jeans"],
            "boy": ["wearing a bright red graphic t-shirt and shorts", "wearing a lime green comfortable hoodie and jeans", "wearing a vibrant orange striped shirt and chinos"],
            "girl": ["wearing a bright pink floral dress", "wearing a teal casual t-shirt and denim skirt", "wearing a yellow colorful sweater and leggings"],
            "baby": ["wearing a cozy mint green pastel baby romper", "wearing a comfortable bright red cotton onesie", "wearing a cute lavender baby overall"]
        }
        
        self.active_wardrobe = {}
        for k, v_list in outfits.items():
            shuffled = list(v_list)
            rng_context.shuffle(shuffled)
            self.active_wardrobe[k] = shuffled

    def get_outfit(self, char: CharacterDTO) -> str:
        noun = "person"
        if char.gender == "male":
            noun = "man" if char.age_group in ["adult", "elder"] else "boy"
        else:
            noun = "woman" if char.age_group in ["adult", "elder"] else "girl"
        
        outfit_key = "baby" if char.age_group == "baby" else noun
            
        if self.active_wardrobe.get(outfit_key):
            return self.active_wardrobe[outfit_key].pop()
        
        return "wearing bright colored casual clothes"

# ==========================================
# 3. Biometric Engine & Factory
# ==========================================
class BiometricFactory:
    """Factory Pattern to generate parametric bone structures with Ground Truth kinetics."""
    
    @staticmethod
    def get_metrics(char: CharacterDTO, base_height: int) -> Dict[str, float]:
        female_mod = 1.05 if char.gender == "female" else 1.0

        height_map = {"elder": 0.95, "adult": 1.0, "teenager": 0.85, "child": 0.6, "baby": 0.35}
        head_map = {"elder": 8.0, "adult": 8.0, "teenager": 7.0, "child": 6.0, "baby": 4.0 * female_mod}
        
        rel_h = height_map.get(char.age_group, 1.0)
        h_ratio = head_map.get(char.age_group, 8.0)
        
        total_pixels = int(base_height * rel_h)
        head_pixels = int(total_pixels / h_ratio)
        
        # SOTA FIX: Non-Linear Deformation Injection
        w_mod = 1.0
        hip_ratio = 0.70
        leg_l_mod = 1.0
        
        if char.build == "heavy":
            w_mod = 1.25
            hip_ratio = 0.85 # Wider pelvis base
            leg_l_mod = 0.95 # Visually shorter legs
        elif char.build == "slim":
            w_mod = 0.85
            hip_ratio = 0.65 # Narrow pelvis
            leg_l_mod = 1.05 # Visually longer legs
        elif char.build == "muscular":
            w_mod = 1.15
            hip_ratio = 0.60 # V-Taper shape
            
        shoulder_width = int(head_pixels * 1.55 * w_mod)
        hip_width = int(shoulder_width * hip_ratio)
        
        arm_len = int(total_pixels * 0.4)
        leg_len = int(total_pixels * 0.5 * leg_l_mod)

        # Baby specific proportions (short legs, narrow shoulders)
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
            # Strict profile with clean parametric amputation
            is_left_hip = getattr(char, "_is_left_hip", True)

            hip_idx = 11 if is_left_hip else 8
            adult_hip = parent_kps.get(hip_idx, (anchor_x, anchor_y))
            adult_neck = parent_kps.get(1, (anchor_x, anchor_y - int(th*0.8)))
            
            waist_y = adult_hip[1] - int((adult_hip[1] - adult_neck[1]) * 0.3)
            baby_torso_len = int(th * 0.39)
            
            # Lean towards the adult
            lean_direction = -1 if is_left_hip else 1 
            lean_offset = int(hw * 0.3) * lean_direction
            
            kps[1] = (adult_hip[0] + lean_offset, waist_y - baby_torso_len)
            kps[0] = (kps[1][0], kps[1][1] - hh)
            
            # Amputate hidden hip (Only draw visible hip)
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
            # Hidden shoulder amputation
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
            # Visible arm wraps the adult, the other is amputated
            is_left_hip = getattr(char, "_is_left_hip", True)
            wrap_direction = -1 if is_left_hip else 1
            arm_reach_x = int(arm_l * 0.4) * wrap_direction
            
            if is_left_hip:
                # Omit right, draw left wrapping
                kps[6] = (kps[5][0] + arm_reach_x, kps[5][1] + int(arm_l*0.2))
                kps[7] = (kps[6][0] + int(arm_l*0.2)*wrap_direction, kps[6][1] - int(arm_l*0.1)) 
            else:
                # Omit left, draw right wrapping
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
            
            # Relaxed arm close to the body (Prevents Bounding Box crossing)
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
            # Arms close to body in the cluster
            arm_swing_r = rng_context.uniform(0.05, 0.08) * arm_l
            kps[3] = (kps[2][0] - int(arm_swing_r), kps[2][1] + int(arm_l*0.5))
            kps[4] = (kps[3][0] - int(arm_l*0.05), kps[3][1] + int(arm_l*0.5))
            arm_swing_l = rng_context.uniform(0.05, 0.08) * arm_l
            kps[6] = (kps[5][0] + int(arm_swing_l), kps[5][1] + int(arm_l*0.5))
            kps[7] = (kps[6][0] + int(arm_l*0.05), kps[6][1] + int(arm_l*0.5))
        
        # --- Legs ---
        if char.parent_id:
            # Visible leg wraps the waist, the other is amputated
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
# 4. Graphics & Z-Buffer Engine
# ==========================================
class OpenPoseRenderer:
    """Handles spatial drawing and Volumetric Boolean subtraction for Masks."""
    
    def __init__(self, scene: SceneDTO):
        self.scene = scene
        self.blur_kernel = (11, 11)

    def draw_pose_and_masks(self, characters: List[CharacterDTO]) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        canvas = np.zeros((self.scene.height, self.scene.width, 3), dtype=np.uint8)
        global_z_buffer = np.zeros((self.scene.height, self.scene.width), dtype=np.uint8)
        masks = {}

        # STAGE 1: VOLUMETRIC MASK CALCULATION (Front-to-Back)
        sorted_chars_masks = sorted(characters, key=lambda c: c.z_index, reverse=True)
        
        for char in sorted_chars_masks:
            char_mask = np.zeros((self.scene.height, self.scene.width), dtype=np.uint8)
            
            # Optimized Mask Dilation (Thin limbs to prevent Semantic Bleeding)
            base_thickness = 50 if char.build in ["heavy", "muscular"] else 35
            if char.age_group in ["child", "baby"]:
                base_thickness = int(base_thickness * 0.7)

            # Draw Torso Volume (Solid Polygon)
            if all(k in char.keypoints for k in [1, 2, 5, 8, 11]):
                torso_pts = np.array([
                    char.keypoints[2],  
                    char.keypoints[5],  
                    char.keypoints[11], 
                    char.keypoints[8],  
                ], np.int32)
                cv2.fillPoly(char_mask, [torso_pts], 255)
            
            # Draw Thick Limbs
            for pair in POSE_PAIRS:
                if pair[0] in char.keypoints and pair[1] in char.keypoints:
                    pt1 = char.keypoints[pair[0]]
                    pt2 = char.keypoints[pair[1]]
                    cv2.line(char_mask, pt1, pt2, 255, thickness=base_thickness)
            
            # Expanded Head Volume for Hair Generation
            if 0 in char.keypoints:
                head_radius = 85 if char.gender == "female" else 75
                if char.age_group in ["child", "baby"]:
                    head_radius = int(head_radius * 0.7)
                cv2.circle(char_mask, char.keypoints[0], head_radius, 255, thickness=-1)

            # Apply Gaussian Blur BEFORE Z-Buffer subtraction to ensure clean edge preservation
            blurred_char_mask = cv2.GaussianBlur(char_mask, self.blur_kernel, 0)
            char_mask_exclusive = cv2.bitwise_and(blurred_char_mask, cv2.bitwise_not(global_z_buffer))
            masks[char.char_id] = char_mask_exclusive
            
            # Update the Global Z-Buffer with the HARD mask
            global_z_buffer = cv2.bitwise_or(global_z_buffer, char_mask)

        # STAGE 2: OPENPOSE CANVAS DRAWING (Back-to-Front)
        sorted_chars_canvas = sorted(characters, key=lambda c: c.z_index, reverse=False)
        
        for char in sorted_chars_canvas:
            for pair in POSE_PAIRS:
                if pair[0] in char.keypoints and pair[1] in char.keypoints:
                    pt1 = char.keypoints[pair[0]]
                    pt2 = char.keypoints[pair[1]]
                    color = POSE_COLORS[pair[1]]
                    cv2.line(canvas, pt1, pt2, color, thickness=4, lineType=cv2.LINE_AA)
            
            for k, pt in char.keypoints.items():
                color = POSE_COLORS[k]
                cv2.circle(canvas, pt, 4, color, thickness=-1, lineType=cv2.LINE_AA)

        # Strict preservation of Original Index (Guarantees Mask 0 = Char 1)
        mask_list = [torch.from_numpy(masks[c.char_id]).float() / 255.0 for c in self.scene.original_characters]
        masks_tensor = torch.stack(mask_list) if mask_list else torch.empty((0, self.scene.height, self.scene.width))
        
        bg_mask = cv2.bitwise_not(global_z_buffer)
        bg_mask_tensor = torch.from_numpy(bg_mask).float() / 255.0
        
        return canvas, masks_tensor, bg_mask_tensor

# ==========================================
# 5. Main ComfyUI Node
# ==========================================
class DW_DynamicPoseComposer:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "clip": ("CLIP",),
                "global_positive": ("STRING", {"multiline": True, "default": "RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT4, highly detailed"}),
                "global_negative": ("STRING", {"multiline": True, "default": "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, watermark, watermarked, oversaturated, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"}),
                "payload_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "vision_context": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("POSE_CANVAS", "Z_BUFFER_MASKS", "TELEMETRY_REPORT", "POSITIVE", "NEGATIVE", "PHENOTYPES")
    FUNCTION = "generate_rigs"
    CATEGORY = "DW_Nodes/Pose"

    def generate_rigs(self, background_image: torch.Tensor, clip, global_positive: str, global_negative: str, payload_json: str, vision_context: str = ""):
        _, height, width, _ = background_image.shape
        
        try:
            data = json.loads(payload_json)
        except json.JSONDecodeError:
            raise ValueError("[DW] Critical Error: Payload JSON is corrupted or poorly formatted.")

        scene_data = data.get("scene", {})
        
        passed_seed = scene_data.get("seed", None)
        rng_seed = int(passed_seed) if passed_seed is not None else random.randint(0, 9999999)
        
        print(f"--- [DW] Spawning Family (Deterministic Debugging) ---")
        print(f"[DW] Generation Seed: {rng_seed} (Pass this in JSON 'scene.seed' to reproduce poses)")
        
        rng_context = random.Random(rng_seed)
        wardrobe_manager = WardrobeManager(rng_context)

        scene = SceneDTO(
            width=width, 
            height=height,
            floor_y_percent=float(scene_data.get("floor_y_percent", 0.85)),
            global_scale=float(scene_data.get("global_scale", 1.0)),
            seed=rng_seed
        )

        characters = []
        raw_chars = data.get("characters", [])
        
        available_adults = [c.get("id", "unknown") for c in raw_chars if c.get("age_group", "adult") in ["adult", "elder"]]
        used_adults = set()
        
        phenotypes_list = []
        vqa_data = VQAContextParser.parse_biometrics(vision_context)
        
        for raw_c in raw_chars:
            char_build = raw_c.get("build", "regular")
            
            # Context Injection Engine: Resolve "auto" payload flag via VQA
            if char_build.lower() == "auto":
                char_build = vqa_data.get("build", "regular")

            char = CharacterDTO(
                char_id=raw_c.get("id", "unknown"), 
                gender=raw_c.get("gender", "male"),
                age_group=raw_c.get("age_group", "adult"), 
                build=char_build
            )
            
            traits = raw_c.get("traits", "").strip()
            
            if traits:
                phenotype_line = f"{char.age_group} {char.gender}, {traits}"
            else:
                phenotype_line = f"{char.age_group} {char.gender}"
                
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
                    parent_char._holding_baby_on_left = rng_context.choice([True, False])
                    char._is_left_hip = parent_char._holding_baby_on_left

        scene.original_characters = list(characters) 

        processing_order = sorted(characters, key=lambda c: c.parent_id is not None)

        base_h = int(height * 0.7 * scene.global_scale)
        floor_y = int(height * scene.floor_y_percent)
        
        # Dynamic Z-Index Oclusion Engine (Strict Hierarchy)
        for char in characters:
            metrics = BiometricFactory.get_metrics(char, base_h)
            if char.parent_id:
                char.z_index = 99999 
            elif char.is_holding_baby:
                char.z_index = 50000 
            else:
                char.z_index = int(10000 / metrics["total_h"])

        # Dynamic Bounding Box with Anti-Fusion Padding
        independent_chars = [c for c in processing_order if not c.parent_id]
        dependent_chars = [c for c in processing_order if c.parent_id]
        
        char_metrics = []
        cluster_total_width = 0
        
        for char in independent_chars:
            metrics = BiometricFactory.get_metrics(char, base_h)
            
            # Parametric Distance Adjustments
            spacing_factor = 0.75 if char.age_group in ["adult", "elder", "teenager"] else 0.40
            personal_space = int(metrics["shoulder_w"] * spacing_factor)
            
            padded_width = metrics["shoulder_w"] + personal_space
            char_metrics.append((char, metrics, padded_width))
            cluster_total_width += padded_width
            
        start_x = (width - cluster_total_width) // 2 if cluster_total_width < width else 50
        x_current = start_x
        
        adult_anchors = {} 
        
        # 1. Process Independent Adults/Children (1D Layout)
        for i, (char, metrics, padded_width) in enumerate(char_metrics):
            char_center_x = x_current + (padded_width // 2)
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
        # Conditioning & Telemetry
        # =========================================================
        clip_encoder = nodes.CLIPTextEncode()
        cond_set_mask = nodes.ConditioningSetMask()
        cond_combine = nodes.ConditioningCombine()

        global_pos_cond_raw, = clip_encoder.encode(clip, global_positive)
        global_neg_cond, = clip_encoder.encode(clip, global_negative)
        
        # Background Isolation Lock
        bg_mask_tensor_unsqueeze = bg_mask_tensor.unsqueeze(0)
        final_pos_cond, = cond_set_mask.append(global_pos_cond_raw, bg_mask_tensor_unsqueeze, "default", 1.0)

        telemetry_lines = [
            "# 📊 SOTA TELEMETRY REPORT",
            f"**Seed:** `{rng_seed}`",
            "---",
            f"### 🌍 GLOBAL PROMPTS",
            f"**Positive:** `{global_positive}`",
            f"**Negative:** `{global_negative}`",
            "---",
            "### 👤 REGIONAL PROMPTS (Multiplexed)"
        ]

        if vision_context:
            telemetry_lines.append(f"*(VQA Parsed Context: {vqa_data})*")

        for i, char in enumerate(characters):
            noun = "person"
            if char.gender == "male":
                noun = "man" if char.age_group in ["adult", "elder"] else "boy"
            else:
                noun = "woman" if char.age_group in ["adult", "elder"] else "girl"
            
            if char.age_group == "baby":
                noun = f"baby {noun}"
                
            char_outfit = wardrobe_manager.get_outfit(char)
            action_context = "being carried in arms" if char.parent_id else ("crawling on the floor" if char.age_group == "baby" else "standing")
            
            regional_text = f"A photorealistic {char.age_group}, {char.build} build {noun.replace('baby ', '')}, {char_outfit}, {action_context}"
            telemetry_lines.append(f"- **{char.char_id}** ({char.age_group}/{char.gender} | Build: {char.build}): `{regional_text}`")
            
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
    "DW_DynamicPoseComposer": "DW Dynamic Pose Composer SOTA"
}