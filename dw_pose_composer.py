import json
import torch
import numpy as np
import cv2
import random
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
    camera_elevation: float = 0.0 # FIX: Perspective anchor
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

# Official OpenPose RGB Color Table
POSE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), 
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), 
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), 
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), 
    (255, 0, 170), (255, 0, 85)
]

POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

# ==========================================
# 2. Biometric Engine & Factory
# ==========================================
class BiometricFactory:
    @staticmethod
    def get_metrics(char: CharacterDTO, base_height: int, camera_elevation: float = 0.0) -> Dict[str, float]:
        female_mod = 1.05 if char.gender == "female" else 1.0
        height_map = {"elder": 0.95, "adult": 1.0, "teenager": 0.85, "child": 0.6, "baby": 0.35}
        head_map = {"elder": 8.0, "adult": 8.0, "teenager": 7.0, "child": 6.0, "baby": 4.0 * female_mod}
        width_map = {"slim": 0.85, "regular": 1.0, "muscular": 1.15, "heavy": 1.25}
        
        rel_h = height_map.get(char.age_group, 1.0)
        h_ratio = head_map.get(char.age_group, 8.0)
        w_mod = width_map.get(char.build, 1.0)
        
        # FIX: Foreshortening (Perspective Distortion)
        perspective_head_mod = 1.0 + (camera_elevation * 0.15)
        perspective_leg_mod = 1.0 - (camera_elevation * 0.15)
        
        total_pixels = int(base_height * rel_h * (1.0 - abs(camera_elevation) * 0.05))
        head_pixels = int((total_pixels / h_ratio) * perspective_head_mod)
        
        shoulder_width = int(head_pixels * 1.55 * w_mod)
        hip_width = int(shoulder_width * 0.70)
        arm_len = int(total_pixels * 0.4)
        leg_len = int(total_pixels * 0.5 * perspective_leg_mod)

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
            
        if char.parent_id and parent_kps:
            is_left_hip = getattr(char, "_is_left_hip", True)
            hip_idx = 11 if is_left_hip else 8
            adult_hip = parent_kps.get(hip_idx, (anchor_x, anchor_y))
            adult_neck = parent_kps.get(1, (anchor_x, anchor_y - int(th*0.8)))
            
            waist_y = adult_hip[1] - int((adult_hip[1] - adult_neck[1]) * 0.3)
            baby_torso_len = int(th * 0.39)
            
            lean_direction = -1 if is_left_hip else 1 
            lean_offset = int(hw * 0.3) * lean_direction
            
            kps[1] = (adult_hip[0] + lean_offset, waist_y - baby_torso_len)
            kps[0] = (kps[1][0], kps[1][1] - hh)
            
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
        
        face_offset_x = int(hh * 0.2)
        face_offset_y = int(hh * 0.2)
        kps[14] = (kps[0][0] - face_offset_x, int(kps[0][1] - face_offset_y - head_tilt))
        kps[15] = (kps[0][0] + face_offset_x, int(kps[0][1] - face_offset_y + head_tilt))
        ear_offset_x = int(hh * 0.4)
        kps[16] = (kps[0][0] - ear_offset_x, int(kps[0][1] - head_tilt))
        kps[17] = (kps[0][0] + ear_offset_x, int(kps[0][1] + head_tilt))
        
        shoulder_drop = int(hh * 0.05) 
        if char.parent_id:
            is_left_hip = getattr(char, "_is_left_hip", True)
            if is_left_hip:
                kps[5] = (kps[1][0], kps[1][1] + shoulder_drop) 
            else:
                kps[2] = (kps[1][0], kps[1][1] + shoulder_drop) 
        else:
            kps[2] = (kps[1][0] - sw//2, kps[1][1] + shoulder_drop + int(shoulder_tilt)) 
            kps[5] = (kps[1][0] + sw//2, kps[1][1] + shoulder_drop - int(shoulder_tilt))
        
        if char.parent_id:
            is_left_hip = getattr(char, "_is_left_hip", True)
            wrap_direction = -1 if is_left_hip else 1
            arm_reach_x = int(arm_l * 0.4) * wrap_direction
            
            if is_left_hip:
                kps[6] = (kps[5][0] + arm_reach_x, kps[5][1] + int(arm_l*0.2))
                kps[7] = (kps[6][0] + int(arm_l*0.2)*wrap_direction, kps[6][1] - int(arm_l*0.1)) 
            else:
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
            arm_swing_r = rng_context.uniform(0.05, 0.08) * arm_l
            kps[3] = (kps[2][0] - int(arm_swing_r), kps[2][1] + int(arm_l*0.5))
            kps[4] = (kps[3][0] - int(arm_l*0.05), kps[3][1] + int(arm_l*0.5))
            arm_swing_l = rng_context.uniform(0.05, 0.08) * arm_l
            kps[6] = (kps[5][0] + int(arm_swing_l), kps[5][1] + int(arm_l*0.5))
            kps[7] = (kps[6][0] + int(arm_l*0.05), kps[6][1] + int(arm_l*0.5))
        
        if char.parent_id:
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
    def __init__(self, scene: SceneDTO):
        self.scene = scene
        self.blur_kernel = (11, 11)

    def draw_pose_and_masks(self, characters: List[CharacterDTO]) -> Tuple[np.ndarray, torch.Tensor]:
        canvas = np.zeros((self.scene.height, self.scene.width, 3), dtype=np.uint8)
        global_z_buffer = np.zeros((self.scene.height, self.scene.width), dtype=np.uint8)
        masks = {}

        sorted_chars_masks = sorted(characters, key=lambda c: c.z_index, reverse=True)
        
        for char in sorted_chars_masks:
            char_mask = np.zeros((self.scene.height, self.scene.width), dtype=np.uint8)
            
            base_thickness = 50 if char.build in ["heavy", "muscular"] else 35
            if char.age_group in ["child", "baby"]:
                base_thickness = int(base_thickness * 0.7)

            if all(k in char.keypoints for k in [1, 2, 5, 8, 11]):
                torso_pts = np.array([
                    char.keypoints[2], 
                    char.keypoints[5], 
                    char.keypoints[11],
                    char.keypoints[8], 
                ], np.int32)
                cv2.fillPoly(char_mask, [torso_pts], 255)
                
            # FIX: Pelvic block to prevent Depth Piercing (Thigh Gap closure)
            if all(k in char.keypoints for k in [8, 11]):
                groin_x = (char.keypoints[8][0] + char.keypoints[11][0]) // 2
                groin_y = max(char.keypoints[8][1], char.keypoints[11][1]) + int(base_thickness * 1.8)
                pelvis_pts = np.array([
                    char.keypoints[8],
                    char.keypoints[11],
                    (groin_x, groin_y)
                ], np.int32)
                cv2.fillPoly(char_mask, [pelvis_pts], 255)
            
            for pair in POSE_PAIRS:
                if pair[0] in char.keypoints and pair[1] in char.keypoints:
                    pt1 = char.keypoints[pair[0]]
                    pt2 = char.keypoints[pair[1]]
                    
                    # FIX: Dynamic anatomical thickness
                    current_thickness = base_thickness
                    if pair in [(8, 9), (11, 12)]: # Thighs
                        current_thickness = int(base_thickness * 1.5)
                    elif pair in [(9, 10), (12, 13)]: # Calves
                        current_thickness = int(base_thickness * 1.2)
                    elif pair in [(2, 3), (5, 6), (3, 4), (6, 7)]: # Arms
                        current_thickness = int(base_thickness * 0.8)
                        
                    cv2.line(char_mask, pt1, pt2, 255, thickness=current_thickness, lineType=cv2.LINE_AA)
            
            if 0 in char.keypoints:
                head_radius = 85 if char.gender == "female" else 75
                if char.age_group in ["child", "baby"]:
                    head_radius = int(head_radius * 0.7)
                cv2.circle(char_mask, char.keypoints[0], head_radius, 255, thickness=-1, lineType=cv2.LINE_AA)

            blurred_char_mask = cv2.GaussianBlur(char_mask, self.blur_kernel, 0)
            char_mask_exclusive = cv2.bitwise_and(blurred_char_mask, cv2.bitwise_not(global_z_buffer))
            masks[char.char_id] = char_mask_exclusive
            
            global_z_buffer = cv2.bitwise_or(global_z_buffer, char_mask)

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

        mask_list = [torch.from_numpy(masks[c.char_id]).float() / 255.0 for c in self.scene.original_characters]
        masks_tensor = torch.stack(mask_list) if mask_list else torch.empty((0, self.scene.height, self.scene.width))
        
        bg_mask = cv2.bitwise_not(global_z_buffer)
        bg_mask_tensor = torch.from_numpy(bg_mask).float() / 255.0
        
        return canvas, masks_tensor, bg_mask_tensor

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
                "vision_context": ("STRING", {"forceInput": True, "multiline": True}),
                "clip": ("CLIP",),
                "global_positive": ("STRING", {"multiline": True, "forceInput": True}), # SOTA FIX: Now receives the raw BG JSON
                "global_negative": ("STRING", {"multiline": True, "default": "deformed, bad anatomy"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("POSE_CANVAS", "Z_BUFFER_MASKS", "COMBINED_MASK", "TELEMETRY_REPORT", "POSITIVE", "NEGATIVE", "PHENOTYPES")
    FUNCTION = "generate_rigs"
    CATEGORY = "DW_Nodes/Pose"

    def generate_rigs(self, width: int, height: int, vision_context: str, clip, global_positive: str, global_negative: str):
        import json
        
        # FIX: Dynamic Spatial Parsing from Background Qwen JSON
        floor_y_percent, global_scale, camera_elevation = 0.85, 0.70, 0.0
        parsed_global_positive = global_positive
        
        try:
            bg_data = json.loads(global_positive)
            if isinstance(bg_data, list) and len(bg_data) > 0 and isinstance(bg_data[0], dict):
                bg_dict = bg_data[0]
                floor_y_percent = float(bg_dict.get("floor_y_percent", 0.85))
                global_scale = float(bg_dict.get("global_scale", 0.70))
                camera_elevation = float(bg_dict.get("camera_elevation", 0.0))
                
                parts = []
                for key in ["location_name", "architecture_style", "ground_material", "lighting_conditions", "atmosphere", "camera_properties"]:
                    if bg_dict.get(key): parts.append(bg_dict[key])
                parsed_global_positive = ", ".join([str(p).strip() for p in parts if str(p).strip()])
        except Exception:
            pass # Fallback to raw string if parsing fails
        
        try:
            vision_data_list = json.loads(vision_context)
            if not isinstance(vision_data_list, list): vision_data_list = [vision_data_list]
        except Exception as e:
            raise ValueError(f"[DW_ERROR] vision_context is not a valid JSON array. Error: {e}")
            
        num_characters = len(vision_data_list)
        if num_characters == 0: raise ValueError("[DW] Critical Error: vision_context is empty.")

        rng_seed = random.randint(0, 9999999)
        rng_context = random.Random(rng_seed)

        scene = SceneDTO(
            width=width, height=height,
            floor_y_percent=floor_y_percent,
            global_scale=global_scale,
            camera_elevation=camera_elevation,
            seed=rng_seed
        )

        characters = []
        phenotypes_list = []
        parsed_chars_data = []
        
        for i in range(num_characters):
            raw = vision_data_list[i]
            
            # FIX: Semantic Outfit Sanitizer to intercept VLM failures
            def get_valid_outfit(val, default_val):
                s = str(val or "").lower()
                if not s or s == "null" or "not visible" in s or "unknown" in s or "none" in s:
                    return default_val
                return str(val)
                
            outfit_upper = get_valid_outfit(raw.get("outfit_upper"), "neutral shirt")
            outfit_lower = get_valid_outfit(raw.get("outfit_lower"), "jeans")
            outfit_footwear = get_valid_outfit(raw.get("outfit_footwear"), "sneakers")
            
            combined_outfit = f"{outfit_upper}, {outfit_lower}, {outfit_footwear}"
            
            mapped = {
                "gender": str(raw.get("gender") or "male").lower(),
                "age_group": str(raw.get("age_group") or "adult").lower(),
                "exact_age": str(raw.get("exact_age") or "30 years old").lower(),
                "build_cat": str(raw.get("physical_build") or raw.get("build_cat") or "regular").lower(),
                "exact_build": str(raw.get("exact_build") or "average build").lower(),
                "skin": str(raw.get("skin_tone") or raw.get("skin") or "natural skin").lower(),
                "hair_length": str(raw.get("hair_length") or "").lower(),
                "hair_volume": str(raw.get("hair_volume") or "").lower(),
                "hair_color": str(raw.get("hair_color") or "dark").lower(),
                "hair_texture": str(raw.get("hair_texture") or "straight").lower(),
                "eyes": str(raw.get("eyes") or "brown eyes").lower(),
                "beard": str(raw.get("beard_style_and_color") or raw.get("beard") or "no beard").lower(),
                "glasses": str(raw.get("glasses") or "no glasses").lower(),
                "outfit": str(combined_outfit or "modern stylish casual clothes").lower()
            }
            
            if mapped["age_group"] not in ["baby", "child", "teenager", "adult", "elder"]: mapped["age_group"] = "adult"
            if mapped["build_cat"] not in ["slim", "regular", "heavy", "muscular"]: mapped["build_cat"] = "regular"
            parsed_chars_data.append(mapped)

        available_adults = [f"person_{i}" for i, v in enumerate(parsed_chars_data) if v["age_group"] in ["adult", "elder"]]
        used_adults = set()
        
        for i, v_data in enumerate(parsed_chars_data):
            char = CharacterDTO(char_id=f"person_{i}", gender=v_data["gender"], age_group=v_data["age_group"], build=v_data["build_cat"])
            
            hair_len_vol = f"{v_data.get('hair_length', '')} {v_data.get('hair_volume', '')}".strip()
            hair_color_tex = f"{v_data.get('hair_texture', '')} {v_data.get('hair_color', '')}".strip()
            final_hair = "bald" if 'bald' in hair_color_tex or 'bald' in hair_len_vol else f"{hair_len_vol} {hair_color_tex} hair".strip()
            
            glasses = ", wearing glasses" if "no" not in v_data.get("glasses", "no") else ""
            beard = f", {v_data.get('beard', '')}" if "no" not in v_data.get('beard', 'no') else ""
            traits = f"{v_data.get('skin', 'natural skin')}, {final_hair}, {v_data.get('eyes', 'eyes')}{beard}{glasses}"
                
            phenotype_line = f"{v_data.get('exact_age', '')} {v_data.get('build_cat', '')} {v_data.get('exact_build', '')} {char.gender}, {traits}"
            phenotypes_list.append(" ".join(phenotype_line.split()))
            
            if char.age_group == "baby":
                free_adults = [a for a in available_adults if a not in used_adults]
                if free_adults:
                    char.z_index = 2; char.parent_id = free_adults[0]; used_adults.add(free_adults[0])
                else: char.parent_id = None; char.z_index = 1
                
            characters.append(char)
            
        phenotypes_output = "\n".join(phenotypes_list)

        for char in characters:
            if char.parent_id:
                parent_char = next((c for c in characters if c.char_id == char.parent_id), None)
                if parent_char:
                    parent_char.is_holding_baby = True; parent_char._holding_baby_on_left = rng_context.choice([True, False])
                    char._is_left_hip = parent_char._holding_baby_on_left

        scene.original_characters = list(characters)
        processing_order = sorted(characters, key=lambda c: c.parent_id is not None)
        base_h = int(height * 0.7 * scene.global_scale)
        floor_y = int(height * scene.floor_y_percent)
        
        for char in characters:
            metrics = BiometricFactory.get_metrics(char, base_h, camera_elevation)
            if char.parent_id: char.z_index = 99999 
            elif char.is_holding_baby: char.z_index = 50000 
            else: char.z_index = int(10000 / metrics["total_h"])

        independent_chars = [c for c in processing_order if not c.parent_id]
        dependent_chars = [c for c in processing_order if c.parent_id]
        
        char_metrics = []
        cluster_total_width = 0
        
        for char in independent_chars:
            metrics = BiometricFactory.get_metrics(char, base_h, camera_elevation)
            spacing_factor = 0.75 if char.age_group in ["adult", "elder", "teenager"] else 0.40
            personal_space = int(metrics["shoulder_w"] * spacing_factor)
            padded_width = metrics["shoulder_w"] + personal_space
            char_metrics.append((char, metrics, padded_width))
            cluster_total_width += padded_width
            
        start_x = (width - cluster_total_width) // 2 if cluster_total_width < width else 50
        x_current = start_x
        adult_anchors = {} 
        
        for i, (char, metrics, padded_width) in enumerate(char_metrics):
            char_center_x = x_current + (padded_width // 2)
            y_jitter = int(metrics["head_h"] * 0.08) if i % 2 == 0 else 0
            char.keypoints = BiometricFactory.build_skeleton(char, char_center_x, floor_y + y_jitter, metrics, rng_context)
            adult_anchors[char.char_id] = char.keypoints 
            x_current += padded_width

        for char in dependent_chars:
            metrics = BiometricFactory.get_metrics(char, base_h)
            parent_kps = adult_anchors.get(char.parent_id, None)
            char.keypoints = BiometricFactory.build_skeleton(char, 0, 0, metrics, rng_context, parent_kps=parent_kps)

        # FIX: Dynamic Y-Axis Bounding Box Auto-Shifter
        # Prevents headless skeletons if Qwen hallucinates extreme floor_y or global_scale
        min_y = float('inf')
        for char in characters:
            for k, pt in char.keypoints.items():
                if pt[1] < min_y: min_y = pt[1]
                
        safe_margin = 80
        if min_y < safe_margin:
            shift_y = safe_margin - min_y
            for char in characters:
                for k, pt in char.keypoints.items():
                    char.keypoints[k] = (pt[0], pt[1] + int(shift_y))

        renderer = OpenPoseRenderer(scene)
        pose_canvas_np, masks_tensor, bg_mask_tensor = renderer.draw_pose_and_masks(characters)
        pose_canvas_tensor = torch.from_numpy(pose_canvas_np).float() / 255.0
        pose_canvas_tensor = pose_canvas_tensor.unsqueeze(0)

        people = []
        for char in characters:
            pose_2d = []
            for i in range(18):
                if i in char.keypoints: pose_2d.extend([float(char.keypoints[i][0]), float(char.keypoints[i][1]), 1.0])
                else: pose_2d.extend([0.0, 0.0, 0.0])
            people.append({"pose_keypoints_2d": pose_2d, "face_keypoints_2d": None, "hand_left_keypoints_2d": None, "hand_right_keypoints_2d": None})
            
        pose_keypoint_str = json.dumps([{"people": people, "canvas_height": height, "canvas_width": width}])

        clip_encoder = nodes.CLIPTextEncode()
        cond_set_mask = nodes.ConditioningSetMask()
        cond_combine = nodes.ConditioningCombine()

        global_pos_cond_raw, = clip_encoder.encode(clip, parsed_global_positive)
        global_neg_cond, = clip_encoder.encode(clip, global_negative)
        
        bg_mask_tensor_unsqueeze = bg_mask_tensor.unsqueeze(0)
        final_pos_cond, = cond_set_mask.append(global_pos_cond_raw, bg_mask_tensor_unsqueeze, "default", 1.0)

        telemetry_lines = [
            "# 📊 TELEMETRY REPORT", f"**Seed:** `{rng_seed}`", "---",
            "### 🧠 VLM RAW PAYLOADS",
            f"**BG Qwen:** `{global_positive}`",
            f"**Chars Qwen:** `{vision_context}`", "---",
            "### 🌍 GLOBAL PROMPTS",
            f"**Spatial Engine:** `Scale: {global_scale} | Floor Y: {floor_y_percent} | Cam Elev: {camera_elevation}`",
            f"**Positive:** `{parsed_global_positive}`",
            f"**Negative:** `{global_negative}`", "---", "### 👤 REGIONAL PROMPTS (Multiplexed)"
        ]

        for i, char in enumerate(characters):
            noun = "person"
            if char.gender == "male": noun = "man" if char.age_group in ["adult", "elder"] else "boy"
            else: noun = "woman" if char.age_group in ["adult", "elder"] else "girl"
            if char.age_group == "baby": noun = "baby"
                
            action_context = "being carried in arms" if char.parent_id else ("crawling on the floor" if char.age_group == "baby" else "standing, dynamic cinematic pose")
            v_data = parsed_chars_data[i] if i < len(parsed_chars_data) else None
            
            if v_data:
                skin = v_data.get('skin', 'natural'); skin += " skin" if "skin" not in skin else ""
                eyes = v_data.get('eyes', 'brown'); eyes += " eyes" if "eyes" not in eyes else ""
                hair_len, hair_vol, hair_tex, hair_color = v_data.get('hair_length', ''), v_data.get('hair_volume', ''), v_data.get('hair_texture', ''), v_data.get('hair_color', '')
                
                final_hair = "bald head" if 'bald' in hair_len or 'bald' in hair_tex or 'bald' in hair_color else f"{hair_len} {hair_vol} {hair_tex} {hair_color} hair"
                final_hair = " ".join(final_hair.split())
                
                beard = v_data.get('beard', 'no beard')
                beard_str = f"with a {beard}" if "no" not in beard else "clean-shaven"
                glasses = v_data.get('glasses', 'no glasses')
                glasses_str = f"wearing {glasses}" if "no" not in glasses else ""

                traits = f"having {skin}, {eyes}, and {final_hair}, {beard_str}"
                if glasses_str: traits += f", {glasses_str}"
                
                exact_age, build_cat, exact_build, outfit = v_data.get("exact_age", "adult"), v_data.get("build_cat", "regular"), v_data.get("exact_build", "average build"), v_data.get("outfit", "")
            else:
                traits, exact_age, build_cat, exact_build, outfit = "having natural skin, brown eyes, detailed face", "adult", "regular", "regular build", "stylish casual clothes"

            clean_outfit = outfit.replace("wearing ", "").strip()
            perspective_shot = "eye-level shot"
            if camera_elevation > 0.3: perspective_shot = "high angle shot, looking down"
            elif camera_elevation < -0.3: perspective_shot = "low angle shot, looking up"
            
            regional_text = f"A photorealistic {exact_age} {build_cat} {exact_build} {noun}, {traits}, perfectly shaped symmetric ears, wearing {clean_outfit}, {action_context}, {perspective_shot}, cinematic lighting"
            regional_text = " ".join(regional_text.split())
            telemetry_lines.append(f"- **{char.char_id}**: `{regional_text}`")
            
            reg_cond, = clip_encoder.encode(clip, regional_text)
            char_mask = masks_tensor[i].unsqueeze(0)
            masked_reg_cond, = cond_set_mask.append(reg_cond, char_mask, "default", 1.0)
            final_pos_cond, = cond_combine.combine(final_pos_cond, masked_reg_cond)

        telemetry_lines.append("---")
        telemetry_lines.append("### 🦴 RAW POSE JSON")
        telemetry_lines.append("```json\n" + pose_keypoint_str + "\n```")
        
        telemetry_report_str = "\n".join(telemetry_lines)

        # FIX: Collapse the batch into a single mask and preserve [1, H, W] dimension for ImageCompositeMasked
        combined_mask = torch.clamp(torch.sum(masks_tensor, dim=0, keepdim=True), 0.0, 1.0)

        return (pose_canvas_tensor, masks_tensor, combined_mask, telemetry_report_str, final_pos_cond, global_neg_cond, phenotypes_output)
            
NODE_CLASS_MAPPINGS = {"DW_DynamicPoseComposer": DW_DynamicPoseComposer}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_DynamicPoseComposer": "DW Dynamic Pose Composer"}