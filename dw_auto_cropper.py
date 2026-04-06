import torch
import torch.nn.functional as F
import nodes

class DW_AutoFaceCropper:
    """
    O(N) Batch Looper for Face Detection and Native Tensor Cropping.
    Bypasses Impact Pack's volatile image conversion nodes by slicing the 
    tensors mathematically and interpolating them to a uniform 512x512 batch.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "crop_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("FACE_IMAGE_BATCH",)
    FUNCTION = "batch_crop"
    CATEGORY = "DW_Nodes/Vision"

    def batch_crop(self, images, bbox_detector, crop_factor):
        BboxDetector = nodes.NODE_CLASS_MAPPINGS.get("BboxDetectorSEGS")
        
        if not BboxDetector:
            raise RuntimeError("ERROR: Impact Pack 'BboxDetectorSEGS' not found. Ensure Impact Pack is installed.")
            
        bbox_func = getattr(BboxDetector(), BboxDetector().FUNCTION)
        
        processed_crops = []
        batch_size = images.shape[0]
        target_size = 512
        
        for i in range(batch_size):
            single_img = images[i].unsqueeze(0) # [1, H, W, C]
            _, h, w, _ = single_img.shape
            
            # 1. Execute detection (crop_factor=1.0 here because we apply it mathematically later)
            segs_out = bbox_func(bbox_detector, single_img, 0.5, 0, 1.0, 10, "face")[0]
            segs_list = segs_out[1]
            
            if len(segs_list) == 0:
                print(f"[DW_AutoCropper] Warning: No face detected in frame {i}. Passing original image.")
                crop_tensor = single_img
            else:
                # 2. Extract largest face coordinates
                try:
                    # SEG object attribute
                    largest_seg = max(segs_list, key=lambda s: (s.bbox[2]-s.bbox[0]) * (s.bbox[3]-s.bbox[1]))
                    x1, y1, x2, y2 = largest_seg.bbox
                except AttributeError:
                    # Tuple fallback
                    largest_seg = max(segs_list, key=lambda s: (s[1][2]-s[1][0]) * (s[1][3]-s[1][1]))
                    x1, y1, x2, y2 = largest_seg[1]
                    
                # 3. Mathematical Crop with crop_factor
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                box_w = x2 - x1
                box_h = y2 - y1
                
                side = max(box_w, box_h) * crop_factor
                
                new_x1 = max(0, int(cx - side / 2.0))
                new_y1 = max(0, int(cy - side / 2.0))
                new_x2 = min(w, int(cx + side / 2.0))
                new_y2 = min(h, int(cy + side / 2.0))
                
                crop_tensor = single_img[:, new_y1:new_y2, new_x1:new_x2, :]
            
            # 4. Standardize dimensions for batching [1, 512, 512, 3]
            crop_c = crop_tensor.permute(0, 3, 1, 2) # [1, C, H, W]
            resized_c = F.interpolate(crop_c, size=(target_size, target_size), mode='bicubic', align_corners=False)
            processed_crops.append(resized_c.permute(0, 2, 3, 1)) # [1, target_size, target_size, C]
            
        # 5. Assemble final tensor batch
        final_batch = torch.cat(processed_crops, dim=0) # [N, 512, 512, 3]
        
        return (final_batch,)

NODE_CLASS_MAPPINGS = {"DW_AutoFaceCropper": DW_AutoFaceCropper}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_AutoFaceCropper": "DW Auto Face Cropper"}