import torch
import nodes

class DW_AutoFaceCropper:
    """
    O(N) Batch Looper for Impact Pack BBox Detector.
    Bypasses the single-image limitation by programmatically iterating over a batch,
    detecting the largest face, applying the relative crop factor, and returning an aligned batch.
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
        SEGSToImage = nodes.NODE_CLASS_MAPPINGS.get("ImpactSEGSToImageBatch")
        
        if not BboxDetector or not SEGSToImage:
            raise RuntimeError("ERROR: Impact Pack nodes not found. Ensure it is properly installed.")
            
        bbox_func = getattr(BboxDetector(), BboxDetector().FUNCTION)
        seg_img_func = getattr(SEGSToImage(), SEGSToImage().FUNCTION)
        
        cropped_faces = []
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            single_img = images[i].unsqueeze(0)
            
            # Execute detection for the single image using the defined crop_factor
            # Parameters: detector, image, threshold=0.5, dilation=0, crop_factor, drop_size=10, labels="face"
            segs_out = bbox_func(bbox_detector, single_img, 0.5, 0, crop_factor, 10, "face")[0]
            
            # Fallback protection against detection failure
            if len(segs_out[1]) == 0:
                print(f"[DW_AutoCropper] Warning: No face detected in frame {i}. Passing original image.")
                cropped_faces.append(single_img)
                continue
                
            # Extract only the main face (largest bounding box) to ignore background faces
            segs_list = segs_out[1]
            largest_seg = max(segs_list, key=lambda s: (s.bbox[2]-s.bbox[0]) * (s.bbox[3]-s.bbox[1]))
            single_seg_out = (segs_out[0], [largest_seg])
            
            # Convert the cropped bounding box back to an Image tensor
            face_img = seg_img_func(single_seg_out)[0]
            cropped_faces.append(face_img)
            
        # No resizing needed here. DW_IdentityMultiplexer handles pad_to_512 internally.
        return (torch.cat(cropped_faces, dim=0),)

NODE_CLASS_MAPPINGS = {"DW_AutoFaceCropper": DW_AutoFaceCropper}
NODE_DISPLAY_NAME_MAPPINGS = {"DW_AutoFaceCropper": "DW Auto Face Cropper"}