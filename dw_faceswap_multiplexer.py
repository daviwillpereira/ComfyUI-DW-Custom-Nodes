import torch
import nodes

class DW_FaceSwapMultiplexer:
    """
    O(N) Daisy-Chain Engine for ReActor.
    Programmatically iterates over a reference image batch and applies sequential face swapping
    on a single high-resolution canvas, strictly mapping left-to-right spatial indexes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled_canvas": ("IMAGE",),
                "reference_image_batch": ("IMAGE",),
                "face_model": (["inswapper_128.onnx"], {"default": "inswapper_128.onnx"}),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n", "yolov8n"], {"default": "retinaface_resnet50"}),
                "face_restore_model": (["none", "GPEN-BFR-512.onnx", "GFPGANv1.4.pth", "CodeFormer.pth"], {"default": "none"}),
                "face_restore_visibility": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("SWAPPED_CANVAS", "TELEMETRY_REPORT")
    FUNCTION = "multiplex_faceswap"
    CATEGORY = "DW_Nodes/FaceSwap"

    def multiplex_faceswap(self, upscaled_canvas, reference_image_batch, face_model, facedetection, face_restore_model, face_restore_visibility, codeformer_weight):
        
        # 1. Dependency Injection (Fetching ReActor Native Node)
        ReActorNode = nodes.NODE_CLASS_MAPPINGS.get("ReActorFaceSwap")
        if not ReActorNode:
            raise RuntimeError("ERROR: ReActor custom node not found. Ensure 'comfyui-reactor-node' is properly installed.")

        reactor = ReActorNode()
        reactor_func = getattr(reactor, reactor.FUNCTION)

        batch_size = reference_image_batch.shape[0]
        current_canvas = upscaled_canvas
        
        telemetry = [
            "# 🎭 HARD LIKENESS MULTIPLEXER REPORT", 
            "---",
            f"**Subjects Detected:** `{batch_size}`",
            f"**Face Model:** `{face_model}`",
            f"**Restoration:** `{face_restore_model}` (Vis: {face_restore_visibility})",
            "---",
            "### 🔄 DAISY-CHAIN SWAP LOG"
        ]

        # 2. O(N) Programmatic Daisy-Chain Loop
        for i in range(batch_size):
            ref_face = reference_image_batch[i].unsqueeze(0)
            
            try:
                # Dynamic index mapping: Subject 'i' matches Face 'i' (Left-to-Right layout logic)
                res = reactor_func(
                    enabled=True,
                    input_image=current_canvas,
                    source_image=ref_face,
                    swap_model=face_model,
                    facedetection=facedetection,
                    face_restore_model=face_restore_model,
                    face_restore_visibility=face_restore_visibility,
                    codeformer_weight=codeformer_weight,
                    detect_gender_input="no",
                    detect_gender_source="no",
                    input_faces_index=str(i),
                    source_faces_index="0", 
                    console_log_level=1
                )
                
                # Overwrite the canvas for the next iteration (Daisy-Chain logic)
                current_canvas = res[0]
                telemetry.append(f"- **Subject {i}**: ✅ Face Swapped (Spatial Index: {i})")
                
            except Exception as e:
                telemetry.append(f"- **Subject {i}**: ❌ ERROR ({str(e)})")
                
        return (current_canvas, "\n".join(telemetry))

# --- REGISTRATION ---
NODE_CLASS_MAPPINGS = {
    "DW_FaceSwapMultiplexer": DW_FaceSwapMultiplexer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DW_FaceSwapMultiplexer": "DW FaceSwap Multiplexer"
}