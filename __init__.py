from .dw_pose_composer import NODE_CLASS_MAPPINGS as pose_mappings
from .dw_pose_composer import NODE_DISPLAY_NAME_MAPPINGS as pose_names
from .dw_identity_multiplexer import NODE_CLASS_MAPPINGS as id_mappings
from .dw_identity_multiplexer import NODE_DISPLAY_NAME_MAPPINGS as id_names
from .dw_pure_vqa import NODE_CLASS_MAPPINGS as vqa_mappings
from .dw_pure_vqa import NODE_DISPLAY_NAME_MAPPINGS as vqa_names

NODE_CLASS_MAPPINGS = {
    **pose_mappings,
    **id_mappings,
    **vqa_mappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **pose_names,
    **id_names,
    **vqa_names
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]