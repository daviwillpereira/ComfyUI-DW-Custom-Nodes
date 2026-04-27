import os
import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

print("[DW Custom Nodes] Initializing Domain-Driven Module Discovery...")

for dirpath, dirnames, filenames in os.walk(ROOT_PATH):
    # Ignore cache and hidden directories
    if "__pycache__" in dirpath or os.path.basename(dirpath).startswith("."):
        continue

    for filename in filenames:
        if filename.endswith(".py") and filename != "__init__.py":
            rel_path = os.path.relpath(os.path.join(dirpath, filename), ROOT_PATH)
            # Convert file path to python module notation (e.g., universal.vlm.dw_qwen_batch)
            module_name = rel_path.replace(os.sep, ".")[:-3]

            try:
                module = importlib.import_module(f".{module_name}", package=__name__)
                
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"[DW Custom Nodes] Failed to load module {module_name}: {e}")
                traceback.print_exc()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']