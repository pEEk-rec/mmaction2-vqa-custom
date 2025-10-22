# # import sys
# # sys.path.insert(0, r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1')

# # from mmaction.registry import MODELS
# # import mmaction.models.data_preprocessors.data_preprocessor

# # print("ActionDataPreprocessor registered:", 
# #       'ActionDataPreprocessor' in MODELS.module_dict)
# # print("\nAll registered preprocessors:")
# # for key in MODELS.module_dict.keys():
# #     if 'preprocess' in key.lower():
# #         print(f"  - {key}")

# import sys
# sys.path.insert(0, r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1')

# from mmengine import Config

# # Load your config
# cfg = Config.fromfile(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\mmaction\configs\recognition\swin\VQA_swinConfigPipeline.py')

# # Check if custom_imports is present
# print("custom_imports in config:", hasattr(cfg, 'custom_imports'))
# if hasattr(cfg, 'custom_imports'):
#     print("Imports:", cfg.custom_imports)

# # Check registry after config load
# from mmaction.registry import MODELS
# print("\nActionDataPreprocessor registered after config load:", 
#       'ActionDataPreprocessor' in MODELS.module_dict)

# # Try to build the model
# print("\nAttempting to build model...")
# try:
#     model = MODELS.build(cfg.model)
#     print("✓ Model built successfully!")
#     print(f"  Model type: {type(model).__name__}")
#     print(f"  Data preprocessor type: {type(model.data_preprocessor).__name__}")
# except Exception as e:
#     print(f"✗ Error building model: {e}")

import sys
sys.path.insert(0, r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1')

from mmengine import Config
from mmaction.registry import MODELS

# Load config
cfg = Config.fromfile(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\mmaction\configs\recognition\swin\VQA_swinConfigPipeline.py')

print("1. After loading config:")
print("   ActionDataPreprocessor registered:", 'ActionDataPreprocessor' in MODELS.module_dict)

# Manually process custom_imports (MMEngine should do this, but let's be explicit)
if hasattr(cfg, 'custom_imports'):
    import importlib
    for module_name in cfg.custom_imports['imports']:
        print(f"   Importing: {module_name}")
        try:
            importlib.import_module(module_name)
            print(f"     ✓ Success")
        except Exception as e:
            print(f"     ✗ Failed: {e}")

print("\n2. After manual imports:")
print("   ActionDataPreprocessor registered:", 'ActionDataPreprocessor' in MODELS.module_dict)

# Try building model
print("\n3. Attempting to build model...")
try:
    model = MODELS.build(cfg.model)
    print("   ✓ Model built successfully!")
except Exception as e:
    print(f"   ✗ Error: {e}")