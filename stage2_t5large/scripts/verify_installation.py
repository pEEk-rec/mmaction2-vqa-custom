"""
Verify Transformers is installed correctly with T5 support
"""
import transformers
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

print("="*60)
print("Verifying Installation")
print("="*60)

print(f"\n[Package Versions]")
print(f"  Transformers: {transformers.__version__}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

print(f"\n[Testing T5 Import]")
try:
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")  # Small for quick test
    print(f"  ✓ T5Tokenizer works")
    
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    print(f"  ✓ T5ForConditionalGeneration works")
    
    print(f"\n Installation verified - ready for T5-Large!")
except Exception as e:
    print(f"   Error: {e}")

print("="*60)
