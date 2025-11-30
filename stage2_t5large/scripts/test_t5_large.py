"""
Official way to load T5-Large (from Hugging Face Transformers)
Reference: https://github.com/huggingface/transformers
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

print("="*60)
print("Testing T5-Large Setup")
print("="*60)

# Check GPU
print(f"\n[1/3] GPU Check:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load T5-Large (official method)
print(f"\n[2/3] Loading FLAN-T5-Large...")
print("  (First time will download ~3GB)")

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.float16,  # Use FP16 to save VRAM
    device_map="auto"            # Automatic GPU placement
)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

print(f"  ✓ Model loaded")
print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"  ✓ Hidden dim: {model.config.d_model}")
print(f"  ✓ Device: {model.device}")

# Test generation
print(f"\n[3/3] Testing generation:")
test_input = "Describe video quality: "
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=30)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"  Input: {test_input}")
print(f"  Output: {generated}")

# VRAM check
if torch.cuda.is_available():
    print(f"\n[VRAM Usage]")
    print(f"  Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")

print(f"\n{'='*60}")
print(" T5-Large setup complete!")
print("="*60)
