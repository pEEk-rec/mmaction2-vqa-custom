"""
Step 1: Verify T5-Large loads and runs on your GPU
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

print("="*60)
print("TEST 1: Loading FLAN-T5-Large")
print("="*60)

# Check CUDA
print(f"\n[GPU Check]")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load model
print(f"\n[Loading Model]")
print("Downloading FLAN-T5-Large (first time ~3GB)...")
start = time.time()

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.float16,  # Use FP16 to save VRAM
    device_map="auto"           # Automatic device placement
)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

print(f"✓ Loaded in {time.time()-start:.1f}s")
print(f"✓ Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
print(f"✓ Hidden dimension: {model.config.d_model}")
print(f"✓ Device: {model.device}")

# Test generation
print(f"\n[Testing Generation]")
test_prompt = "Describe the quality of a video: "
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=2
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {test_prompt}")
print(f"Output: {generated_text}")

# Check VRAM usage
if torch.cuda.is_available():
    print(f"\n[VRAM Usage]")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")

print(f"\n{'='*60}")
print(" TEST PASSED - T5-Large is ready!")
print("="*60)
