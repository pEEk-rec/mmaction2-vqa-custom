"""
Step 2: Test 512-D → 1024-D projection and feeding to T5
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

print("="*60)
print("TEST 2: Projection Layer (512-D → 1024-D)")
print("="*60)

# Load T5-Large
print("\n[1/4] Loading T5-Large...")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
device = model.device
print(f"✓ Model on {device}")

# Simulate your 512-D fused embeddings
print("\n[2/4] Creating dummy 512-D embeddings (like your Stage 1 output)...")
batch_size = 2
fused_dim = 512  # Your dimension
dummy_embeddings = torch.randn(batch_size, fused_dim, dtype=torch.float16).to(device)
print(f"✓ Dummy embeddings: {dummy_embeddings.shape}")

# Create projection layer
print("\n[3/4] Creating projection layer...")
projection = torch.nn.Linear(fused_dim, model.config.d_model).to(device).to(torch.float16)
projected = projection(dummy_embeddings)  # [2, 1024]
print(f"✓ After projection: {projected.shape}")
print(f"✓ Projection params: {sum(p.numel() for p in projection.parameters())/1e3:.1f}K")

# Feed to T5
print("\n[4/4] Feeding to T5 for generation...")
# Add sequence dimension: [B, 1024] → [B, 1, 1024]
inputs_embeds = projected.unsqueeze(1)

with torch.no_grad():
    outputs = model.generate(
        inputs_embeds=inputs_embeds,
        max_length=30,
        num_beams=2,
        do_sample=False
    )

generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("✓ Generated outputs:")
for i, text in enumerate(generated_texts):
    print(f"   Sample {i+1}: {text}")

print(f"\n{'='*60}")
print(" TEST PASSED - Projection works!")
print("="*60)
print("\nKey findings:")
print(f"  • 512-D → 1024-D projection: ✓")
print(f"  • T5 accepts projected embeddings: ✓")
print(f"  • Generation from embeddings: ✓")
