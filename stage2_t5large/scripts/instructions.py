"""
Step 3: Test combining embeddings + instruction text
"""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

print("="*60)
print("TEST 3: Embeddings + Instruction Prompt")
print("="*60)

# Load model
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
device = model.device

# Your 512-D embeddings
batch_size = 2
dummy_embeddings = torch.randn(batch_size, 512, dtype=torch.float16).to(device)

# Project to 1024-D
projection = torch.nn.Linear(512, 1024).to(device).to(torch.float16)
projected = projection(dummy_embeddings).unsqueeze(1)  # [2, 1, 1024]

# Create instruction prompt
instruction = "Describe the video quality in detail: "
instruction_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(device)
instruction_embeds = model.encoder.embed_tokens(instruction_ids)  # [1, seq_len, 1024]

# Combine: [video embedding] + [instruction tokens]
combined_embeds = torch.cat([
    projected,  # [2, 1, 1024]
    instruction_embeds.expand(batch_size, -1, -1)  # [2, seq_len, 1024]
], dim=1)

print(f"Video embedding: {projected.shape}")
print(f"Instruction embedding: {instruction_embeds.shape}")
print(f"Combined: {combined_embeds.shape}")

# Generate
with torch.no_grad():
    outputs = model.generate(
        inputs_embeds=combined_embeds,
        max_length=50,
        num_beams=3
    )

print(f"\nGenerated:")
for i, text in enumerate(tokenizer.batch_decode(outputs, skip_special_tokens=True)):
    print(f"  {i+1}. {text}")

print(f"\n✅ Instruction-guided generation works!")
