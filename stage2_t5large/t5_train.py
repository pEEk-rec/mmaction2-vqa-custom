"""
Stage 2: T5-Large Fine-tuning for Video Quality Description
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    # Paths
    DATA_DIR = r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\stage2_t5large"  # Current directory
    OUTPUT_DIR = r"D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\stage2_t5large\outputs"
    
    # Model
    T5_MODEL = "t5-large"
    FUSION_DIM = 512
    T5_HIDDEN_DIM = 1024
    
    # LoRA
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 4
    EPOCHS = 20
    LR = 5e-4
    MAX_INPUT_LENGTH = 256
    MAX_TARGET_LENGTH = 256
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
print(f"🚀 Device: {config.DEVICE}")

# ============================================================================
# DATASET
# ============================================================================
class VideoQualityDataset(Dataset):
    def __init__(self, fusion_logits, metadata_df, tokenizer, config):
        self.fusion_logits = torch.tensor(fusion_logits, dtype=torch.float32)
        self.questions = metadata_df['question'].tolist()
        self.answers = metadata_df['answer'].tolist()
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self):
        return len(self.fusion_logits)
    
    def __getitem__(self, idx):
        fusion_feat = self.fusion_logits[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        input_encoding = self.tokenizer(
            question,
            max_length=self.config.MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            answer,
            max_length=self.config.MAX_TARGET_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'fusion_logits': fusion_feat,
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0)
        }

# ============================================================================
# MODEL
# ============================================================================
class FusionToT5Projector(nn.Module):
    def __init__(self, fusion_dim=512, t5_dim=1024):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(fusion_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, t5_dim)
        )
    
    def forward(self, fusion_logits):
        return self.projector(fusion_logits)

class Stage2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Load T5-Large
        self.t5 = T5ForConditionalGeneration.from_pretrained(config.T5_MODEL)
        self.tokenizer = T5Tokenizer.from_pretrained(config.T5_MODEL)
        
        # Add LoRA
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=["q", "v"],
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.t5 = get_peft_model(self.t5, lora_config)
        
        # Projector
        self.fusion_projector = FusionToT5Projector(config.FUSION_DIM, config.T5_HIDDEN_DIM)
        
        print("✓ T5-Large loaded with LoRA")
        self.t5.print_trainable_parameters()
    
    def forward(self, fusion_logits, input_ids, attention_mask, labels=None):
        batch_size = fusion_logits.shape[0]
        
        # Project fusion to T5 space
        fusion_embeds = self.fusion_projector(fusion_logits).unsqueeze(1)  # (batch, 1, 1024)
        
        # Get T5 text embeddings
        text_embeds = self.t5.get_input_embeddings()(input_ids)  # (batch, seq_len, 1024)
        
        # Concatenate: fusion token as prefix
        combined_embeds = torch.cat([fusion_embeds, text_embeds], dim=1)
        
        # Extend attention mask
        fusion_attention = torch.ones(batch_size, 1, device=attention_mask.device)
        extended_attention_mask = torch.cat([fusion_attention, attention_mask], dim=1)
        
        # Forward through T5
        outputs = self.t5(
            inputs_embeds=combined_embeds,
            attention_mask=extended_attention_mask,
            labels=labels
        )
        
        return outputs

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        fusion_logits = batch['fusion_logits'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(fusion_logits, input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            fusion_logits = batch['fusion_logits'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(fusion_logits, input_ids, attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("STAGE 2: T5-LARGE TRAINING")
    print("="*80 + "\n")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.T5_MODEL)
    
    # Load data
    print("📂 Loading data...")
    metadata = pd.read_csv(f"{config.DATA_DIR}/demo_metadata.csv")
    train_logits = np.load(f"{config.DATA_DIR}/video_only_logits_512d_train.npy")
    val_logits = np.load(f"{config.DATA_DIR}/video_only_logits_512d_val.npy")
    
    train_meta = metadata.iloc[:1600]
    val_meta = metadata.iloc[1600:1800]
    
    print(f"  ✓ Train: {len(train_logits)}")
    print(f"  ✓ Val: {len(val_logits)}")
    
    # Create datasets
    train_dataset = VideoQualityDataset(train_logits, train_meta, tokenizer, config)
    val_dataset = VideoQualityDataset(val_logits, val_meta, tokenizer, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Model
    print("\n🔧 Initializing model...")
    model = Stage2Model(config).to(config.DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    
    # Training loop
    print("\n🏋️ Training...\n")
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, config.DEVICE)
        val_loss = validate(model, val_loader, config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/stage2_best.pt")
            print(f"💾 Saved (val_loss: {val_loss:.4f})")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()