import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import argparse
from tqdm import tqdm
import logging
import json
from pathlib import Path
from transformers import T5Tokenizer
import matplotlib.pyplot as plt

# Import utilities from utils.py
from utils import (
    setup_logging, set_seed, load_temporal_features, load_video_annotations,
    load_question_answers, ProceduralVideoDataset, create_dataloaders,
    save_model, load_model, get_learning_rate_scheduler,
    visualize_attention, visualize_attention_timeline, plot_training_curves,
    create_experiment_dir, save_config, load_config, set_device, count_parameters
)

# Set up logging
logger = setup_logging(log_file='cross_modal_fusion.log')

class AdaptiveEarlyStopping:
    """
    Early stopping with adaptive patience.
    """
    def __init__(self, initial_patience=8, max_patience=20, patience_increase=2):
        self.patience = initial_patience
        self.initial_patience = initial_patience
        self.max_patience = max_patience
        self.patience_increase = patience_increase
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """
        Returns True if training should stop, False otherwise.
        
        Args:
            score: The validation score (lower is better)
            epoch: Current epoch number
        """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
            
    def increase_patience(self):
        """Increase patience after learning rate reduction."""
        self.patience = min(self.patience + self.patience_increase, self.max_patience)
        logger.info(f"Increased patience to {self.patience}")
        
    def state_dict(self):
        return {
            'patience': self.patience,
            'best_score': self.best_score,
            'counter': self.counter,
            'best_epoch': self.best_epoch
        }
    
    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
        self.best_epoch = state_dict['best_epoch']

# Define the cross-modal fusion model
class CrossModalFusionModel(nn.Module):
    """
    Cross-modal fusion model that aligns video temporal features with textual descriptions
    and generates prompt tokens for T5.
    """
    def __init__(self, video_dim=512, text_dim=768, hidden_dim=512, num_heads=8, 
                 num_layers=4, dropout=0.1, num_prompt_tokens=24, prompt_dim=512):
        super().__init__()
        
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_dim = prompt_dim
        
        # Project video and text features to the same dimension
        self.video_projection = nn.Linear(video_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Layer normalization for inputs
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Prompt token generation
        self.prompt_tokens = nn.Parameter(torch.randn(1, num_prompt_tokens, hidden_dim))
        self.prompt_projection = nn.Linear(hidden_dim, prompt_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' not in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'prompt_tokens' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
    
    def forward(self, video_features, text_features, video_mask=None, text_mask=None, return_attention=False):
        """
        Forward pass through the cross-modal fusion model.
        
        Args:
            video_features: Tensor of video features [batch_size, video_len, video_dim]
            text_features: Tensor of text features [batch_size, text_len, text_dim]
            video_mask: Optional mask for video features [batch_size, video_len]
            text_mask: Optional mask for text features [batch_size, text_len]
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            prompt_tokens: Generated prompt tokens [batch_size, num_prompt_tokens, prompt_dim]
            attention_weights: Optional attention weights for visualization
        """
        batch_size = video_features.shape[0]
        
        # Project features to hidden dimension
        video_features = self.video_projection(video_features)
        text_features = self.text_projection(text_features)
        
        # Apply layer normalization
        video_features = self.video_norm(video_features)
        text_features = self.text_norm(text_features)
        
        # Initialize prompt tokens for this batch
        prompt_tokens = self.prompt_tokens.expand(batch_size, -1, -1)
        
        # Store attention weights if needed
        all_attention_weights = [] if return_attention else None
        
        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            prompt_tokens, attention_weights = layer(
                prompt_tokens, video_features, text_features, 
                video_mask, text_mask, return_attention
            )
            
            if return_attention:
                all_attention_weights.append(attention_weights)
        
        # Apply final normalization
        prompt_tokens = self.final_norm(prompt_tokens)
        
        # Project to output dimension
        prompt_tokens = self.prompt_projection(prompt_tokens)
        
        if return_attention:
            # Return the last layer's attention weights for visualization
            return prompt_tokens, all_attention_weights[-1]
        else:
            return prompt_tokens


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for fusing video and text features.
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-attention for prompt tokens
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention for video features
        self.video_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.video_cross_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention for text features
        self.text_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_cross_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, prompt_tokens, video_features, text_features, 
                video_mask=None, text_mask=None, return_attention=False):
        """
        Forward pass through the cross-attention block.
        
        Args:
            prompt_tokens: Prompt token features [batch_size, num_tokens, hidden_dim]
            video_features: Video features [batch_size, video_len, hidden_dim]
            text_features: Text features [batch_size, text_len, hidden_dim]
            video_mask: Optional mask for video features [batch_size, video_len]
            text_mask: Optional mask for text features [batch_size, text_len]
            return_attention: Whether to return attention weights
            
        Returns:
            prompt_tokens: Updated prompt tokens
            attention_weights: Optional attention weights
        """
        # Self-attention
        residual = prompt_tokens
        prompt_tokens = self.self_attention_norm(prompt_tokens)
        prompt_tokens, _ = self.self_attention(
            prompt_tokens, prompt_tokens, prompt_tokens
        )
        prompt_tokens = self.dropout(prompt_tokens)
        prompt_tokens = residual + prompt_tokens
        
        # Cross-attention with video
        residual = prompt_tokens
        prompt_tokens = self.video_cross_norm(prompt_tokens)
        
        # Convert mask to proper format for MultiheadAttention
        video_attn_mask = None
        if video_mask is not None:
            # Make sure it's a boolean mask
            video_attn_mask = ~video_mask.bool()
        
        prompt_tokens, video_attn_weights = self.video_cross_attention(
            prompt_tokens, video_features, video_features, 
            key_padding_mask=video_attn_mask
        )
        prompt_tokens = self.dropout(prompt_tokens)
        prompt_tokens = residual + prompt_tokens
        
        # Cross-attention with text
        residual = prompt_tokens
        prompt_tokens = self.text_cross_norm(prompt_tokens)
        
        # Convert mask to proper format for MultiheadAttention
        text_attn_mask = None
        if text_mask is not None:
            # Make sure it's a boolean mask
            text_attn_mask = ~text_mask.bool()
        
        prompt_tokens, text_attn_weights = self.text_cross_attention(
            prompt_tokens, text_features, text_features,
            key_padding_mask=text_attn_mask
        )
        prompt_tokens = self.dropout(prompt_tokens)
        prompt_tokens = residual + prompt_tokens
        
        # Feed-forward network
        residual = prompt_tokens
        prompt_tokens = self.ffn_norm(prompt_tokens)
        prompt_tokens = self.ffn(prompt_tokens)
        prompt_tokens = residual + prompt_tokens
        
        # Return attention weights if requested
        if return_attention:
            # Combine video and text attention weights
            return prompt_tokens, (video_attn_weights, text_attn_weights)
        else:
            return prompt_tokens, None

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, video_embeds, text_embeds):
        """
        Calculate InfoNCE loss for video and text embeddings.
        
        Args:
            video_embeds: Video embeddings [batch_size, embed_dim]
            text_embeds: Text embeddings [batch_size, embed_dim]
            
        Returns:
            loss: InfoNCE loss
        """
        # Normalize embeddings
        video_embeds = F.normalize(video_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)
        
        # Calculate similarity matrix
        sim_matrix = torch.matmul(video_embeds, text_embeds.t()) / self.temperature
        
        # Labels are the diagonal elements (matching pairs)
        batch_size = video_embeds.shape[0]
        labels = torch.arange(batch_size, device=video_embeds.device)
        
        # Calculate loss for video->text direction
        v2t_loss = self.criterion(sim_matrix, labels)
        
        # Calculate loss for text->video direction
        t2v_loss = self.criterion(sim_matrix.t(), labels)
        
        # Combine losses
        loss = (v2t_loss + t2v_loss) / 2
        
        return loss


class StepDescriptionEncoder(nn.Module):
    """
    Encoder for step descriptions using T5 tokenizer embeddings.
    """
    def __init__(self, t5_tokenizer, hidden_dim=768, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.tokenizer = t5_tokenizer
        self.vocab_size = t5_tokenizer.vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding layer - use T5's vocab size
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=self.tokenizer.pad_token_id)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Initialize weights
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' not in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Encode step descriptions.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            embeddings: Encoded text embeddings [batch_size, seq_len, hidden_dim]
        """
        # Get token embeddings
        embeddings = self.embedding(input_ids)
        
        # Apply transformer encoder
        if attention_mask is not None:
            # Convert mask to proper format (1 for tokens to attend to, 0 for tokens to ignore)
            key_padding_mask = ~attention_mask.bool()
            encoded = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        else:
            encoded = self.encoder(embeddings)
        
        return encoded


class ProceduralVideoStepDataset(torch.utils.data.Dataset):
    """
    Dataset for procedural video steps with temporal features.
    """
    def __init__(self, features_dir, annotation_file, split="train", t5_tokenizer=None, max_step_length=128):
        self.features_dir = features_dir
        self.split = split
        self.max_step_length = max_step_length
        self.t5_tokenizer = t5_tokenizer
        
        # Load annotations
        logger.info(f"Loading annotations from {annotation_file}")
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Extract videos for this split
            self.videos = []
            
            # Handle different data formats
            if 'database' in data:
                # ActNet format
                for video_id, video_data in data['database'].items():
                    self.videos.append({
                        'video_id': video_id,
                        'steps': video_data.get('annotation', []),
                        'class': video_data.get('class', '')
                    })
            else:
                # Direct mapping format
                for video_id, video_data in data.items():
                    self.videos.append({
                        'video_id': video_id,
                        'steps': video_data.get('annotation', []),
                        'class': video_data.get('class', '')
                    })
            
            logger.info(f"Loaded {len(self.videos)} videos for {split} split")
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            self.videos = []
            
        # If no videos were loaded, try to find them directly in the features directory
        if len(self.videos) == 0:
            self._load_from_features_dir()
    
    def _load_from_features_dir(self):
        """Load videos directly from the features directory"""
        logger.info(f"Attempting to load videos from features directory for {self.split} split")
        self.videos = []
        
        # The expected path structure: features_dir/split/split/subfolder/video_id.pt
        split_dir = os.path.join(self.features_dir, self.split)
        
        if os.path.exists(split_dir):
            # Find all .pt files recursively
            pt_files = []
            for root, _, files in os.walk(split_dir):
                for file in files:
                    if file.endswith('.pt'):
                        pt_files.append(os.path.join(root, file))
            
            # Extract video IDs from the paths
            for file_path in pt_files:
                # Get the base filename without extension
                file_name = os.path.basename(file_path)
                video_id = os.path.splitext(file_name)[0]
                
                self.videos.append({
                    'video_id': video_id,
                    'file_path': file_path,
                    'steps': [],  # No step info available
                    'class': ''   # No class info available
                })
            
            logger.info(f"Found {len(self.videos)} videos from features directory for {self.split} split")
        else:
            logger.error(f"Features directory for {self.split} split does not exist: {split_dir}")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_data = self.videos[idx]
        video_id = video_data['video_id']
        
        # Try to load features using the file path if available
        if 'file_path' in video_data and os.path.exists(video_data['file_path']):
            feature_path = video_data['file_path']
        else:
            # Dynamically search for the feature file
            feature_path = None
            split_dir = os.path.join(self.features_dir, self.split)
            
            # First try the expected nested path structure
            for subdir in os.listdir(os.path.join(split_dir, self.split)):
                potential_path = os.path.join(split_dir, self.split, subdir, f"{video_id}.pt")
                if os.path.exists(potential_path):
                    feature_path = potential_path
                    break
            
            # If not found, do a recursive search
            if feature_path is None:
                for root, _, files in os.walk(split_dir):
                    for file in files:
                        if file == f"{video_id}.pt":
                            feature_path = os.path.join(root, file)
                            break
                    if feature_path:
                        break
        
        # Load the features
        if feature_path and os.path.exists(feature_path):
            try:
                features_data = torch.load(feature_path)
                video_features = features_data['features']
                timestamps = features_data.get('metadata', {}).get('timestamps', None)
            except Exception as e:
                logger.error(f"Error loading features for {video_id}: {e}")
                video_features = torch.zeros(1, 512)
                timestamps = None
        else:
            logger.warning(f"Feature file for {video_id} not found")
            video_features = torch.zeros(1, 512)
            timestamps = None
        
        # Process steps
        steps = video_data.get('steps', [])
        step_text = ""
        step_boundaries = []
        
        for step in steps:
            if 'label' in step and 'segment' in step:
                step_text += step['label'] + " "
                if timestamps is not None:
                    start_time, end_time = step['segment']
                    start_idx = 0
                    end_idx = len(timestamps) - 1
                    
                    # Find closest indices
                    if len(timestamps) > 0:
                        for i, t in enumerate(timestamps):
                            if t >= start_time:
                                start_idx = i
                                break
                        
                        for i in range(len(timestamps)-1, -1, -1):
                            if timestamps[i] <= end_time:
                                end_idx = i
                                break
                    
                    step_boundaries.append((start_idx, end_idx))
        
        # If no step text, use a placeholder
        if not step_text:
            step_text = "No step information available"
        
        # Tokenize step text
        if self.t5_tokenizer:
            tokenized = self.t5_tokenizer(
                step_text.strip(),
                padding="max_length",
                truncation=True,
                max_length=self.max_step_length,
                return_tensors="pt"
            )
            input_ids = tokenized.input_ids.squeeze(0)
            attention_mask = tokenized.attention_mask.squeeze(0)
        else:
            # Create dummy tensors if no tokenizer
            input_ids = torch.zeros(self.max_step_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_step_length, dtype=torch.long)
        
        return {
            'video_id': video_id,
            'features': video_features,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'step_text': step_text,
            'step_boundaries': step_boundaries,
            'class': video_data.get('class', '')
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Handle empty batch case
    if not batch:
        raise ValueError("Empty batch received")
    
    # Check if features exist in the first item
    if 'features' not in batch[0] or batch[0]['features'] is None:
        raise ValueError(f"Missing or invalid 'features' in batch item: {batch[0].keys()}")
    
    # Get feature dimensions from the first item
    try:
        # Handle both 1D and 2D features
        if len(batch[0]['features'].shape) == 1:
            video_dim = batch[0]['features'].shape[0]
            # Reshape 1D tensor to 2D
            for i in range(len(batch)):
                if len(batch[i]['features'].shape) == 1:
                    batch[i]['features'] = batch[i]['features'].unsqueeze(0)
        else:
            video_dim = batch[0]['features'].shape[1]
    except Exception as e:
        raise ValueError(f"Error getting feature dimensions: {e}, Feature shape: {batch[0]['features'].shape}")
    
    # Get max video length in batch
    max_video_len = max([item['features'].shape[0] for item in batch])
    
    # Initialize tensors
    batch_size = len(batch)
    video_features = torch.zeros(batch_size, max_video_len, video_dim)
    video_masks = torch.zeros(batch_size, max_video_len, dtype=torch.bool)
    
    # Text fields have fixed length due to tokenizer padding
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Other fields
    video_ids = []
    step_texts = []
    step_boundaries = []
    classes = []
    
    for i, item in enumerate(batch):
        # Video features
        vid_len = item['features'].shape[0]
        video_features[i, :vid_len] = item['features']
        video_masks[i, :vid_len] = True  # Mark valid positions
        
        # Other fields
        video_ids.append(item['video_id'])
        step_texts.append(item.get('step_text', ''))
        step_boundaries.append(item.get('step_boundaries', []))
        classes.append(item.get('class', ''))
    
    return {
        'video_ids': video_ids,
        'features': video_features,
        'video_masks': video_masks,
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'step_texts': step_texts,
        'step_boundaries': step_boundaries,
        'classes': classes
    }

def train_epoch(model, text_encoder, dataloader, optimizer, criterion, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: Cross-modal fusion model
        text_encoder: Text encoder model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        average_loss: Average loss for this epoch
    """
    model.train()
    text_encoder.train()
    
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} (train)')
    
    for batch in progress_bar:
        # Move data to device
        # In train_epoch
        video_features = batch['features'].to(device)
        video_masks = batch['video_masks'].to(device)  # Make sure this is boolean
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)  # Make sure this is boolean
        
        # Encode text
        text_features = text_encoder(input_ids, attention_masks)
        
        # Forward pass through fusion model
        prompt_tokens = model(
            video_features,
            text_features,
            video_masks,
            attention_masks
        )
        
        # Calculate contrastive loss
        # Use average of prompt tokens as video representation
        video_embeds = prompt_tokens.mean(dim=1)
        # Use average of text features as text representation
        text_embeds = text_features.mean(dim=1)
        
        loss = criterion(video_embeds, text_embeds)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    average_loss = total_loss / num_batches
    return average_loss


def validate(model, text_encoder, dataloader, criterion, device, epoch):
    """
    Validate the model.
    
    Args:
        model: Cross-modal fusion model
        text_encoder: Text encoder model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        average_loss: Average validation loss
    """
    model.eval()
    text_encoder.eval()
    
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} (val)')
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            video_features = batch['features'].to(device)
            video_masks = batch['video_masks'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            
            # Encode text
            text_features = text_encoder(input_ids, attention_masks)
            
            # Forward pass through fusion model
            prompt_tokens = model(
                video_features,
                text_features,
                video_masks,
                attention_masks
            )
            
            # Calculate contrastive loss
            video_embeds = prompt_tokens.mean(dim=1)
            text_embeds = text_features.mean(dim=1)
            
            loss = criterion(video_embeds, text_embeds)
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    average_loss = total_loss / num_batches
    return average_loss


def visualize_attention_maps(model, text_encoder, dataloader, device, output_dir, num_samples=5):
    """
    Visualize attention maps for selected samples.
    
    Args:
        model: Cross-modal fusion model
        text_encoder: Text encoder model
        dataloader: Dataloader to get samples from
        device: Device to run on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    model.eval()
    text_encoder.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        # Move data to device
        video_features = batch['features'].to(device)
        video_masks = batch['video_masks'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)
        
        # Encode text
        text_features = text_encoder(input_ids, attention_masks)
        
        # Forward pass with attention weights
        prompt_tokens, attention_weights = model(
            video_features,
            text_features,
            video_masks,
            attention_masks,
            return_attention=True
        )
        
        # Video attention weights and text attention weights
        video_attn, text_attn = attention_weights
        
        # Process attention weights for selected samples
        for i in range(min(num_samples, len(batch['video_ids']))):
            video_id = batch['video_ids'][i]
            step_text = batch['step_texts'][i]
            step_boundaries = batch['step_boundaries'][i]
            
            # Create video timeline indices
            video_length = video_features[i].shape[0]
            timeline_indices = list(range(video_length))
            
            # Generate attention heatmap
            video_attn_map = video_attn[i].cpu().numpy()
            
            video_vis_path = os.path.join(output_dir, f"{video_id}_video_attention.png")
            fig1 = visualize_attention(
                video_attn_map, 
                timeline_indices, 
                video_vis_path
            )
            plt.close(fig1)  # Close figure after saving
            
            # Generate timeline visualization
            timeline_path = os.path.join(output_dir, f"{video_id}_timeline.png")
            fig2 = visualize_attention_timeline(
                video_attn_map,
                timeline_indices,
                step_boundaries,
                timeline_path
            )
            plt.close(fig2)  # Close figure after saving
            
            # Create text file with information
            info_path = os.path.join(output_dir, f"{video_id}_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Video ID: {video_id}\n")
                f.write(f"Step text: {step_text}\n")
                f.write(f"Step boundaries: {step_boundaries}\n")
                f.write(f"Video length: {video_length} frames\n")
            
            logger.info(f"Generated visualizations for {video_id}")


def main(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = set_device(args.device_id)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    
    # Save configuration
    config = vars(args)
    save_config(config, os.path.join(exp_dir, 'config.json'))
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
    
    # Initialize T5 tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model_name)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ProceduralVideoStepDataset(
        args.features_dir, 
        args.train_annotation_file,
        split="train",
        t5_tokenizer=t5_tokenizer,
        max_step_length=args.max_text_length
    )
    
    val_dataset = ProceduralVideoStepDataset(
        args.features_dir, 
        args.val_annotation_file,
        split="validation",
        t5_tokenizer=t5_tokenizer,
        max_step_length=args.max_text_length
    )
    
    test_dataset = ProceduralVideoStepDataset(
        args.features_dir, 
        args.test_annotation_file,
        split="test",
        t5_tokenizer=t5_tokenizer,
        max_step_length=args.max_text_length
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create models
    logger.info("Creating models...")
    
    # Text encoder for step descriptions
    text_encoder = StepDescriptionEncoder(
        t5_tokenizer,
        hidden_dim=args.hidden_dim,
        num_layers=args.text_encoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    # Cross-modal fusion model
    model = CrossModalFusionModel(
        video_dim=args.video_dim,
        text_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_prompt_tokens=args.num_prompt_tokens,
        prompt_dim=args.prompt_dim
    ).to(device)
    
    # Log model sizes
    logger.info(f"Text encoder parameters: {count_parameters(text_encoder):,}")
    logger.info(f"Cross-modal fusion parameters: {count_parameters(model):,}")
    
    # Create criterion for contrastive learning
    criterion = InfoNCELoss(temperature=args.temperature)
    
    # Create optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(text_encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True
    )
    
    # Create early stopping
    early_stopping = AdaptiveEarlyStopping(
        initial_patience=args.initial_patience,
        max_patience=args.max_patience,
        patience_increase=args.patience_increase
    )
    
    # Load checkpoint if resuming training
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        logger.info(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'early_stopping' in checkpoint:
            early_stopping.load_state_dict(checkpoint['early_stopping'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.4f}")
    
    # Training loop
    logger.info("Starting training...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, args.max_epochs):
        # Training
        train_loss = train_epoch(
            model, text_encoder, train_loader, 
            optimizer, criterion, device, epoch
        )
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate(
            model, text_encoder, val_loader,
            criterion, device, epoch
        )
        val_losses.append(val_loss)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Increase patience if learning rate decreased
        if new_lr < old_lr:
            early_stopping.increase_patience()
            logger.info(f"Learning rate decreased to {new_lr:.6f}, increased patience to {early_stopping.patience}")
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.max_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"LR: {new_lr:.6f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('LR', new_lr, epoch)
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pt')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping': early_stopping.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"Saved best model with validation loss {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping': early_stopping.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
        
        # Generate visualizations periodically
        if (epoch + 1) % args.visualize_every == 0:
            vis_dir = os.path.join(exp_dir, 'visualizations', f'epoch_{epoch+1}')
            os.makedirs(vis_dir, exist_ok=True)
            
            visualize_attention_maps(
                model, text_encoder, val_loader,
                device, vis_dir, args.vis_samples
            )
            
            # Plot training curves
            plot_training_curves(
                train_losses, val_losses,
                output_dir=os.path.join(exp_dir, 'visualizations')
            )
            
            logger.info(f"Generated visualizations for epoch {epoch+1}")
        
        # Check early stopping
        if early_stopping(val_loss, epoch):
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    best_model_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pt')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model for evaluation")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        
        best_epoch = checkpoint['epoch']
        logger.info(f"Loaded best model from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    
    # Evaluate on test set
    test_loss = validate(model, text_encoder, test_loader, criterion, device, -1)
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Generate final visualizations
    final_vis_dir = os.path.join(exp_dir, 'visualizations', 'final')
    os.makedirs(final_vis_dir, exist_ok=True)
    
    visualize_attention_maps(
        model, text_encoder, test_loader,
        device, final_vis_dir, args.vis_samples
    )
    
    # Plot final training curves
    plot_training_curves(
        train_losses, val_losses,
        output_dir=os.path.join(exp_dir, 'visualizations')
    )
    
    # Save final model
    final_model_path = os.path.join(exp_dir, 'checkpoints', 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'text_encoder_state_dict': text_encoder.state_dict(),
        'config': config,
        'train_loss': train_losses[-1] if train_losses else float('inf'),
        'val_loss': val_losses[-1] if val_losses else float('inf'),
        'test_loss': test_loss
    }, final_model_path)
    
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    writer.close()

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Cross-Modal Fusion Training")
   
   # Data paths
   parser.add_argument('--features_dir', type=str, default='Raw_Dataset/temporal_features',
                      help='Directory containing temporal features')
   parser.add_argument('--train_annotation_file', type=str, default='Raw_Dataset/splits/train.json',
                      help='Path to training annotation file')
   parser.add_argument('--val_annotation_file', type=str, default='Raw_Dataset/splits/validation.json',
                      help='Path to validation annotation file')
   parser.add_argument('--test_annotation_file', type=str, default='Raw_Dataset/splits/test.json',
                      help='Path to test annotation file')
   parser.add_argument('--output_dir', type=str, default='experiments/cross_modal',
                      help='Output directory for experiments')
   
   # Experiment settings
   parser.add_argument('--experiment_name', type=str, default=None,
                      help='Experiment name (default: timestamp)')
   parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
   parser.add_argument('--device_id', type=int, default=0,
                      help='GPU device ID')
   parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of dataloader workers')
   
   # Model parameters
   parser.add_argument('--t5_model_name', type=str, default='t5-base',
                      help='T5 model name for tokenizer')
   parser.add_argument('--video_dim', type=int, default=512,
                      help='Dimension of video features')
   parser.add_argument('--hidden_dim', type=int, default=512,
                      help='Hidden dimension for fusion model')
   parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
   parser.add_argument('--num_layers', type=int, default=4,
                      help='Number of fusion layers')
   parser.add_argument('--text_encoder_layers', type=int, default=4,
                      help='Number of text encoder layers')
   parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
   parser.add_argument('--num_prompt_tokens', type=int, default=24,
                      help='Number of prompt tokens to generate')
   parser.add_argument('--prompt_dim', type=int, default=512,
                      help='Dimension of prompt tokens')
   parser.add_argument('--max_text_length', type=int, default=128,
                      help='Maximum text sequence length')
   
   # Training parameters
   parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
   parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
   parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
   parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of epochs')
   parser.add_argument('--temperature', type=float, default=0.07,
                      help='Temperature for InfoNCE loss')
   
   # Scheduler and early stopping
   parser.add_argument('--lr_factor', type=float, default=0.5,
                      help='Factor by which to reduce learning rate')
   parser.add_argument('--lr_patience', type=int, default=5,
                      help='Patience for learning rate scheduler')
   parser.add_argument('--initial_patience', type=int, default=8,
                      help='Initial patience for early stopping')
   parser.add_argument('--max_patience', type=int, default=20,
                      help='Maximum patience for early stopping')
   parser.add_argument('--patience_increase', type=int, default=2,
                      help='Patience increase after learning rate reduction')
   
   # Checkpointing
   parser.add_argument('--resume_checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume from')
   parser.add_argument('--save_every', type=int, default=5,
                      help='Save checkpoint every N epochs')
   
   # Visualization
   parser.add_argument('--visualize_every', type=int, default=5,
                      help='Generate visualizations every N epochs')
   parser.add_argument('--vis_samples', type=int, default=5,
                      help='Number of samples to visualize')
   
   args = parser.parse_args()
   
   main(args)