import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import logging
import math
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("timesformer_visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed()

# Define model components directly to avoid import errors
class TimeSformerDividedAttention(nn.Module):
    """Divided space-time attention mechanism for TimeSformer."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert boolean mask to float mask where False -> -inf
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            mask = mask.to(x.dtype)  # Convert to same dtype as attention
            mask = mask.masked_fill(~mask.bool(), float("-inf"))
            attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn  # Return attention weights for visualization

class RelativePositionalEncoding(nn.Module):
    """Flexible relative positional encoding for temporal awareness."""
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Dynamically create larger buffer
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :(dim//2)]
            
        self.register_buffer('pe', pe)
        
    def forward(self, x, timestamps=None):
        B, N, D = x.shape
        
        # Dynamically adjust buffer size if needed
        if N > self.max_seq_len:
            # Create a new, larger buffer
            new_max_len = max(N, self.max_seq_len * 2)
            new_pe = torch.zeros(new_max_len, D).to(x.device)
            
            # Recreate positional encoding
            position = torch.arange(0, new_max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
            
            new_pe[:, 0::2] = torch.sin(position * div_term)
            if D % 2 == 0:
                new_pe[:, 1::2] = torch.cos(position * div_term)
            else:
                new_pe[:, 1::2] = torch.cos(position * div_term)[:, :(D//2)]
            
            # Update the buffer
            self.register_buffer('pe', new_pe.to(x.device))
            self.max_seq_len = new_max_len
        
        if timestamps and any(timestamps):
            # Use actual timestamps for more precise temporal modeling
            pos_enc = torch.zeros(B, N, D).to(x.device)
            
            for i in range(B):
                # Normalize timestamps to 0-1 range
                if timestamps[i]:
                    t = torch.tensor(timestamps[i] if isinstance(timestamps[i], list) else [0] * N).to(x.device)
                    
                    # Ensure tensor is same length as sequence
                    if len(t) < N:
                        t = F.pad(t, (0, N - len(t)), value=t[-1] if len(t) > 0 else 0)
                    
                    # Normalize timestamps
                    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)  # Prevent div by zero
                    t = t * (self.max_seq_len - 1)  # Scale to max_seq_len
                    
                    # Interpolate position encodings based on timestamps
                    for j in range(N):
                        idx = min(int(t[j]), self.max_seq_len - 1)
                        pos_enc[i, j] = self.pe[idx]
            
            return x + pos_enc
        else:
            # Use standard sequential positions
            return x + self.pe[:N].unsqueeze(0)

class TimeSformerBlock(nn.Module):
    """TimeSformer block with divided space-time attention."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cross_frame=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TimeSformerDividedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # Drop path for stochastic depth
        self.drop_path = nn.Identity()
        if drop_path > 0:
            self.drop_path = nn.Dropout(drop_path)
            
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Cross-frame attention enhancement
        self.cross_frame = cross_frame
        if cross_frame:
            self.cross_norm = norm_layer(dim)
            self.cross_attn = TimeSformerDividedAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x, attention_mask=None):
        # Standard attention block
        attn_output, attn_weights = self.attn(self.norm1(x), attention_mask)
        x = x + self.drop_path(attn_output)
        
        # Optional cross-frame attention
        if self.cross_frame:
            cross_output, cross_weights = self.cross_attn(self.cross_norm(x), attention_mask)
            x = x + self.drop_path(cross_output)
            # Combine attention weights
            attn_weights = (attn_weights + cross_weights) / 2
            
        # MLP block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, attn_weights

class HierarchicalAttentionPooling(nn.Module):
    """Hierarchical attention pooling with step-aware aggregation."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x, attention_mask=None, step_boundaries=None):
        """Apply hierarchical attention pooling."""
        B, N, D = x.shape
        
        if step_boundaries is not None:
            # Step-aware pooling
            pooled_steps = []
            
            for i in range(B):
                # Get steps for this video
                if i < len(step_boundaries) and step_boundaries[i]:
                    steps = step_boundaries[i]
                    video_features = []
                    
                    # Pool each step separately
                    for start, end in steps:
                        start_idx = min(max(0, start), N-1)
                        end_idx = min(max(start_idx+1, end), N)
                        
                        if end_idx - start_idx > 1:
                            # Apply attention within this step
                            step_x = x[i:i+1, start_idx:end_idx]
                            step_mask = None
                            if attention_mask is not None:
                                step_mask = attention_mask[i:i+1, start_idx:end_idx]
                                
                            query = self.query.expand(1, 1, D)
                            step_pooled, _ = self.attention(query, step_x, step_x, 
                                                            key_padding_mask=~step_mask if step_mask is not None else None)
                            video_features.append(step_pooled.squeeze(0))
                    
                    if video_features:
                        # Secondary pooling across steps
                        video_x = torch.stack(video_features, dim=0)
                        second_query = self.query.expand(1, 1, D)
                        video_pooled, _ = self.attention(second_query, video_x, video_x)
                        pooled_steps.append(video_pooled.squeeze(0))
                    else:
                        # Fallback if no valid steps
                        pooled_steps.append(self.global_pool(x[i:i+1], 
                                                           attention_mask[i:i+1] if attention_mask is not None else None))
                else:
                    # Fallback to global pooling
                    pooled_steps.append(self.global_pool(x[i:i+1], 
                                                       attention_mask[i:i+1] if attention_mask is not None else None))
            
            return torch.cat(pooled_steps, dim=0)
        else:
            # Global pooling for all videos
            return self.global_pool(x, attention_mask)
            
    def global_pool(self, x, attention_mask=None):
        """Global attention pooling."""
        B = x.shape[0]
        query = self.query.expand(B, 1, x.shape[-1])
        
        # Apply global attention pooling
        if attention_mask is not None:
            pooled, _ = self.attention(query, x, x, key_padding_mask=~attention_mask)
        else:
            pooled, _ = self.attention(query, x, x)
            
        return pooled.squeeze(1)

class TimeSformer(nn.Module):
    """TimeSformer model for temporal modeling of CLIP features."""
    def __init__(
        self,
        input_dim=512,
        output_dim=512,  # Reduced from 768
        depth=6,  # Reduced from 8
        num_heads=4,  # Reduced from 8
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        max_seq_len=512,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.norm_in = nn.LayerNorm(output_dim)
        
        # Output projection (for self-supervised training)
        self.output_proj = nn.Linear(output_dim, input_dim)
        
        # Positional encoding
        self.pos_encoding = RelativePositionalEncoding(output_dim, max_seq_len)
        
        # TimeSformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Use cross-frame attention in later layers
            use_cross_frame = i >= depth // 2
            
            self.blocks.append(
                TimeSformerBlock(
                    dim=output_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    cross_frame=use_cross_frame
                )
            )
        
        # Output pooling
        self.pooling = HierarchicalAttentionPooling(output_dim, num_heads=num_heads)
        self.norm_out = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use Xavier initialization for linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, attention_mask=None, timestamps=None, step_boundaries=None, return_attention=False):
        """Forward pass through the TimeSformer."""
        # Project input features
        x = self.input_proj(x)
        x = self.norm_in(x)
        
        # Add positional encoding
        x = self.pos_encoding(x, timestamps)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Apply transformer blocks
        for block in self.blocks:
            x, attn = block(x, attention_mask)
            if return_attention:
                attention_weights.append(attn)
        
        # Apply pooling
        x = self.pooling(x, attention_mask, step_boundaries)
        x = self.norm_out(x)
        
        if return_attention:
            return x, attention_weights
        return x

def visualize_attention(model, features_dir, output_dir, num_videos=5, device_id=1):
    """Visualize attention patterns from the TimeSformer model."""
    logger.info("Visualizing attention patterns...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load a trained model
    checkpoint_path = os.path.join('Raw_Dataset/models/timesformer', 'best_model.pth')
    logger.info(f"Loading model from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info(f"Loaded model from epoch {checkpoint['epoch']+1}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Find feature files
    feature_files = []
    for root, _, files in os.walk(features_dir):
        for file in files:
            if file.endswith('.pt'):
                feature_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    if len(feature_files) == 0:
        logger.error(f"No feature files found in {features_dir}")
        return
    
    # Randomly select videos
    selected_files = random.sample(feature_files, min(num_videos, len(feature_files)))
    logger.info(f"Selected {len(selected_files)} videos for attention visualization")
    
    for file_path in tqdm(selected_files, desc="Processing videos"):
        try:
            # Load features
            data = torch.load(file_path)
            features = data['features'].unsqueeze(0).to(device)  # Add batch dimension
            video_id = data['metadata']['video_id']
            timestamps = data['metadata'].get('timestamps', None)
            
            # Get sequence length
            seq_len = features.size(1)
            
            # Create attention mask
            attention_mask = torch.ones(1, seq_len).bool().to(device)
            
            # Forward pass with attention weights
            with torch.no_grad():
                _, attention_weights = model(features, attention_mask, timestamps, return_attention=True)
            
            # Visualize attention for each layer
            num_layers = len(attention_weights)
            rows = (num_layers + 3) // 4  # Ceiling division
            cols = min(4, num_layers)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
            if rows * cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, attn in enumerate(attention_weights):
                if i < len(axes):  # Visualize available layers
                    ax = axes[i]
                    # Average over attention heads
                    attn_map = attn.mean(dim=1).numpy()
                    im = ax.imshow(attn_map[0], cmap='viridis', aspect='auto')
                    ax.set_title(f"Layer {i+1}")
                    ax.set_xlabel("Key position")
                    ax.set_ylabel("Query position")
                    fig.colorbar(im, ax=ax)
            
            # Hide unused subplots
            for i in range(num_layers, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{video_id}_attention.png"), dpi=300)
            plt.close()
            
            # Create temporal flow visualization
            if len(attention_weights) > 0 and seq_len > 1:
                # Take the last layer's attention for temporal analysis
                last_layer_attn = attention_weights[-1].mean(dim=1)[0].numpy()
                
                plt.figure(figsize=(12, 6))
                plt.imshow(last_layer_attn, cmap='inferno', aspect='auto')
                plt.colorbar(label='Attention Weight')
                plt.title(f"Temporal Attention Flow - {video_id}")
                plt.xlabel("Frame Position (Target)")
                plt.ylabel("Frame Position (Source)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{video_id}_temporal_flow.png"), dpi=300)
                plt.close()
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info(f"Attention visualizations saved to {output_dir}")

def visualize_embeddings(temporal_features_dir, output_dir, num_videos=200):
    """Visualize temporal embeddings using t-SNE and PCA."""
    logger.info("Visualizing temporal embeddings...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if temporal features directory exists
    if not os.path.exists(temporal_features_dir):
        logger.error(f"Temporal features directory {temporal_features_dir} does not exist")
        return
    
    # Check if train split exists
    train_dir = os.path.join(temporal_features_dir, 'train')
    if not os.path.exists(train_dir):
        logger.error(f"Train split directory {train_dir} does not exist")
        return
    
    # Collect temporal embeddings
    embeddings = []
    labels = []  # Using subfolder as label
    video_ids = []
    classes = []  # Store video class if available
    
    # Find all feature files in train split
    feature_files = []
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.pt'):
                feature_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(feature_files)} temporal feature files")
    
    if len(feature_files) == 0:
        logger.error(f"No temporal feature files found in {train_dir}")
        return
    
    # Randomly sample if too many
    if len(feature_files) > num_videos:
        feature_files = random.sample(feature_files, num_videos)
        logger.info(f"Randomly sampled {num_videos} videos")
    
    # Load embeddings
    for file_path in tqdm(feature_files, desc="Loading embeddings"):
        try:
            data = torch.load(file_path)
            embedding = data['features'].numpy()
            embeddings.append(embedding)
            
            # Get video ID
            video_id = data['metadata']['video_id']
            video_ids.append(video_id)
            
            # Try to get class
            if 'class' in data['metadata']:
                classes.append(data['metadata']['class'])
            else:
                classes.append('unknown')
            
            # Get label from subfolder
            parts = Path(file_path).parts
            subfolder_idx = parts.index('train') + 1
            if subfolder_idx < len(parts):
                label = parts[subfolder_idx]
                labels.append(label)
            else:
                labels.append('unknown')
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if len(embeddings) == 0:
        logger.error("No valid embeddings loaded")
        return
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Run t-SNE
    try:
        logger.info("Running t-SNE...")
        perplexity = min(30, len(embeddings)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create colormap based on subfolder labels
        unique_labels = list(set(labels))
        label_to_color = {label: i for i, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in labels]
        
        # Plot embeddings by subfolder
        plt.figure(figsize=(14, 12))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='tab20', alpha=0.7)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab20(label_to_color[label] / len(unique_labels)), 
                                    markersize=10, label=f"{label} ({labels.count(label)})") 
                        for label in unique_labels]
        plt.legend(handles=legend_elements, loc='best', title='Subfolders', bbox_to_anchor=(1.05, 1), fontsize='small')
        
        plt.title('t-SNE Visualization of Temporal Embeddings by Subfolder')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_embeddings_tsne.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error running t-SNE: {e}")
    
    # Run PCA
    try:
        logger.info("Running PCA...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Plot embeddings by subfolder
        plt.figure(figsize=(14, 12))
        scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=colors, cmap='tab20', alpha=0.7)
        
        # Add legend
        plt.legend(handles=legend_elements, loc='best', title='Subfolders', bbox_to_anchor=(1.05, 1), fontsize='small')
        
        plt.title('PCA Visualization of Temporal Embeddings')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_embeddings_pca.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error running PCA: {e}")
    
    logger.info(f"Embedding visualizations saved to {output_dir}")

def plot_training_curves(log_dir, output_dir):
    """Plot training and validation loss curves from TensorBoard logs."""
    logger.info("Plotting training curves...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Check if log directory exists
        if not os.path.exists(log_dir):
            logger.error(f"Log directory {log_dir} does not exist")
            return
        
        # Load TensorBoard data
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        available_tags = event_acc.Tags()['scalars']
        logger.info(f"Available scalar tags: {available_tags}")
        
        # Check if required tags exist
        required_tags = ['Loss/train', 'Loss/validation']
        missing_tags = [tag for tag in required_tags if tag not in available_tags]
        if missing_tags:
            logger.error(f"Missing required tags: {missing_tags}")
            return
        
        # Get scalars
        train_loss = [(s.step, s.value) for s in event_acc.Scalars('Loss/train')]
        val_loss = [(s.step, s.value) for s in event_acc.Scalars('Loss/validation')]
        
        # Check if learning rate tag exists
        if 'LearningRate' in available_tags:
            lr = [(s.step, s.value) for s in event_acc.Scalars('LearningRate')]
            lr_steps, lr_values = zip(*lr)
        else:
            logger.warning("LearningRate tag not found in TensorBoard logs")
            lr_steps, lr_values = None, None
        
        # Extract steps and values
        train_steps, train_values = zip(*train_loss)
        val_steps, val_values = zip(*val_loss)
        
        # Plot training and validation loss
        plt.figure(figsize=(14, 7))
        plt.plot(train_steps, train_values, 'b-', linewidth=1.5, label='Training Loss')
        plt.plot(val_steps, val_values, 'r-', linewidth=1.5, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set reasonable y-axis limits
        min_loss = min(min(train_values), min(val_values))
        max_loss = max(max(train_values[:10]), max(val_values[:10]))  # Consider only early epochs for max
        plt.ylim([max(0, min_loss * 0.95), min(max_loss * 1.05, max(train_values[-20:]) * 2)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
        plt.close()
        
        # Plot zoomed-in view of later epochs
        if len(train_steps) > 20:
            plt.figure(figsize=(14, 7))
            start_idx = len(train_steps) // 3  # Start from 1/3 of training
            
            plt.plot(train_steps[start_idx:], train_values[start_idx:], 'b-', linewidth=1.5, label='Training Loss')
            plt.plot(val_steps[start_idx:], val_values[start_idx:], 'r-', linewidth=1.5, label='Validation Loss')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss (Later Epochs)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust y-axis for zoom view
            later_min = min(min(train_values[start_idx:]), min(val_values[start_idx:]))
            later_max = max(max(train_values[start_idx:]), max(val_values[start_idx:]))
            plt.ylim([max(0, later_min * 0.95), later_max * 1.05])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'loss_curves_zoomed.png'), dpi=300)
            plt.close()
        
        # Plot learning rate if available
        if lr_steps and lr_values:
            plt.figure(figsize=(14, 7))
            plt.plot(lr_steps, lr_values, 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300)
            plt.close()
        
        # Calculate and save statistics
        train_min_idx = np.argmin(train_values)
        val_min_idx = np.argmin(val_values)
        
        stats = {
            "total_epochs": len(train_steps),
            "train_loss_min": min(train_values),
            "train_loss_min_epoch": train_steps[train_min_idx],
            "val_loss_min": min(val_values),
            "val_loss_min_epoch": val_steps[val_min_idx],
            "train_loss_final": train_values[-1],
            "val_loss_final": val_values[-1]
        }
        
        with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        
        logger.info(f"Training statistics: Best validation loss {stats['val_loss_min']} at epoch {stats['val_loss_min_epoch']}")
        
    except Exception as e:
        logger.error(f"Error plotting training curves: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info(f"Training curves saved to {output_dir}")

def analyze_embeddings_similarity(temporal_features_dir, output_dir, num_videos=100):
    """Analyze similarity between temporal embeddings of different videos."""
    logger.info("Analyzing embedding similarities...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if temporal features directory exists
    if not os.path.exists(temporal_features_dir):
        logger.error(f"Temporal features directory {temporal_features_dir} does not exist")
        return
    
    # Collect temporal embeddings from all splits
    all_embeddings = []
    all_video_ids = []
    all_classes = []
    
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(temporal_features_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory {split_dir} does not exist")
            continue
        
        # Find all feature files in the split
        feature_files = []
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith('.pt'):
                    feature_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(feature_files)} temporal feature files in {split} split")
        
        # Randomly sample if too many
        if len(feature_files) > num_videos:
            feature_files = random.sample(feature_files, num_videos)
            logger.info(f"Randomly sampled {num_videos} videos from {split} split")
        
        # Load embeddings
        for file_path in tqdm(feature_files, desc=f"Loading {split} embeddings"):
            try:
                data = torch.load(file_path)
                embedding = data['features'].numpy()
                all_embeddings.append(embedding)
                
                # Get video ID
                video_id = data['metadata']['video_id']
                all_video_ids.append(f"{split}_{video_id}")
                
                # Try to get class
                if 'class' in data['metadata']:
                    all_classes.append(data['metadata']['class'])
                else:
                    all_classes.append('unknown')
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
    
    if len(all_embeddings) == 0:
        logger.error("No valid embeddings loaded")
        return
    
    # Convert to array and normalize embeddings
    all_embeddings = np.array(all_embeddings)
    normalized_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Plot similarity matrix heatmap
    plt.figure(figsize=(16, 14))
    plt.imshow(similarity_matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity Between Temporal Embeddings')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_similarity_matrix.png'), dpi=300)
    plt.close()
    
    # Calculate and plot class-wise similarity
    if all_classes and any(c != 'unknown' for c in all_classes):
        unique_classes = list(set(all_classes))
        num_classes = len(unique_classes)
        
        if num_classes > 1:
            # Create class index mapping
            class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
            class_indices = [class_to_idx[cls] for cls in all_classes]
            
            # Calculate average similarity within and between classes
            class_similarities = np.zeros((num_classes, num_classes))
            class_counts = np.zeros((num_classes, num_classes))
            
            for i in range(len(all_embeddings)):
                for j in range(len(all_embeddings)):
                    if i != j:  # Exclude self-similarity
                        class_i = class_indices[i]
                        class_j = class_indices[j]
                        class_similarities[class_i, class_j] += similarity_matrix[i, j]
                        class_counts[class_i, class_j] += 1
            
            # Average the similarities
            for i in range(num_classes):
                for j in range(num_classes):
                    if class_counts[i, j] > 0:
                        class_similarities[i, j] /= class_counts[i, j]
            
            # Plot class similarity matrix
            plt.figure(figsize=(14, 12))
            plt.imshow(class_similarities, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Average Cosine Similarity')
            plt.title('Average Similarity Between Classes')
            plt.xticks(range(num_classes), unique_classes, rotation=90)
            plt.yticks(range(num_classes), unique_classes)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'class_similarity_matrix.png'), dpi=300)
            plt.close()
    
    logger.info(f"Embedding similarity analysis saved to {output_dir}")

def main(args):
    """Main function for TimeSformer visualization."""
    logger.info("Starting TimeSformer visualization")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot training curves
    if os.path.exists(args.log_dir):
        logger.info(f"Plotting training curves from {args.log_dir}")
        plot_training_curves(args.log_dir, args.output_dir)
    else:
        logger.error(f"Log directory {args.log_dir} does not exist")
    
    # Visualize embeddings if temporal features exist
    if os.path.exists(args.temporal_dir):
        logger.info(f"Visualizing embeddings from {args.temporal_dir}")
        visualize_embeddings(args.temporal_dir, args.output_dir, num_videos=args.num_videos)
        
        # Also analyze embedding similarities
        logger.info("Analyzing embedding similarities")
        analyze_embeddings_similarity(args.temporal_dir, args.output_dir, num_videos=args.num_videos)
    else:
        logger.warning(f"Temporal features directory {args.temporal_dir} does not exist yet")
    
    # Load model for attention visualization if CLIP features exist
    if os.path.exists(args.features_dir):
        try:
            # Create model with matching parameters to your trained model
            logger.info("Creating TimeSformer model for attention visualization")
            model = TimeSformer(
                input_dim=512,    # CLIP feature dimension
                output_dim=512,   # Updated to match your current training
                depth=6,          # Updated to match your current training
                num_heads=4       # Updated to match your current training
                
            )
            
            visualize_attention(
                model, 
                args.features_dir, 
                args.output_dir, 
                num_videos=min(5, args.num_videos),
                device_id=args.device_id
            )
        except Exception as e:
            logger.error(f"Error during attention visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"Features directory {args.features_dir} does not exist")
    
    logger.info(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimeSformer Visualization")
    parser.add_argument('--model_dir', type=str, default='Raw_Dataset/models/timesformer',
                        help='Directory containing saved model checkpoints')
    parser.add_argument('--features_dir', type=str, default='Raw_Dataset/features',
                        help='Directory containing CLIP features')
    parser.add_argument('--temporal_dir', type=str, default='Raw_Dataset/temporal_features',
                        help='Directory containing temporal features')
    parser.add_argument('--output_dir', type=str, default='Raw_Dataset/visualizations/timesformer',
                        help='Directory to save visualizations')
    parser.add_argument('--log_dir', type=str, default='runs/timesformer_training',
                        help='Directory containing TensorBoard logs')
    parser.add_argument('--num_videos', type=int, default=100,
                        help='Number of videos to use for visualizations')
    parser.add_argument('--device_id', type=int, default=1,
                        help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("TimeSformer Visualization Configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Run main function
    try:
        main(args)
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())