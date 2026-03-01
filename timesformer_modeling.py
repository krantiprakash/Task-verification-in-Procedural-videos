import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import math
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("timesformer_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
set_seed()

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

class FeatureDataset(Dataset):
    """
    Dataset for loading CLIP features with variable length.
    """
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.file_paths = []
        
        # Recursively find all .pt files
        for root, _, files in os.walk(feature_dir):
            for file in files:
                if file.endswith('.pt'):
                    self.file_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(self.file_paths)} feature files in {feature_dir}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)
        
        features = data['features']
        metadata = data['metadata']
        
        # Return features with metadata for flexible processing
        return {
            'features': features,
            'video_id': metadata['video_id'],
            'timestamps': metadata.get('timestamps', []),
            'num_frames': metadata['num_frames'],
            'file_path': file_path
        }

def collate_fn(batch):
    """
    Custom collate function for variable length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Get max sequence length in the batch
    max_len = max([item['features'].shape[0] for item in batch])
    
    # Initialize padded features tensor
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[1]
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # Fill in data and masks
    video_ids = []
    file_paths = []
    timestamps = []
    
    for i, item in enumerate(batch):
        features = item['features']
        seq_len = features.shape[0]
        
        # Add features and update mask
        padded_features[i, :seq_len] = features
        attention_mask[i, :seq_len] = 1
        
        # Store metadata
        video_ids.append(item['video_id'])
        file_paths.append(item['file_path'])
        timestamps.append(item['timestamps'])
    
    return {
        'features': padded_features,
        'attention_mask': attention_mask,
        'video_ids': video_ids,
        'file_paths': file_paths,
        'timestamps': timestamps
    }
class TimeSformerDividedAttention(nn.Module):
    """
    Divided space-time attention mechanism for TimeSformer.
    """
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
        
        return x

class RelativePositionalEncoding(nn.Module):
    """
    Flexible relative positional encoding for temporal awareness.
    """
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
        """
        Apply positional encoding based on timestamps or sequence position.
        
        Args:
            x: Input tensor [B, N, D]
            timestamps: Optional list of timestamp lists for each video
            
        Returns:
            Tensor with same shape as input
        """
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
    """
    TimeSformer block with divided space-time attention.
    """
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
        x = x + self.drop_path(self.attn(self.norm1(x), attention_mask))
        
        # Optional cross-frame attention
        if self.cross_frame:
            x = x + self.drop_path(self.cross_attn(self.cross_norm(x), attention_mask))
            
        # MLP block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class HierarchicalAttentionPooling(nn.Module):
    """
    Hierarchical attention pooling with step-aware aggregation.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        
    def forward(self, x, attention_mask=None, step_boundaries=None):
        """
        Apply hierarchical attention pooling.
        
        Args:
            x: Input tensor [B, N, D]
            attention_mask: Boolean mask for attention
            step_boundaries: Optional list of step timestamp boundaries
            
        Returns:
            Pooled tensor [B, D]
        """
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
    """
    TimeSformer model for temporal modeling of CLIP features.
    """
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
    
    def forward(self, x, attention_mask=None, timestamps=None, step_boundaries=None):
        """
        Forward pass through the TimeSformer.
        
        Args:
            x: Input features [B, N, input_dim]
            attention_mask: Boolean attention mask [B, N]
            timestamps: Optional list of timestamps for each frame
            step_boundaries: Optional list of step timestamp boundaries
            
        Returns:
            Tensor of shape [B, output_dim]
        """
        # Project input features
        x = self.input_proj(x)
        x = self.norm_in(x)
        
        # Add positional encoding
        x = self.pos_encoding(x, timestamps)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Apply pooling
        x = self.pooling(x, attention_mask, step_boundaries)
        x = self.norm_out(x)
        
        return x
    
def train_epoch(model, data_loader, optimizer, criterion, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        # Move data to device
        features = batch['features'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        timestamps = batch['timestamps']
        
        # Clear previous gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision context
        with torch.cuda.amp.autocast():
            # Get temporal embeddings
            embeddings = model(features, attention_mask, timestamps)
            
            # Use the output projection to map back to feature space
            next_frame_target = features[:, 1:, :]
            
            # Create predictions using output projection layer
            pred_features = model.output_proj(embeddings).unsqueeze(1).expand(-1, features.size(1)-1, -1)
            
            # Compute loss
            loss = criterion(pred_features, next_frame_target)
        
        # Scaled gradient update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': loss.item(),
            'memory_allocated': f'{torch.cuda.memory_allocated(device)/1e9:.2f}G',
            'max_memory': f'{torch.cuda.max_memory_allocated(device)/1e9:.2f}G'
        })
    
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    """Validate the model with mixed precision."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validation'):
            features = batch['features'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            timestamps = batch['timestamps']
            
            # Mixed precision context
            with torch.cuda.amp.autocast():
                # Get temporal embeddings
                embeddings = model(features, attention_mask, timestamps)
                
                # Use the output projection to map back to feature space
                next_frame_target = features[:, 1:, :]
                pred_features = model.output_proj(embeddings).unsqueeze(1).expand(-1, features.size(1)-1, -1)
                
                # Compute loss
                loss = criterion(pred_features, next_frame_target)
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def extract_features(model, data_loader, output_dir, device):
    """Extract temporal features using the trained model."""
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting temporal features'):
            features = batch['features'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            file_paths = batch['file_paths']
            video_ids = batch['video_ids']
            timestamps = batch['timestamps']
            
            # Get temporal embeddings
            embeddings = model(features, attention_mask, timestamps)
            
            # Save embeddings
            for i, (video_id, file_path) in enumerate(zip(video_ids, file_paths)):
                # Load original file to get metadata
                original_data = torch.load(file_path)
                metadata = original_data['metadata']
                
                # Create output path preserving directory structure
                rel_path = os.path.relpath(file_path, args.feature_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save temporal features with metadata
                torch.save({
                    'features': embeddings[i].cpu(),
                    'metadata': metadata
                }, output_path)

def main(args):
    """Main function for TimeSformer training and feature extraction."""
    # Set environment for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Explicitly set GPU
    torch.cuda.set_device(1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset = FeatureDataset(os.path.join(args.feature_dir, 'train'))
    val_dataset = FeatureDataset(os.path.join(args.feature_dir, 'validation'))
    test_dataset = FeatureDataset(os.path.join(args.feature_dir, 'test'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Reduced from 16 to 8
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,  # Consistent batch size reduction
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model with reduced complexity
    model = TimeSformer(
        input_dim=512,  # CLIP feature dimension
        output_dim=512,  # Reduced from 768
        depth=6,  # Reduced from 8
        num_heads=4,  # Reduced from 8
        drop_rate=0.1,
        attn_drop_rate=0.1,
        max_seq_len=512  # Reduced max sequence length
    ).to(device)
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Initialize early stopping
    early_stopping = AdaptiveEarlyStopping(
        initial_patience=args.initial_patience,
        max_patience=args.max_patience,
        patience_increase=args.patience_increase
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.max_epochs):
        logger.info(f"Epoch {epoch+1}/{args.max_epochs}")
        
        # Clear GPU cache before each epoch
        torch.cuda.empty_cache()
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Increase patience if learning rate decreased
        if new_lr < old_lr:
            early_stopping.increase_patience()
            writer.add_scalar('Patience', early_stopping.patience, epoch)
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.model_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping': early_stopping.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Best model saved at epoch {epoch+1}")
        
        # Check early stopping
        if early_stopping(val_loss, epoch):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Calculate training time
    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time/60:.2f} minutes")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model for feature extraction
    checkpoint = torch.load(os.path.join(args.model_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Extract features for all splits
    logger.info("Extracting temporal features for train split")
    extract_features(model, train_loader, os.path.join(args.output_dir, 'train'), device)
    
    logger.info("Extracting temporal features for validation split")
    extract_features(model, val_loader, os.path.join(args.output_dir, 'validation'), device)
    
    logger.info("Extracting temporal features for test split")
    extract_features(model, test_loader, os.path.join(args.output_dir, 'test'), device)
    
    logger.info("Feature extraction complete!")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TimeSformer for Temporal Modeling")
    
    # Data paths
    parser.add_argument('--feature_dir', type=str, default='Raw_Dataset/features', 
                        help='Directory containing CLIP features')
    parser.add_argument('--output_dir', type=str, default='Raw_Dataset/temporal_features', 
                        help='Directory to save temporal features')
    parser.add_argument('--model_dir', type=str, default='Raw_Dataset/models/timesformer', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs/timesformer_training', 
                        help='Directory for TensorBoard logs')
    
    # Model hyperparameters
    parser.add_argument('--num_layers', type=int, default=6, 
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument('--drop_rate', type=float, default=0.1, 
                        help='Dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.1, 
                        help='Attention dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=512, 
                        help='Maximum sequence length')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size (reduced for memory efficiency)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=150, 
                        help='Maximum number of training epochs')
    
    # Early stopping parameters
    parser.add_argument('--initial_patience', type=int, default=8, 
                        help='Initial patience for early stopping')
    parser.add_argument('--max_patience', type=int, default=20, 
                        help='Maximum patience for early stopping')
    parser.add_argument('--patience_increase', type=int, default=2, 
                        help='Patience increase after learning rate reduction')
    
    # Data loading parameters
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of worker threads for data loading')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    logger.info("TimeSformer Configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Run main function
    try:
        main(args)
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise