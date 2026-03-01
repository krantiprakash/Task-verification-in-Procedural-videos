import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk

# Make sure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Import utilities from utils.py
from utils import (
    setup_logging, set_seed, load_temporal_features, get_split_files,
    load_split_info, get_t5_tokenizer, get_t5_model,
    evaluate_answer_generation, log_generated_answers,
    save_model, load_model, get_learning_rate_scheduler,
    plot_training_curves, create_experiment_dir, save_config,
    set_device, count_parameters, AdaptiveEarlyStopping
)

# Set up logging with a component-specific name
logger = setup_logging(log_file='t5_answer_generation.log', logger_name='t5_answer_generation')

class VideoQADataset(Dataset):
    """
    Dataset for procedural video question answering with T5.
    """
    def __init__(self, features_dir, cross_modal_model_dir, annotation_file, 
                 split='train', t5_tokenizer=None, max_seq_len=512):
        """
        Initialize the dataset.
        
        Args:
            features_dir: Directory with temporal features
            cross_modal_model_dir: Directory with cross-modal fusion model
            annotation_file: Path to annotations file
            split: Data split (train, validation, test)
            t5_tokenizer: T5 tokenizer for questions/answers
            max_seq_len: Maximum sequence length
        """
        self.features_dir = features_dir
        self.cross_modal_model_dir = cross_modal_model_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.t5_tokenizer = t5_tokenizer
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Extract videos and QA pairs for this split
        self.samples = []
        
        if 'database' in data:
            # ActNet format
            for video_id, video_data in data['database'].items():
                if 'verification_data' in video_data:
                    for qa_pair in video_data['verification_data']:
                        self.samples.append({
                            'video_id': video_id,
                            'question': qa_pair['question'],
                            'answer': qa_pair['answer']
                        })
        else:
            # Direct format
            for video_id, video_data in data.items():
                if 'verification_data' in video_data:
                    for qa_pair in video_data['verification_data']:
                        self.samples.append({
                            'video_id': video_id,
                            'question': qa_pair['question'],
                            'answer': qa_pair['answer']
                        })
        
        logger.info(f"Loaded {len(self.samples)} QA pairs for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with features, question, and answer."""
        sample = self.samples[idx]
        video_id = sample['video_id']
        question = sample['question']
        answer = sample['answer']
        
        # Find the feature file path
        feature_path = None
        split_dir = os.path.join(self.features_dir, self.split)
        
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith('.pt') and video_id in file:
                    feature_path = os.path.join(root, file)
                    break
            if feature_path:
                break
        
        if feature_path:
            try:
                # Load temporal features
                features_data = torch.load(feature_path)
                video_features = features_data['features']
            except Exception as e:
                logger.error(f"Error loading features for {video_id}: {e}")
                # Return placeholder tensor if loading fails
                video_features = torch.zeros(512)  # Match dimensionality of features
        else:
            logger.warning(f"Feature file for {video_id} not found")
            video_features = torch.zeros(512)  # Match dimensionality of features
        
        # Tokenize question and answer
        if self.t5_tokenizer:
            # For T5, prefix the question with "question: "
            question_text = f"question: {question}"
            
            question_tokens = self.t5_tokenizer(
                question_text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            answer_tokens = self.t5_tokenizer(
                answer,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'video_id': video_id,
                'features': video_features,
                'question': question,
                'answer': answer,
                'question_input_ids': question_tokens.input_ids.squeeze(),
                'question_attention_mask': question_tokens.attention_mask.squeeze(),
                'answer_input_ids': answer_tokens.input_ids.squeeze(),
                'answer_attention_mask': answer_tokens.attention_mask.squeeze()
            }
        else:
            return {
                'video_id': video_id,
                'features': video_features,
                'question': question,
                'answer': answer
            }


def collate_fn(batch):
    """
    Custom collate function for variable length sequences.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Extract fields
    video_ids = [item['video_id'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    # Stack features, handling different shapes
    features = torch.stack([item['features'] for item in batch])
    
    # Stack token IDs and attention masks
    question_input_ids = torch.stack([item['question_input_ids'] for item in batch])
    question_attention_mask = torch.stack([item['question_attention_mask'] for item in batch])
    answer_input_ids = torch.stack([item['answer_input_ids'] for item in batch])
    answer_attention_mask = torch.stack([item['answer_attention_mask'] for item in batch])
    
    return {
        'video_ids': video_ids,
        'features': features,
        'questions': questions,
        'answers': answers,
        'question_input_ids': question_input_ids,
        'question_attention_mask': question_attention_mask,
        'answer_input_ids': answer_input_ids,
        'answer_attention_mask': answer_attention_mask
    }


class VideoQAT5Model(nn.Module):
    """
    T5-based model for video question answering.
    """
    def __init__(self, cross_modal_model, text_encoder, t5_model, prompt_dim=512, num_prompt_tokens=24):
        super().__init__()
        
        self.cross_modal_model = cross_modal_model
        self.text_encoder = text_encoder
        self.t5_model = t5_model
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_dim = prompt_dim
        
        # Make sure cross_modal_model and text_encoder are in eval mode and don't update
        self.cross_modal_model.eval()
        self.text_encoder.eval()
        for param in self.cross_modal_model.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Project prompt tokens to T5 embedding space
        self.prompt_projection = nn.Linear(prompt_dim, self.t5_model.config.d_model)
    
    def forward(self, video_features, question_input_ids, question_attention_mask, 
                answer_input_ids=None, video_masks=None, generate=False):
        """
        Forward pass through the model.
        
        Args:
            video_features: Video features [batch_size, feature_dim]
            question_input_ids: Question token IDs [batch_size, seq_len]
            question_attention_mask: Question attention mask [batch_size, seq_len]
            answer_input_ids: Answer token IDs for training [batch_size, seq_len]
            video_masks: Optional masks for video features
            generate: Whether to generate answers or calculate loss
            
        Returns:
            Loss during training, generated IDs during inference
        """
        batch_size = video_features.shape[0]
        
        # Reshape video features for the cross-modal model if needed
        # (the cross-modal model expects 3D features: [batch_size, seq_len, feature_dim])
        if len(video_features.shape) == 2:
            # If we have flat features, reshape to single-frame features
            video_features = video_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
            
            # Create all-ones mask for the single frame
            if video_masks is None:
                video_masks = torch.ones(batch_size, 1, dtype=torch.bool, device=video_features.device)
        
        # Create dummy text features (the cross-modal model expects text features)
        # Since we've frozen the text_encoder, we don't need to process real text here
        # We just need to match the expected shape for the model
        text_features = torch.zeros(
            batch_size, 1, self.cross_modal_model.hidden_dim, 
            device=video_features.device
        )
        text_masks = torch.ones(batch_size, 1, dtype=torch.bool, device=video_features.device)
        
        # Forward pass through cross-modal model to get prompt tokens
        with torch.no_grad():
            prompt_tokens = self.cross_modal_model(
                video_features, text_features, video_masks, text_masks
            )
        
        # Project prompt tokens to T5 embedding space
        t5_prompt_embeddings = self.prompt_projection(prompt_tokens)
        
        # Prepare T5 inputs
        # Get the embedding layer from T5
        t5_word_embeddings = self.t5_model.get_input_embeddings()
        
        # Embed the question input ids
        question_embeddings = t5_word_embeddings(question_input_ids)
        
        # Combine prompt embeddings with question embeddings
        # [batch_size, prompt_len + question_len, d_model]
        combined_embeddings = torch.cat([t5_prompt_embeddings, question_embeddings], dim=1)
        
        # Extend the attention mask for the prompt tokens
        prompt_attention_mask = torch.ones(
            batch_size, self.num_prompt_tokens, 
            dtype=question_attention_mask.dtype, 
            device=question_attention_mask.device
        )
        extended_attention_mask = torch.cat([prompt_attention_mask, question_attention_mask], dim=1)
        
        # Process through T5
        if not generate:
            # Training mode - calculate loss
            outputs = self.t5_model(
                inputs_embeds=combined_embeddings,
                attention_mask=extended_attention_mask,
                labels=answer_input_ids,
                return_dict=True
            )
            return outputs.loss
        else:
            # Inference mode - generate answer
            generated_ids = self.t5_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=extended_attention_mask,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
            return generated_ids


def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: VideoQAT5Model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        average_loss: Average loss for this epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} (train)')
    
    for batch in progress_bar:
        # Move data to device
        video_features = batch['features'].to(device)
        question_input_ids = batch['question_input_ids'].to(device)
        question_attention_mask = batch['question_attention_mask'].to(device)
        answer_input_ids = batch['answer_input_ids'].to(device)
        
        # Forward pass
        loss = model(
            video_features,
            question_input_ids,
            question_attention_mask,
            answer_input_ids
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    average_loss = total_loss / num_batches
    return average_loss


def validate(model, dataloader, device, epoch):
    """
    Validate the model.
    
    Args:
        model: VideoQAT5Model
        dataloader: Validation dataloader
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        average_loss: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} (val)')
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            video_features = batch['features'].to(device)
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            answer_input_ids = batch['answer_input_ids'].to(device)
            
            # Forward pass
            loss = model(
                video_features,
                question_input_ids,
                question_attention_mask,
                answer_input_ids
            )
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    average_loss = total_loss / num_batches
    return average_loss


def generate_answers(model, dataloader, tokenizer, device, num_samples=None):
    """
    Generate answers for evaluation.
    
    Args:
        model: VideoQAT5Model
        dataloader: Dataloader
        tokenizer: T5 tokenizer
        device: Device to run on
        num_samples: Optional limit on number of samples
        
    Returns:
        Dictionary with generated answers and ground truth
    """
    model.eval()
    
    video_ids = []
    questions = []
    reference_answers = []
    generated_answers = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating answers'):
            # Move data to device
            video_features = batch['features'].to(device)
            question_input_ids = batch['question_input_ids'].to(device)
            question_attention_mask = batch['question_attention_mask'].to(device)
            
            # Generate answers
            generated_ids = model(
                video_features,
                question_input_ids,
                question_attention_mask,
                generate=True
            )
            
            # Decode the generated ids
            decoded_answers = [
                tokenizer.decode(g, skip_special_tokens=True) 
                for g in generated_ids
            ]
            
            # Store results
            video_ids.extend(batch['video_ids'])
            questions.extend(batch['questions'])
            reference_answers.extend(batch['answers'])
            generated_answers.extend(decoded_answers)
            
            # Limit number of samples if specified
            if num_samples and len(video_ids) >= num_samples:
                video_ids = video_ids[:num_samples]
                questions = questions[:num_samples]
                reference_answers = reference_answers[:num_samples]
                generated_answers = generated_answers[:num_samples]
                break
    
    return {
        'video_ids': video_ids,
        'questions': questions,
        'reference_answers': reference_answers,
        'generated_answers': generated_answers
    }


def evaluate_model(model, dataloader, tokenizer, device, output_dir):
    """
    Evaluate the model and save results.
    
    Args:
        model: VideoQAT5Model
        dataloader: Test dataloader
        tokenizer: T5 tokenizer
        device: Device to run on
        output_dir: Directory to save results
        
    Returns:
        metrics: Evaluation metrics
    """
    # Generate answers
    results = generate_answers(model, dataloader, tokenizer, device)
    
    # Evaluate with metrics
    metrics = evaluate_answer_generation(
        results['reference_answers'],
        results['generated_answers'],
        logger=logger
    )
    
    # Log detailed results with metrics
    output_file = os.path.join(output_dir, 'generated_answers.txt')
    log_generated_answers(
        results['questions'],
        results['reference_answers'],
        results['generated_answers'],
        output_file,
        logger=logger
    )
    
    # Save metrics as JSON
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print main metrics
    logger.info("Evaluation Metrics:")
    logger.info(f"BLEU: {metrics['bleu']:.4f}")
    logger.info(f"ROUGE-L: {metrics['rougeL']:.4f}")
    logger.info(f"METEOR: {metrics['meteor']:.4f}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    return metrics


def load_cross_modal_model(model_path, device):
    """
    Load the cross-modal fusion model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (cross_modal_model, text_encoder)
    """
    from cross_model_fusion import CrossModalFusionModel, StepDescriptionEncoder
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create models
    cross_modal_model = CrossModalFusionModel(
        video_dim=config['video_dim'],
        text_dim=config['hidden_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_prompt_tokens=config['num_prompt_tokens'],
        prompt_dim=config['prompt_dim']
    ).to(device)
    
    # Load T5 tokenizer to create text encoder
    t5_tokenizer = T5Tokenizer.from_pretrained(config['t5_model_name'])
    
    text_encoder = StepDescriptionEncoder(
        t5_tokenizer,
        hidden_dim=config['hidden_dim'],
        num_layers=config['text_encoder_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    
    # Load weights
    cross_modal_model.load_state_dict(checkpoint['model_state_dict'])
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    
    # Set to eval mode and freeze
    cross_modal_model.eval()
    text_encoder.eval()
    
    logger.info(f"Loaded cross-modal fusion model from {model_path}")
    return cross_modal_model, text_encoder


def main(args):
    """
    Main function for T5 answer generation training.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Set device - pass the logger
    device = set_device(args.device_id, logger=logger)
    
    # Create or get experiment directory
    if args.eval_only and args.experiment_name:
        # If evaluating only, use the existing experiment directory
        exp_dir = os.path.join(args.output_dir, args.experiment_name)
        if not os.path.exists(exp_dir):
            logger.error(f"Experiment directory {exp_dir} does not exist")
            return
        logger.info(f"Using existing experiment directory: {exp_dir}")
    else:
        # Create a new experiment directory
        exp_dir = create_experiment_dir(args.output_dir, args.experiment_name, logger=logger)
        
        # Save configuration - pass the logger
        config = vars(args)
        save_config(config, os.path.join(exp_dir, 'config.json'), logger=logger)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
    
    # Initialize T5 tokenizer and model - pass the logger
    t5_tokenizer = get_t5_tokenizer(args.t5_model_name, logger=logger)
    t5_model = get_t5_model(args.t5_model_name, logger=logger).to(device)
    
    # Load cross-modal fusion model
    cross_modal_model, text_encoder = load_cross_modal_model(
        args.cross_modal_checkpoint, device
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    # We always need the test dataset for evaluation
    test_dataset = VideoQADataset(
        args.features_dir, 
        args.cross_modal_dir,
        args.test_annotation_file,
        split="test",
        t5_tokenizer=t5_tokenizer,
        max_seq_len=args.max_text_length
    )
    
    # Only create train and validation datasets if not in eval-only mode
    if not args.eval_only:
        train_dataset = VideoQADataset(
            args.features_dir, 
            args.cross_modal_dir,
            args.train_annotation_file,
            split="train",
            t5_tokenizer=t5_tokenizer,
            max_seq_len=args.max_text_length
        )
        
        val_dataset = VideoQADataset(
            args.features_dir, 
            args.cross_modal_dir,
            args.val_annotation_file,
            split="validation",
            t5_tokenizer=t5_tokenizer,
            max_seq_len=args.max_text_length
        )
    
        # Create train and validation dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create the combined model
    model = VideoQAT5Model(
        cross_modal_model=cross_modal_model,
        text_encoder=text_encoder,
        t5_model=t5_model,
        prompt_dim=args.prompt_dim,
        num_prompt_tokens=args.num_prompt_tokens
    ).to(device)
    
    # Log model sizes
    t5_params = count_parameters(model.t5_model)
    projection_params = count_parameters(model.prompt_projection)
    total_params = count_parameters(model)
    
    logger.info(f"T5 parameters: {t5_params:,}")
    logger.info(f"Prompt projection parameters: {projection_params:,}")
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    if args.eval_only:
        # Skip training and just load the best model for evaluation
        logger.info("Skipping training, loading best model for evaluation only")
        best_model_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pt')
        
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            best_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', 0)
            logger.info(f"Loaded best model from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
            
            # Run evaluation
            logger.info("Evaluating on test set...")
            eval_dir = os.path.join(exp_dir, 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            try:
                metrics = evaluate_model(model, test_loader, t5_tokenizer, device, eval_dir)
                logger.info(f"Evaluation complete. Results saved to {eval_dir}")
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.error(f"Cannot find best model at {best_model_path}")
            logger.error("Please specify the correct experiment name with --experiment_name")
            
        writer.close()
        return
    
    # If not eval_only, continue with training
    
    # Create optimizer - only optimize T5 and projection parameters
    optimizer = optim.AdamW(
        [
            {'params': model.t5_model.parameters()},
            {'params': model.prompt_projection.parameters()}
        ],
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
    
    # Training loop
    logger.info("Starting training...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        logger.info(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'early_stopping' in checkpoint:
            early_stopping.load_state_dict(checkpoint['early_stopping'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.4f}")
    
    for epoch in range(start_epoch, args.max_epochs):
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate(
            model, val_loader, device, epoch
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
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'early_stopping': early_stopping.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
        
        # Generate sample answers periodically
        if (epoch + 1) % args.evaluate_every == 0:
            # Use a small subset of validation data for periodic evaluation
            sample_results = generate_answers(
                model, val_loader, t5_tokenizer, device, num_samples=args.eval_samples
            )
            
            # Create a simple comparison table
            comparison_file = os.path.join(exp_dir, 'results', f'sample_answers_epoch_{epoch+1}.txt')
            os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
            
            with open(comparison_file, 'w') as f:
                for i in range(len(sample_results['questions'])):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"Question: {sample_results['questions'][i]}\n")
                    f.write(f"Reference: {sample_results['reference_answers'][i]}\n")
                    f.write(f"Generated: {sample_results['generated_answers'][i]}\n\n")
            
            logger.info(f"Generated sample answers saved to {comparison_file}")
            
            # Plot training curves - pass the logger
            plot_training_curves(
                train_losses, val_losses,
                output_dir=os.path.join(exp_dir, 'visualizations'),
                logger=logger
            )
        
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
        
        best_epoch = checkpoint['epoch']
        logger.info(f"Loaded best model from epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    eval_dir = os.path.join(exp_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    try:
        metrics = evaluate_model(model, test_loader, t5_tokenizer, device, eval_dir)
        
        # Plot final training curves
        plot_training_curves(
            train_losses, val_losses,
            output_dir=os.path.join(exp_dir, 'visualizations'),
            logger=logger
        )
        
        # Save final model
        final_model_path = os.path.join(exp_dir, 'checkpoints', 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics
        }, final_model_path)
        
        logger.info(f"Training completed. Final model saved to {final_model_path}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    writer.close()

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="T5 Answer Generation Training")
   
   # Data paths
   parser.add_argument('--features_dir', type=str, default='Raw_Dataset/temporal_features',
                      help='Directory containing temporal features')
   parser.add_argument('--cross_modal_dir', type=str, default='experiments/cross_modal',
                      help='Directory containing cross-modal fusion model')
   parser.add_argument('--cross_modal_checkpoint', type=str, 
                      default='experiments/cross_modal/20250312-174838/checkpoints/best_model.pt',
                      help='Path to cross-modal fusion model checkpoint')
   parser.add_argument('--train_annotation_file', type=str, default='Raw_Dataset/splits/train.json',
                      help='Path to training annotation file')
   parser.add_argument('--val_annotation_file', type=str, default='Raw_Dataset/splits/validation.json',
                      help='Path to validation annotation file')
   parser.add_argument('--test_annotation_file', type=str, default='Raw_Dataset/splits/test.json',
                      help='Path to test annotation file')
   parser.add_argument('--output_dir', type=str, default='experiments/t5_qa',
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
                      help='T5 model name')
   parser.add_argument('--num_prompt_tokens', type=int, default=24,
                      help='Number of prompt tokens from cross-modal fusion')
   parser.add_argument('--prompt_dim', type=int, default=512,
                      help='Dimension of prompt tokens')
   parser.add_argument('--max_text_length', type=int, default=128,
                      help='Maximum text sequence length')
   
   # Training parameters
   parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
   parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
   parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
   parser.add_argument('--max_epochs', type=int, default=100,
                      help='Maximum number of epochs')
   
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
   
   # Checkpointing and evaluation
   parser.add_argument('--resume_checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume from')
   parser.add_argument('--save_every', type=int, default=5,
                      help='Save checkpoint every N epochs')
   parser.add_argument('--evaluate_every', type=int, default=5,
                      help='Generate and evaluate sample answers every N epochs')
   parser.add_argument('--eval_samples', type=int, default=10,
                      help='Number of samples to evaluate during training')
   parser.add_argument('--eval_only', action='store_true',
                      help='Skip training and only run evaluation on the test set')
   
   args = parser.parse_args()
   
   main(args)