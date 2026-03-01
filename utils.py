import os
import json
import time
import torch
import numpy as np
import pandas as pd
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import torch.nn.functional as F

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Also download wordnet for METEOR score
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up logging
def setup_logging(log_file='procedural_video_qa.log', console_level=logging.INFO, logger_name=None):
    """Set up logging configuration."""
    # Create directory for log file if it doesn't exist
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    
    # Configure root logger for general settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Create logger with specific name if provided
    logger = logging.getLogger(logger_name or __name__)
    return logger

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# ---------------------------
# Data Loading Functions
# ---------------------------

def load_temporal_features(features_dir, video_id, split='train', logger=None):
    """
    Load pre-computed TimeSformer temporal features for a video.
    
    Args:
        features_dir: Base directory for features
        video_id: ID of the video
        split: Data split (train, validation, test)
        logger: Optional logger instance
        
    Returns:
        Dictionary with features and metadata
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    # Construct path to feature file
    feature_path = find_feature_file(features_dir, video_id, split, logger=_logger)
    
    if not feature_path:
        _logger.warning(f"Feature file for video {video_id} not found in {split} split")
        return None
    
    try:
        # Load features
        features_data = torch.load(feature_path)
        return features_data
    except Exception as e:
        _logger.error(f"Error loading features for {video_id}: {e}")
        return None

def find_feature_file(base_dir, video_id, split, logger=None):
    """
    Find a feature file by recursively searching in the directory.
    
    Args:
        base_dir: Base directory to search in
        video_id: Video ID to find
        split: Data split (train, validation, test)
        logger: Optional logger instance
        
    Returns:
        Path to the feature file or None if not found
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    # First try the expected path
    split_dir = os.path.join(base_dir, split)
    
    # Walk through directory
    for root, _, files in os.walk(split_dir):
        for file in files:
            if file.endswith('.pt'):
                # Check if this is the right video
                if video_id in file:
                    return os.path.join(root, file)
    
    return None

def load_video_annotations(annotations_file, video_id, logger=None):
    """
    Load step annotations for a specific video.
    
    Args:
        annotations_file: Path to annotations JSON file
        video_id: Video ID to load annotations for
        logger: Optional logger instance
        
    Returns:
        Dictionary with step annotations and video metadata
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Check if video exists in annotations
        if video_id in annotations:
            return annotations[video_id]
        elif 'database' in annotations and video_id in annotations['database']:
            return annotations['database'][video_id]
        else:
            _logger.warning(f"Video {video_id} not found in annotations file")
            return None
    except Exception as e:
        _logger.error(f"Error loading annotations for {video_id}: {e}")
        return None

def load_question_answers(qa_file, video_id=None, logger=None):
    """
    Load question-answer pairs for videos.
    
    Args:
        qa_file: Path to QA JSON file
        video_id: Optional specific video ID to load QA pairs for
        logger: Optional logger instance
        
    Returns:
        Dictionary mapping video IDs to lists of QA pairs
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    try:
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        if video_id:
            # Return QA pairs for specific video if it exists
            if video_id in qa_data:
                return {video_id: qa_data[video_id]}
            else:
                _logger.warning(f"Video {video_id} not found in QA file")
                return {}
        else:
            # Return all QA pairs
            return qa_data
    except Exception as e:
        _logger.error(f"Error loading QA data: {e}")
        return {}

def get_split_files(split_dir):
    """
    Get paths to split files.
    
    Args:
        split_dir: Directory containing split files
        
    Returns:
        Dictionary with paths to split files
    """
    return {
        'train': os.path.join(split_dir, 'train.json'),
        'validation': os.path.join(split_dir, 'validation.json'),
        'test': os.path.join(split_dir, 'test.json'),
        'info': os.path.join(split_dir, 'split_info.json')
    }

def load_split_info(split_info_path, logger=None):
    """
    Load split information from JSON file.
    
    Args:
        split_info_path: Path to split info JSON file
        logger: Optional logger instance
        
    Returns:
        Dictionary with split information
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    try:
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
        return split_info
    except Exception as e:
        _logger.error(f"Error loading split info: {e}")
        return None

# ---------------------------
# Dataset Classes
# ---------------------------

class ProceduralVideoDataset(Dataset):
    """
    Dataset for procedural video question answering.
    """
    def __init__(self, features_dir, annotation_file, qa_file, split='train', max_seq_len=512, t5_tokenizer=None, logger=None):
        """
        Initialize the dataset.
        
        Args:
            features_dir: Directory with temporal features
            annotation_file: Path to annotations file
            qa_file: Path to QA file
            split: Data split (train, validation, test)
            max_seq_len: Maximum sequence length
            t5_tokenizer: Optional T5 tokenizer for questions/answers
            logger: Optional logger instance
        """
        self.features_dir = features_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.t5_tokenizer = t5_tokenizer
        
        # Use provided logger or get a default one
        self.logger = logger or logging.getLogger()
        
        # Load annotations and QA data
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        with open(qa_file, 'r') as f:
            self.qa_data = json.load(f)
        
        # Get video IDs for this split
        if 'database' in self.annotations:
            self.video_ids = [
                vid for vid, data in self.annotations['database'].items()
                if data.get('subset', split) == split and vid in self.qa_data
            ]
        else:
            self.video_ids = [
                vid for vid in self.annotations.keys()
                if vid in self.qa_data
            ]
        
        self.logger.info(f"Loaded {len(self.video_ids)} videos for {split} split")
        
        # Create QA samples
        self.samples = []
        for vid in self.video_ids:
            if vid in self.qa_data:
                for qa_pair in self.qa_data[vid].get('verification_data', []):
                    self.samples.append({
                        'video_id': vid,
                        'question': qa_pair['question'],
                        'answer': qa_pair['answer']
                    })
        
        self.logger.info(f"Created {len(self.samples)} QA samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with features, question, and answer."""
        sample = self.samples[idx]
        video_id = sample['video_id']
        
        # Load temporal features
        features_data = load_temporal_features(self.features_dir, video_id, self.split, logger=self.logger)
        
        if features_data is None:
            # Return dummy tensor if features not found
            features = torch.zeros(1, 512)  # Adjust size as needed
            self.logger.warning(f"Using dummy features for {video_id}")
        else:
            features = features_data['features']
        
        # Get step annotations
        if 'database' in self.annotations:
            video_anno = self.annotations['database'].get(video_id, {})
        else:
            video_anno = self.annotations.get(video_id, {})
            
        steps = []
        for step in video_anno.get('annotation', []):
            if 'segment' in step and 'label' in step:
                steps.append({
                    'start': step['segment'][0],
                    'end': step['segment'][1],
                    'label': step['label']
                })
        
        # Tokenize text if tokenizer provided
        question = sample['question']
        answer = sample['answer']
        
        if self.t5_tokenizer:
            question_tokens = self.t5_tokenizer(
                question,
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
                'features': features,
                'steps': steps,
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
                'features': features,
                'steps': steps,
                'question': question,
                'answer': answer
            }

def create_dataloaders(dataset_args, batch_size=16, num_workers=4, t5_tokenizer=None, logger=None):
    """
    Create dataloaders for all splits.
    
    Args:
        dataset_args: Dictionary with dataset arguments
        batch_size: Batch size
        num_workers: Number of worker threads
        t5_tokenizer: Optional T5 tokenizer
        logger: Optional logger instance
        
    Returns:
        Dictionary with dataloaders for each split
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    dataloaders = {}
    
    for split in ['train', 'validation', 'test']:
        dataset = ProceduralVideoDataset(
            **dataset_args,
            split=split,
            t5_tokenizer=t5_tokenizer,
            logger=_logger
        )
        
        shuffle = (split == 'train')
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders

# ---------------------------
# Text Processing Utilities
# ---------------------------

def get_t5_tokenizer(model_name='t5-base', logger=None):
    """
    Initialize T5 tokenizer.
    
    Args:
        model_name: T5 model name
        logger: Optional logger instance
        
    Returns:
        T5 tokenizer
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        _logger.info(f"Loaded T5 tokenizer: {model_name}")
        return tokenizer
    except Exception as e:
        _logger.error(f"Error loading T5 tokenizer: {e}")
        return None

def get_t5_model(model_name='t5-base', logger=None):
    """
    Initialize T5 model.
    
    Args:
        model_name: T5 model name
        logger: Optional logger instance
        
    Returns:
        T5 model
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        _logger.info(f"Loaded T5 model: {model_name}")
        return model
    except Exception as e:
        _logger.error(f"Error loading T5 model: {e}")
        return None

def encode_text_descriptions(tokenizer, descriptions, max_length=128):
    """
    Encode text descriptions using tokenizer.
    
    Args:
        tokenizer: Tokenizer to use
        descriptions: List of text descriptions
        max_length: Maximum sequence length
        
    Returns:
        Tensor of token IDs and attention mask
    """
    encoded = tokenizer(
        descriptions,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encoded.input_ids, encoded.attention_mask

# ---------------------------
# Model Utilities
# ---------------------------

def save_model(model, optimizer, epoch, loss, path, logger=None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        logger: Optional logger instance
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    
    _logger.info(f"Model saved to {path}")

def load_model(model, path, optimizer=None, device='cuda', logger=None):
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        path: Path to checkpoint
        optimizer: Optional optimizer to load state
        device: Device to load model on
        logger: Optional logger instance
        
    Returns:
        Dictionary with checkpoint data
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    if not os.path.exists(path):
        _logger.error(f"Checkpoint not found: {path}")
        return None
    
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        _logger.info(f"Loaded model from {path} (epoch {checkpoint['epoch']})")
        return checkpoint
    except Exception as e:
        _logger.error(f"Error loading model: {e}")
        return None

def freeze_model_parameters(model):
    """
    Freeze all parameters of a model.
    
    Args:
        model: Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    
    logging.getLogger().info("Model parameters frozen")

def get_learning_rate_scheduler(optimizer, scheduler_type='reduce_on_plateau', **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 10),
            eta_min=kwargs.get('eta_min', 0)
        )
    else:
        logging.getLogger().warning(f"Unknown scheduler type: {scheduler_type}")
        return None

# ---------------------------
# Evaluation Metrics
# ---------------------------

def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score.
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Generated text
        
    Returns:
        BLEU score
    """
    if not hypothesis:
        return 0.0
    
    try:
        # Tokenize using nltk.word_tokenize directly instead of imported word_tokenize
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Handle edge cases
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Create reference list format required by NLTK
        references = [ref_tokens]
        
        # Calculate BLEU with smoothing
        smoothie = SmoothingFunction().method1
        score = sentence_bleu(references, hyp_tokens, smoothing_function=smoothie)
        return score
    except Exception as e:
        logging.getLogger().error(f"Error calculating BLEU: {e}")
        return 0.0

def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores.
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        
    Returns:
        Dictionary with ROUGE scores
    """
    if not hypothesis:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        # Extract F1 scores
        result = {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
        return result
    except Exception as e:
        logging.getLogger().error(f"Error calculating ROUGE: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def calculate_meteor(reference, hypothesis):
    """
    Calculate METEOR score.
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        
    Returns:
        METEOR score
    """
    if not hypothesis:
        return 0.0
    
    try:
        # Tokenize
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Calculate METEOR
        score = meteor_score([ref_tokens], hyp_tokens)
        return score
    except Exception as e:
        logging.getLogger().error(f"Error calculating METEOR: {e}")
        return 0.0

def calculate_exact_match(reference, hypothesis):
    """
    Calculate exact match score.
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not hypothesis:
        return 0.0
    
    # Normalize
    ref_norm = reference.lower().strip()
    hyp_norm = hypothesis.lower().strip()
    
    return 1.0 if ref_norm == hyp_norm else 0.0

def calculate_f1_score(reference, hypothesis):
    """
    Calculate word-level F1 score.
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        
    Returns:
        F1 score
    """
    if not hypothesis:
        return 0.0
    
    # Tokenize and create sets
    ref_tokens = set(nltk.word_tokenize(reference.lower()))
    hyp_tokens = set(nltk.word_tokenize(hypothesis.lower()))
    
    # Calculate precision, recall, F1
    if len(hyp_tokens) == 0:
        return 0.0
    
    common = ref_tokens.intersection(hyp_tokens)
    precision = len(common) / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def calculate_bertscore(references, hypotheses, model_type='bert-base-uncased'):
    """
    Calculate BERTScore.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated texts
        model_type: Model type for BERTScore
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(hypotheses, references, lang='en', model_type=model_type)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    except ImportError:
        logging.getLogger().warning("BERTScore not available. Install with: pip install bert-score")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    except Exception as e:
        logging.getLogger().error(f"Error calculating BERTScore: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

def calculate_cross_modal_alignment(visual_emb, text_emb):
    """
    Calculate cross-modal alignment score.
    
    Args:
        visual_emb: Visual embeddings tensor
        text_emb: Text embeddings tensor
        
    Returns:
        Alignment score (cosine similarity)
    """
    # Normalize embeddings
    visual_emb_norm = F.normalize(visual_emb, p=2, dim=-1)
    text_emb_norm = F.normalize(text_emb, p=2, dim=-1)
    
    # Calculate cosine similarity
    sim = torch.sum(visual_emb_norm * text_emb_norm, dim=-1)
    return sim.mean().item()

def evaluate_step_identification(predicted_steps, ground_truth_steps):
    """
    Calculate step identification accuracy.
    
    Args:
        predicted_steps: List of predicted step indices
        ground_truth_steps: List of ground truth step indices
        
    Returns:
        Accuracy score
    """
    if not predicted_steps or not ground_truth_steps:
        return 0.0
    
    correct = 0
    for pred, gt in zip(predicted_steps, ground_truth_steps):
        if pred == gt:
            correct += 1
    
    return correct / len(ground_truth_steps)

def evaluate_answer_generation(references, hypotheses, logger=None):
    """
    Calculate comprehensive evaluation metrics for generated answers.
    
    Args:
        references: List of reference answers
        hypotheses: List of generated answers
        logger: Optional logger instance
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    if len(references) != len(hypotheses):
        _logger.error(f"Mismatch in evaluation: {len(references)} references vs {len(hypotheses)} hypotheses")
        return {}
    
    metrics = {
        'bleu': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'meteor': [],
        'exact_match': [],
        'f1': []
    }
    
    for ref, hyp in zip(references, hypotheses):
        # Calculate metrics for each sample
        metrics['bleu'].append(calculate_bleu(ref, hyp))
        
        rouge_scores = calculate_rouge(ref, hyp)
        metrics['rouge1'].append(rouge_scores['rouge1'])
        metrics['rouge2'].append(rouge_scores['rouge2'])
        metrics['rougeL'].append(rouge_scores['rougeL'])
        
        metrics['meteor'].append(calculate_meteor(ref, hyp))
        metrics['exact_match'].append(calculate_exact_match(ref, hyp))
        metrics['f1'].append(calculate_f1_score(ref, hyp))
    
    # Add BERTScore if more than 5 samples (for efficiency)
    if len(references) >= 5:
        bert_scores = calculate_bertscore(references, hypotheses)
        metrics['bertscore_precision'] = bert_scores['precision']
        metrics['bertscore_recall'] = bert_scores['recall']
        metrics['bertscore_f1'] = bert_scores['f1']
    
    # Calculate averages
    results = {}
    for key, values in metrics.items():
        if isinstance(values, list):
            results[key] = sum(values) / len(values)
        else:
            results[key] = values
    
    return results

def log_generated_answers(questions, references, hypotheses, output_file='generated_answers.txt', logger=None):
   """
   Log generated answers with references for evaluation.
   
   Args:
       questions: List of questions
       references: List of reference answers
       hypotheses: List of generated answers
       output_file: Path to output file
       logger: Optional logger instance
   """
   # Use provided logger or get a default one
   _logger = logger or logging.getLogger()
   
   os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
   
   with open(output_file, 'w') as f:
       for i, (q, r, h) in enumerate(zip(questions, references, hypotheses)):
           f.write(f"Example {i+1}:\n")
           f.write(f"Question: {q}\n")
           f.write(f"Reference: {r}\n")
           f.write(f"Generated: {h}\n")
           
           # Calculate individual metrics
           bleu = calculate_bleu(r, h)
           rouge = calculate_rouge(r, h)
           meteor = calculate_meteor(r, h)
           exact_match = calculate_exact_match(r, h)
           f1 = calculate_f1_score(r, h)
           
           f.write(f"BLEU: {bleu:.4f}, ROUGE-L: {rouge['rougeL']:.4f}, METEOR: {meteor:.4f}\n")
           f.write(f"Exact Match: {exact_match:.4f}, F1: {f1:.4f}\n\n")
   
   _logger.info(f"Generated answers logged to {output_file}")

# ---------------------------
# Visualization Tools
# ---------------------------

def visualize_attention(attention_weights, video_frames, output_path=None):
    """
    Visualize attention weights on video frames.
    
    Args:
        attention_weights: Attention weight tensor [num_heads, seq_len, seq_len] or numpy array
        video_frames: List of frame indices or timestamps
        output_path: Optional path to save visualization
        
    Returns:
        Matplotlib figure
    """
    # Average attention across heads if tensor has 3 dimensions
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
        
    if len(attention_weights.shape) == 3:
        attn = attention_weights.mean(axis=0)
    else:
        attn = attention_weights
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set labels
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frame Index')
    ax.set_title('Cross-Modal Attention Visualization')
    
    # Set ticks if frames are provided
    if len(video_frames) > 10:
        # Use subset of frames for readability
        step = len(video_frames) // 10
        tick_idx = range(0, len(video_frames), step)
        ax.set_xticks(tick_idx)
        ax.set_yticks(tick_idx)
        ax.set_xticklabels([f"{video_frames[i]:.1f}" for i in tick_idx], rotation=45)
        ax.set_yticklabels([f"{video_frames[i]:.1f}" for i in tick_idx])
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    return fig

def visualize_attention_timeline(attention_weights, timestamps, step_boundaries=None, output_path=None):
    """
    Visualize attention distribution over time.
    
    Args:
        attention_weights: Attention weight tensor
        timestamps: List of frame timestamps
        step_boundaries: Optional list of step boundary timestamps
        output_path: Optional path to save visualization
        
    Returns:
        Matplotlib figure
    """
    # Process attention weights
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Handle different attention weight shapes more robustly
    if len(attention_weights.shape) == 3:  # [heads, seq, seq]
        try:
            # Use attention from CLS token or average over all tokens
            attn = attention_weights.mean(axis=0)[0]
        except IndexError:
            # If first row doesn't exist, use mean of all rows
            attn = attention_weights.mean(axis=0).mean(axis=0)
    elif len(attention_weights.shape) == 2:  # [seq, seq]
        try:
            attn = attention_weights[0]
        except IndexError:
            # If first row doesn't exist, use mean of all rows
            attn = attention_weights.mean(axis=0)
    else:
        attn = attention_weights
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure dimensions match
    min_len = min(len(attn), len(timestamps))
    if min_len == 0:
        logging.getLogger().warning("Empty attention or timestamps array")
        return fig
    
    # Plot attention weights
    ax.plot(timestamps[:min_len], attn[:min_len], 'b-', linewidth=2, label='Attention Weight')
    
    # Add step boundaries if provided
    if step_boundaries:
        for i, (start, end) in enumerate(step_boundaries):
            # Ensure boundaries are within the range of the timeline
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                # Only draw if at least partially within range
                if start <= max(timestamps) and end >= min(timestamps):
                    start_val = max(start, min(timestamps))
                    end_val = min(end, max(timestamps))
                    ax.axvspan(start_val, end_val, alpha=0.2, color='red', 
                              label='Step Boundary' if i == 0 else "")
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Temporal Attention Distribution')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
    
    return fig

def plot_training_curves(train_losses, val_losses, metrics=None, output_dir='plots', logger=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics: Optional dictionary of additional metrics
        output_dir: Directory to save plots
        logger: Optional logger instance
        
    Returns:
        Dictionary with figure objects
    """
    # Use provided logger or get a default one
    _logger = logger or logging.getLogger()
    
    os.makedirs(output_dir, exist_ok=True)
    figures = {}
    
    # Plot losses
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
    ax.plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    figures['loss'] = fig
    
    # Plot additional metrics if provided
    if metrics:
        for metric_name, values in metrics.items():
            if isinstance(values, list) and len(values) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(values, 'g-', linewidth=2, label=metric_name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} over Training')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name}_curve.png'), dpi=300)
                figures[metric_name] = fig
    
    return figures

def create_attention_visualization_report(video_ids, questions, attention_weights, frame_timestamps, step_boundaries, output_dir='attention_visualizations', logger=None):
   """
   Create a comprehensive attention visualization report.
   
   Args:
       video_ids: List of video IDs
       questions: List of questions
       attention_weights: List of attention weight tensors
       frame_timestamps: List of lists with frame timestamps
       step_boundaries: List of lists with step boundaries
       output_dir: Directory to save visualizations
       logger: Optional logger instance
   """
   # Use provided logger or get a default one
   _logger = logger or logging.getLogger()
   
   os.makedirs(output_dir, exist_ok=True)
   
   # Create report index file
   with open(os.path.join(output_dir, 'report.html'), 'w') as f:
       f.write("<html><head><title>Attention Visualization Report</title>")
       f.write("<style>body{font-family:Arial;margin:20px} .example{margin:20px 0;padding:15px;border:1px solid #ddd}</style>")
       f.write("</head><body>")
       f.write("<h1>Attention Visualization Report</h1>")
       
       for i, (vid, q, attn, timestamps, boundaries) in enumerate(zip(video_ids, questions, attention_weights, frame_timestamps, step_boundaries)):
           # Create visualizations
           heatmap_path = os.path.join(output_dir, f"{vid}_heatmap.png")
           timeline_path = os.path.join(output_dir, f"{vid}_timeline.png")
           
           visualize_attention(attn, timestamps, heatmap_path)
           visualize_attention_timeline(attn, timestamps, boundaries, timeline_path)
           
           # Add to report
           f.write(f"<div class='example'>")
           f.write(f"<h2>Example {i+1}: {vid}</h2>")
           f.write(f"<p><strong>Question:</strong> {q}</p>")
           f.write(f"<h3>Attention Heatmap</h3>")
           f.write(f"<img src='{os.path.basename(heatmap_path)}' width='600'/>")
           f.write(f"<h3>Temporal Attention</h3>")
           f.write(f"<img src='{os.path.basename(timeline_path)}' width='600'/>")
           f.write(f"</div>")
       
       f.write("</body></html>")
   
   _logger.info(f"Attention visualization report created at {os.path.join(output_dir, 'report.html')}")

# ---------------------------
# Utility Functions
# ---------------------------

def set_device(device_id=None, logger=None):
   """
   Set device for training.
   
   Args:
       device_id: Optional specific GPU ID
       logger: Optional logger instance
       
   Returns:
       torch.device
   """
   # Use provided logger or get a default one
   _logger = logger or logging.getLogger()
   
   if device_id is not None and torch.cuda.is_available():
       device = torch.device(f"cuda:{device_id}")
       torch.cuda.set_device(device_id)
   elif torch.cuda.is_available():
       device = torch.device("cuda")
   else:
       device = torch.device("cpu")
   
   _logger.info(f"Using device: {device}")
   return device

def count_parameters(model):
   """
   Count number of trainable parameters in a model.
   
   Args:
       model: PyTorch model
       
   Returns:
       Number of trainable parameters
   """
   return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_experiment_dir(base_dir, experiment_name=None, logger=None):
   """
   Create a directory for experiment outputs.
   
   Args:
       base_dir: Base directory
       experiment_name: Optional experiment name
       logger: Optional logger instance
       
   Returns:
       Path to experiment directory
   """
   # Use provided logger or get a default one
   _logger = logger or logging.getLogger()
   
   if experiment_name is None:
       experiment_name = time.strftime("%Y%m%d-%H%M%S")
   
   exp_dir = os.path.join(base_dir, experiment_name)
   os.makedirs(exp_dir, exist_ok=True)
   
   # Create subdirectories
   os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
   os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
   os.makedirs(os.path.join(exp_dir, 'visualizations'), exist_ok=True)
   os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
   
   _logger.info(f"Created experiment directory: {exp_dir}")
   return exp_dir

def save_config(config, path, logger=None):
   """
   Save experiment configuration.
   
   Args:
       config: Configuration dictionary
       path: Path to save config
       logger: Optional logger instance
   """
   # Use provided logger or get a default one
   _logger = logger or logging.getLogger()
   
   with open(path, 'w') as f:
       json.dump(config, f, indent=2)
   
   _logger.info(f"Configuration saved to {path}")

def load_config(path, logger=None):
   """
   Load experiment configuration.
   
   Args:
       path: Path to config file
       logger: Optional logger instance
       
   Returns:
       Configuration dictionary
   """
   # Use provided logger or get a default one
   _logger = logger or logging.getLogger()
   
   with open(path, 'r') as f:
       config = json.load(f)
   
   _logger.info(f"Configuration loaded from {path}")
   return config

class AdaptiveEarlyStopping:
    """
    Adaptive early stopping mechanism that increases patience 
    when learning rate is reduced.
    """
    def __init__(self, initial_patience=8, max_patience=20, patience_increase=2):
        """
        Initialize adaptive early stopping.
        
        Args:
            initial_patience: Initial number of epochs to wait for improvement
            max_patience: Maximum number of epochs to wait
            patience_increase: Number of epochs to add when learning rate reduces
        """
        self.initial_patience = initial_patience
        self.max_patience = max_patience
        self.patience_increase = patience_increase
        
        self.patience = initial_patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, current_loss, epoch):
        """
        Check if training should stop.
        
        Args:
            current_loss: Current validation loss
            epoch: Current epoch number
        
        Returns:
            bool: Whether to stop training
        """
        if current_loss < self.best_loss:
            # Reset counter if loss improves
            self.best_loss = current_loss
            self.counter = 0
            self.best_epoch = epoch
            return False
        
        self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def increase_patience(self):
        """
        Increase patience when learning rate is reduced.
        """
        # Increase patience, but not beyond max_patience
        self.patience = min(
            self.patience + self.patience_increase, 
            self.max_patience
        )
    
    def state_dict(self):
        """
        Get the state of the early stopping.
        
        Returns:
            dict: State dictionary
        """
        return {
            'initial_patience': self.initial_patience,
            'max_patience': self.max_patience,
            'patience_increase': self.patience_increase,
            'patience': self.patience,
            'best_loss': self.best_loss,
            'counter': self.counter,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch
        }
    
    def load_state_dict(self, state):
        """
        Load the state of early stopping.
        
        Args:
            state: State dictionary to load
        """
        self.initial_patience = state['initial_patience']
        self.max_patience = state['max_patience']
        self.patience_increase = state['patience_increase']
        self.patience = state['patience']
        self.best_loss = state['best_loss']
        self.counter = state['counter']
        self.early_stop = state['early_stop']
        if 'best_epoch' in state:
            self.best_epoch = state['best_epoch']

# ---------------------------
# Main execution check
# ---------------------------

if __name__ == "__main__":
   test_logger = setup_logging(log_file='utils_test.log', logger_name='utils_test')
   test_logger.info("Testing utils.py functionality")
   
   # Set random seed
   set_seed(42)
   
   # Test device setup
   device = set_device(logger=test_logger)
   
   # Test experiment directory creation
   exp_dir = create_experiment_dir("experiments", "test_run", logger=test_logger)
   
   # Test config saving/loading
   test_config = {
       "model": "t5-base",
       "batch_size": 16,
       "learning_rate": 1e-4,
       "max_epochs": 100
   }
   save_config(test_config, os.path.join(exp_dir, "config.json"), logger=test_logger)
   loaded_config = load_config(os.path.join(exp_dir, "config.json"), logger=test_logger)
   
   test_logger.info("All utility functions tested successfully")