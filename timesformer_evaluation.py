import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("timesformer_evaluation.log"),
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

def load_video_metadata(split_info_path):
    """
    Load video metadata including classes and step annotations from JSON files.
    
    Args:
        split_info_path: Path to the split information JSON file
        
    Returns:
        video_classes: Dictionary mapping video IDs to class names
        video_steps: Dictionary mapping video IDs to step annotations
    """
    # Create path to directory containing split files
    split_dir = os.path.dirname(split_info_path)
    
    # Initialize dictionaries
    video_classes = {}
    video_steps = {}
    
    # Load data for each split
    for split_name in ['train', 'validation', 'test']:
        split_path = os.path.join(split_dir, f"{split_name}.json")
        
        try:
            with open(split_path, 'r') as f:
                split_data = json.load(f)
            
            # Extract class and step information
            for video_id, data in split_data['database'].items():
                # Get class information
                if 'class' in data:
                    video_classes[video_id] = data['class']
                
                # Get step annotations
                if 'annotation' in data and len(data['annotation']) > 0:
                    steps = []
                    for step in data['annotation']:
                        if 'segment' in step and len(step['segment']) >= 2:
                            steps.append({
                                'start': step['segment'][0],
                                'end': step['segment'][1],
                                'label': step.get('label', '')
                            })
                    
                    if steps:
                        video_steps[video_id] = steps
        except Exception as e:
            logger.error(f"Error loading {split_path}: {e}")
    
    logger.info(f"Loaded class information for {len(video_classes)} videos")
    logger.info(f"Loaded step annotations for {len(video_steps)} videos")
    
    return video_classes, video_steps

def evaluate_temporal_consistency(temporal_dir, video_classes, output_dir, max_videos_per_class=10):
    """
    Evaluate temporal consistency by comparing embeddings of videos with similar procedures.
    
    Args:
        temporal_dir: Directory containing temporal feature files
        video_classes: Dictionary mapping video IDs to class names
        output_dir: Directory to save output files
        max_videos_per_class: Maximum number of videos to include per class
        
    Returns:
        intra_mean: Mean intra-class similarity
        inter_mean: Mean inter-class similarity
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect embeddings by class
    class_embeddings = {}
    found_videos = set()
    
    # Process test split for evaluation
    test_dir = os.path.join(temporal_dir, 'test')
    if not os.path.exists(test_dir):
        logger.error(f"Test directory {test_dir} not found")
        return 0, 0
    
    # Collect all feature files
    feature_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.pt'):
                feature_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(feature_files)} feature files in test set")
    
    # Process features with progress bar
    for file_path in tqdm(feature_files, desc="Loading embeddings"):
        try:
            # Load temporal features
            data = torch.load(file_path)
            video_id = data['metadata']['video_id']
            embedding = data['features'].cpu().numpy()
            
            found_videos.add(video_id)
            
            # Add to class embeddings if class is known
            if video_id in video_classes:
                class_name = video_classes[video_id]
                if class_name not in class_embeddings:
                    class_embeddings[class_name] = []
                
                class_embeddings[class_name].append({
                    'video_id': video_id,
                    'embedding': embedding
                })
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Found temporal embeddings for {len(found_videos)} videos")
    logger.info(f"Found {len(class_embeddings)} classes with embeddings")
    
    # Calculate intra-class and inter-class similarities
    intra_class_sims = []
    inter_class_sims = []
    
    # Classes with enough samples for analysis
    valid_classes = [c for c, e in class_embeddings.items() if len(e) >= 2]
    
    if len(valid_classes) < 2:
        logger.warning("Not enough classes with multiple videos for analysis")
        return 0, 0
    
    # Limit videos per class to prevent bias from large classes
    for class_name in valid_classes:
        if len(class_embeddings[class_name]) > max_videos_per_class:
            class_embeddings[class_name] = random.sample(class_embeddings[class_name], max_videos_per_class)
    
    # Calculate intra-class similarities
    for class_name in tqdm(valid_classes, desc="Calculating intra-class similarities"):
        embeddings = [e['embedding'] for e in class_embeddings[class_name]]
        video_ids = [e['video_id'] for e in class_embeddings[class_name]]
        
        # Calculate pairwise similarities within class
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity(embeddings[i].reshape(1, -1), 
                                       embeddings[j].reshape(1, -1))[0][0]
                intra_class_sims.append(sim)
    
    # Calculate inter-class similarities
    for i, class1 in enumerate(tqdm(valid_classes, desc="Calculating inter-class similarities")):
        for j in range(i+1, len(valid_classes)):
            class2 = valid_classes[j]
            
            # Sample pairs across classes
            for vid1 in class_embeddings[class1]:
                for vid2 in class_embeddings[class2]:
                    sim = cosine_similarity(vid1['embedding'].reshape(1, -1),
                                           vid2['embedding'].reshape(1, -1))[0][0]
                    inter_class_sims.append(sim)
    
    # Skip if no similarities were calculated
    if not intra_class_sims or not inter_class_sims:
        logger.error("No similarity values calculated")
        return 0, 0
    
    # Plot similarity distributions
    plt.figure(figsize=(12, 7))
    
    # Create a nice histogram with KDE
    bins = np.linspace(min(min(intra_class_sims), min(inter_class_sims)),
                      max(max(intra_class_sims), max(inter_class_sims)), 
                      30)
    
    plt.hist(intra_class_sims, alpha=0.6, bins=bins, label=f'Intra-class (n={len(intra_class_sims)})',
            density=True, color='blue', edgecolor='black')
    plt.hist(inter_class_sims, alpha=0.6, bins=bins, label=f'Inter-class (n={len(inter_class_sims)})',
            density=True, color='red', edgecolor='black')
    
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.title('Temporal Embedding Similarity Distribution', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=300)
    plt.close()
    
    # Calculate statistics
    intra_mean = np.mean(intra_class_sims)
    inter_mean = np.mean(inter_class_sims)
    similarity_gap = intra_mean - inter_mean
    
    # Compute additional statistics
    intra_std = np.std(intra_class_sims)
    inter_std = np.std(inter_class_sims)
    
    # Check for statistical significance with simple t-test-like measure
    pooled_std = np.sqrt((intra_std**2 + inter_std**2) / 2)
    effect_size = similarity_gap / pooled_std if pooled_std > 0 else 0
    
    logger.info(f"Intra-class similarity: {intra_mean:.4f} ± {intra_std:.4f}")
    logger.info(f"Inter-class similarity: {inter_mean:.4f} ± {inter_std:.4f}")
    logger.info(f"Similarity gap: {similarity_gap:.4f}")
    logger.info(f"Effect size: {effect_size:.4f}")
    
    # Save detailed statistics
    with open(os.path.join(output_dir, 'similarity_stats.json'), 'w') as f:
        json.dump({
            'intra_class_mean': float(intra_mean),
            'intra_class_std': float(intra_std),
            'inter_class_mean': float(inter_mean),
            'inter_class_std': float(inter_std),
            'similarity_gap': float(similarity_gap),
            'effect_size': float(effect_size),
            'intra_class_count': len(intra_class_sims),
            'inter_class_count': len(inter_class_sims),
            'num_classes': len(valid_classes)
        }, f, indent=2)
    
    return intra_mean, inter_mean

def evaluate_step_alignment(temporal_dir, features_dir, video_steps, output_dir, max_videos=20):
    """
    Evaluate how well temporal embeddings align with step boundaries.
    
    Args:
        temporal_dir: Directory containing temporal feature files
        features_dir: Directory containing CLIP feature files
        video_steps: Dictionary mapping video IDs to step annotations
        output_dir: Directory to save output files
        max_videos: Maximum number of videos to analyze
        
    Returns:
        step_ratio: Average ratio of boundary to non-boundary feature changes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all test temporal feature files
    test_dir = os.path.join(temporal_dir, 'test')
    if not os.path.exists(test_dir):
        logger.error(f"Test directory {test_dir} not found")
        return 0
    
    # Find all feature files in test set
    feature_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.pt'):
                feature_files.append(os.path.join(root, file))
    
    # Keep only videos with step annotations
    step_video_files = []
    for file_path in feature_files:
        # Extract video ID from filename
        file_name = os.path.basename(file_path)
        video_id = os.path.splitext(file_name)[0]
        
        if video_id in video_steps:
            step_video_files.append((video_id, file_path))
    
    logger.info(f"Found {len(step_video_files)} videos with step annotations")
    
    # Limit the number of videos to analyze
    if len(step_video_files) > max_videos:
        step_video_files = random.sample(step_video_files, max_videos)
    
    # Analyze step alignment
    step_changes = []
    analyzed_videos = 0
    
    # Process videos with progress bar
    for video_id, temporal_file in tqdm(step_video_files, desc="Analyzing step alignment"):
        try:
            # Load temporal features
            temporal_data = torch.load(temporal_file)
            
            # Skip if no timestamps in metadata
            if 'metadata' not in temporal_data or 'timestamps' not in temporal_data['metadata']:
                logger.warning(f"No timestamps found in {temporal_file}")
                continue
            
            # Get timestamps
            timestamps = temporal_data['metadata']['timestamps']
            
            # Determine corresponding CLIP features path
            relative_path = os.path.relpath(temporal_file, temporal_dir)
            clip_file = os.path.join(features_dir, relative_path)
            
            # Check if CLIP features exist
            if not os.path.exists(clip_file):
                # Try with just the filename
                dir_path = os.path.dirname(clip_file)
                base_name = os.path.basename(clip_file)
                
                # Try to find the file recursively
                found = False
                for root, _, files in os.walk(features_dir):
                    if base_name in files:
                        clip_file = os.path.join(root, base_name)
                        found = True
                        break
                
                if not found:
                    logger.warning(f"CLIP features not found for {video_id}")
                    continue
            
            # Load CLIP features
            clip_data = torch.load(clip_file)
            clip_features = clip_data['features']
            
            # Calculate frame-to-frame differences
            if len(clip_features) <= 1:
                logger.warning(f"Not enough frames in {video_id}")
                continue
            
            # Calculate L2 differences between consecutive frames
            feature_diffs = []
            for i in range(1, len(clip_features)):
                diff = torch.norm(clip_features[i] - clip_features[i-1]).item()
                feature_diffs.append(diff)
            
            # Normalize differences to [0, 1] range
            if feature_diffs:
                min_diff = min(feature_diffs)
                max_diff = max(feature_diffs)
                if max_diff > min_diff:
                    feature_diffs = [(d - min_diff) / (max_diff - min_diff) for d in feature_diffs]
            
            # Get step boundary indices
            step_boundaries = []
            for step in video_steps[video_id]:
                # Find indices closest to step start/end times
                start_idx = min(range(len(timestamps)), 
                              key=lambda i: abs(timestamps[i] - step['start']))
                end_idx = min(range(len(timestamps)), 
                            key=lambda i: abs(timestamps[i] - step['end']))
                
                # Adjust to match difference indices (which are one fewer)
                if start_idx > 0:
                    step_boundaries.append(start_idx - 1)
                if end_idx > 0 and end_idx < len(feature_diffs):
                    step_boundaries.append(end_idx - 1)
            
            # Skip if no valid boundaries
            if not step_boundaries:
                logger.warning(f"No valid step boundaries for {video_id}")
                continue
            
            # Create a more visually appealing plot
            plt.figure(figsize=(14, 7))
            
            # Plot frame differences
            plt.plot(range(len(feature_diffs)), feature_diffs, '-', 
                    linewidth=2, color='blue', label='Frame Difference')
            
            # Add step boundary markers
            boundary_y = []
            for b in step_boundaries:
                if 0 <= b < len(feature_diffs):
                    plt.axvline(x=b, color='red', linestyle='--', alpha=0.7, 
                               label='Step Boundary' if 'Step Boundary' not in plt.gca().get_legend_handles_labels()[1] else "")
                    boundary_y.append(feature_diffs[b])
            
            # Aesthetic improvements
            plt.title(f'Feature Changes with Step Boundaries - {video_id}', fontsize=14)
            plt.xlabel('Frame Index', fontsize=12)
            plt.ylabel('Normalized Feature Difference', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend(fontsize=12)
            
            # Adjust y-axis for better visualization
            if boundary_y:
                max_boundary = max(boundary_y)
                plt.ylim([0, max(1.0, max_boundary * 1.2)])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{video_id}_step_alignment.png'), dpi=300)
            plt.close()
            
            # Calculate metrics at boundaries vs. non-boundaries
            boundary_diffs = [feature_diffs[i] for i in step_boundaries if 0 <= i < len(feature_diffs)]
            non_boundary_indices = [i for i in range(len(feature_diffs)) if i not in step_boundaries]
            non_boundary_diffs = [feature_diffs[i] for i in non_boundary_indices]
            
            # Skip if not enough data
            if not boundary_diffs or not non_boundary_diffs:
                logger.warning(f"Not enough boundary or non-boundary points for {video_id}")
                continue
            
            # Calculate statistics
            boundary_mean = np.mean(boundary_diffs)
            non_boundary_mean = np.mean(non_boundary_diffs)
            ratio = boundary_mean / non_boundary_mean if non_boundary_mean > 0 else 0
            
            # Store results
            step_changes.append({
                'video_id': video_id,
                'boundary_mean': boundary_mean,
                'non_boundary_mean': non_boundary_mean,
                'ratio': ratio,
                'num_boundaries': len(boundary_diffs),
                'num_non_boundaries': len(non_boundary_diffs)
            })
            
            analyzed_videos += 1
            
        except Exception as e:
            logger.error(f"Error analyzing {video_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"Successfully analyzed step alignment for {analyzed_videos} videos")
    
    # Calculate overall statistics if we have data
    if step_changes:
        # Compute weighted average based on number of boundary points
        total_boundary_points = sum(s['num_boundaries'] for s in step_changes)
        total_non_boundary_points = sum(s['num_non_boundaries'] for s in step_changes)
        
        # Calculate weighted means
        weighted_boundary_mean = sum(s['boundary_mean'] * s['num_boundaries'] for s in step_changes) / total_boundary_points if total_boundary_points > 0 else 0
        weighted_non_boundary_mean = sum(s['non_boundary_mean'] * s['num_non_boundaries'] for s in step_changes) / total_non_boundary_points if total_non_boundary_points > 0 else 0
        
        # Also calculate simple means
        boundary_means = [s['boundary_mean'] for s in step_changes]
        non_boundary_means = [s['non_boundary_mean'] for s in step_changes]
        ratios = [s['ratio'] for s in step_changes]
        
        simple_boundary_mean = np.mean(boundary_means)
        simple_non_boundary_mean = np.mean(non_boundary_means)
        simple_ratio_mean = np.mean(ratios)
        
        # Calculate weighted ratio
        weighted_ratio = weighted_boundary_mean / weighted_non_boundary_mean if weighted_non_boundary_mean > 0 else 0
        
        logger.info(f"Simple average change at boundaries: {simple_boundary_mean:.4f}")
        logger.info(f"Simple average change at non-boundaries: {simple_non_boundary_mean:.4f}")
        logger.info(f"Simple average ratio: {simple_ratio_mean:.4f}")
        logger.info(f"Weighted average change at boundaries: {weighted_boundary_mean:.4f}")
        logger.info(f"Weighted average change at non-boundaries: {weighted_non_boundary_mean:.4f}")
        logger.info(f"Weighted ratio: {weighted_ratio:.4f}")
        
        # Create additional visualization - box plot of ratios
        plt.figure(figsize=(10, 6))
        plt.boxplot(ratios)
        plt.title('Distribution of Boundary/Non-Boundary Ratios', fontsize=14)
        plt.ylabel('Ratio Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ratio_boxplot.png'), dpi=300)
        plt.close()
        
        # Save detailed statistics
        with open(os.path.join(output_dir, 'step_alignment_stats.json'), 'w') as f:
            json.dump({
                'simple_boundary_mean': float(simple_boundary_mean),
                'simple_non_boundary_mean': float(simple_non_boundary_mean),
                'simple_ratio_mean': float(simple_ratio_mean),
                'weighted_boundary_mean': float(weighted_boundary_mean),
                'weighted_non_boundary_mean': float(weighted_non_boundary_mean),
                'weighted_ratio': float(weighted_ratio),
                'total_boundary_points': total_boundary_points,
                'total_non_boundary_points': total_non_boundary_points,
                'analyzed_videos': analyzed_videos,
                'video_details': step_changes
            }, f, indent=2)
        
        return weighted_ratio
    
    logger.warning("No valid step alignment data collected")
    return 0

def main():
    parser = argparse.ArgumentParser(description="TimeSformer Evaluation")
    parser.add_argument('--temporal_dir', type=str, default='Raw_Dataset/temporal_features',
                       help='Directory containing temporal features')
    parser.add_argument('--features_dir', type=str, default='Raw_Dataset/features',
                       help='Directory containing CLIP features')
    parser.add_argument('--split_info', type=str, default='Raw_Dataset/splits/split_info.json',
                       help='Path to split information JSON file')
    parser.add_argument('--output_dir', type=str, default='Raw_Dataset/evaluations/timesformer',
                       help='Directory to save evaluation results')
    parser.add_argument('--max_videos', type=int, default=20,
                       help='Maximum number of videos to analyze for step alignment')
    parser.add_argument('--max_videos_per_class', type=int, default=10,
                       help='Maximum number of videos per class for temporal consistency')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("TimeSformer Evaluation Configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load video metadata
    logger.info("Loading video metadata...")
    video_classes, video_steps = load_video_metadata(args.split_info)
    
    # Evaluate temporal consistency
    logger.info("Evaluating temporal consistency...")
    intra_sim, inter_sim = evaluate_temporal_consistency(
        args.temporal_dir, video_classes, args.output_dir, args.max_videos_per_class)
    
    # Evaluate step alignment
    logger.info("Evaluating step alignment...")
    step_ratio = evaluate_step_alignment(
        args.temporal_dir, args.features_dir, video_steps, args.output_dir, args.max_videos)
    
    # Print summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Temporal Consistency: {intra_sim - inter_sim:.4f} (intra-inter gap)")
    logger.info(f"Step Alignment Ratio: {step_ratio:.4f} (boundary/non-boundary)")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()