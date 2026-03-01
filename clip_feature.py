import os
os.environ["OPENCV_FFMPEG_QUIET"] = "1" 
import json
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import time
import shutil
from pathlib import Path
import logging
import sys
import contextlib

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # Only show errors, suppress warnings

# Create a context manager to temporarily redirect stderr
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, 
                 split_info_path, 
                 output_base_dir,
                 clip_model_name="ViT-B/32",
                 high_sample_rate=5,    # frames per second during key moments
                 medium_sample_rate=2,   # frames per second during normal segments
                 low_sample_rate=0.5,    # frames per second during less important segments
                 transition_window=2.0,  # seconds around start/end of steps
                 min_frames=10,          # minimum frames to sample for very short videos
                 max_frames=500,         # maximum frames to cap for very long videos
                 visualize_subset=True,  # whether to create visualizations for a subset of videos
                 visualization_samples=2 # number of videos to visualize per split
                ):
        """
        Initialize the feature extractor with the given parameters.
        
        Args:
            split_info_path: Path to the split_info.json file
            output_base_dir: Base directory to save extracted features
            clip_model_name: Name of the CLIP model to use
            high_sample_rate: Frames per second for key moments
            medium_sample_rate: Frames per second for normal segments
            low_sample_rate: Frames per second for less important segments
            transition_window: Seconds around start/end of steps to sample at high rate
            min_frames: Minimum frames to extract for very short videos
            max_frames: Maximum frames to cap for very long videos
            visualize_subset: Whether to create visualizations for some videos
            visualization_samples: Number of videos to visualize per split
        """
        self.split_info_path = split_info_path
        self.output_base_dir = output_base_dir
        self.clip_model_name = clip_model_name
        self.high_sample_rate = high_sample_rate
        self.medium_sample_rate = medium_sample_rate
        self.low_sample_rate = low_sample_rate
        self.transition_window = transition_window
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.visualize_subset = visualize_subset
        self.visualization_samples = visualization_samples
        
        # Require GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for feature extraction. Please run on a GPU-enabled machine.")
        self.device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Load CLIP model
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        logger.info("CLIP model loaded successfully")
        
        # Load split information
        with open(split_info_path, 'r') as f:
            self.split_info = json.load(f)
        
        # Create visualization directory
        self.viz_dir = os.path.join(output_base_dir, "visualizations")
        if self.visualize_subset:
            os.makedirs(self.viz_dir, exist_ok=True)
    
    def load_split_data(self, split_name, split_json_path):
        """
        Load the data for a specific split.
        
        Args:
            split_name: Name of the split (train, validation, test)
            split_json_path: Path to the split's JSON file
        
        Returns:
            The loaded split data
        """
        logger.info(f"Loading {split_name} data from {split_json_path}")
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        return split_data
    
    def get_adaptive_sampling_plan(self, video_path, annotations, fps):
        """
        Create an adaptive sampling plan for a video based on its annotations.
        
        Args:
            video_path: Path to the video file
            annotations: List of step annotations with timestamps
            fps: Frames per second of the video
            
        Returns:
            A list of frame indices to sample
        """
        with suppress_stdout_stderr():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            cap.release()
        
        # Handle very short videos
        if duration < 5:  # Less than 5 seconds
            # Sample uniformly to get at least min_frames
            frame_indices = np.linspace(0, total_frames-1, self.min_frames, dtype=int)
            timestamps = [i / fps for i in frame_indices]
            return frame_indices, timestamps
        
        # Create a sampling density map (higher value = more dense sampling)
        density_map = np.ones(total_frames) * self.low_sample_rate
        
        # If no annotations, use uniform medium sampling
        if not annotations or len(annotations) == 0:
            density_map[:] = self.medium_sample_rate
        else:
            # Add high density at the start and end of each step
            for step in annotations:
                if 'segment' in step and len(step['segment']) >= 2:
                    start_time, end_time = step['segment']
                    
                    # Convert times to frame indices
                    start_frame = max(0, int(start_time * fps))
                    end_frame = min(total_frames-1, int(end_time * fps))
                    
                    # Apply high sample rate at beginning and end of step
                    start_window = min(int(self.transition_window * fps), (end_frame - start_frame) // 4)
                    end_window = min(int(self.transition_window * fps), (end_frame - start_frame) // 4)
                    
                    # Beginning of step
                    start_window_end = min(start_frame + start_window, total_frames-1)
                    density_map[start_frame:start_window_end] = self.high_sample_rate
                    
                    # End of step
                    end_window_start = max(0, end_frame - end_window)
                    density_map[end_window_start:end_frame] = self.high_sample_rate
                    
                    # Middle of step - medium sample rate
                    density_map[start_window_end:end_window_start] = self.medium_sample_rate
        
        # Compute the number of frames to sample based on density map
        expected_frames = np.sum(density_map) / fps
        
        # Cap the number of frames if needed
        if expected_frames > self.max_frames:
            # Scale down the density map
            scale_factor = self.max_frames / expected_frames
            density_map *= scale_factor
            expected_frames = self.max_frames
        
        # Ensure minimum number of frames
        if expected_frames < self.min_frames:
            # Scale up the density map
            scale_factor = self.min_frames / expected_frames
            density_map *= scale_factor
        
        # Convert density map to actual sampling plan
        frame_indices = []
        timestamps = []
        
        for frame_idx in range(total_frames):
            # Probability of sampling this frame
            sample_prob = density_map[frame_idx] / fps
            if random.random() < sample_prob:
                frame_indices.append(frame_idx)
                timestamps.append(frame_idx / fps)
        
        # If no frames were selected (rare but possible), fall back to uniform sampling
        if len(frame_indices) == 0:
            frame_indices = np.linspace(0, total_frames-1, self.min_frames, dtype=int)
            timestamps = [i / fps for i in frame_indices]
        
        return frame_indices, timestamps
    
    def extract_frames_and_features(self, video_path, annotations):
        """
        Extract frames according to adaptive sampling plan and compute CLIP features.
        
        Args:
            video_path: Path to the video file
            annotations: List of step annotations with timestamps
            
        Returns:
            A tuple of (features, timestamps, sampled_frames)
        """
        with suppress_stdout_stderr():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get adaptive sampling plan
        frame_indices, timestamps = self.get_adaptive_sampling_plan(video_path, annotations, fps)
        
        # Extract frames and compute features
        features = []
        sampled_frames = []
        
        for frame_idx in tqdm(frame_indices, desc="Extracting features", leave=False):
            # Set the frame position
            with suppress_stdout_stderr():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                continue
            
            # Convert BGR to RGB and to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            sampled_frames.append(pil_image)
            
            # Preprocess for CLIP
            preprocessed = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Extract CLIP features
            with torch.no_grad():
                feature = self.model.encode_image(preprocessed)
                features.append(feature.cpu())
        
        with suppress_stdout_stderr():
            cap.release()
        
        # Stack features into a tensor
        if len(features) > 0:
            features_tensor = torch.cat(features)
        else:
            raise ValueError(f"No features extracted from {video_path}")
        
        return features_tensor, timestamps, sampled_frames
    
    def visualize_sampled_frames(self, video_id, sampled_frames, timestamps, output_dir):
        """
        Create a visualization of sampled frames from a video.
        
        Args:
            video_id: ID of the video
            sampled_frames: List of sampled frames (PIL Images)
            timestamps: List of timestamps for each frame
            output_dir: Directory to save the visualization
        """
        # Create a grid of sampled frames
        n_frames = len(sampled_frames)
        n_cols = min(5, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for i, (frame, timestamp) in enumerate(zip(sampled_frames, timestamps)):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(frame)
            plt.title(f"t={timestamp:.1f}s")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_sampled_frames.png"))
        plt.close()
    
    def visualize_feature_tsne(self, video_id, features, timestamps, output_dir):
        """
        Create a t-SNE visualization of the feature space.
        
        Args:
            video_id: ID of the video
            features: Tensor of features
            timestamps: List of timestamps for each feature
            output_dir: Directory to save the visualization
        """
        # Only perform t-SNE if we have enough frames
        if len(timestamps) < 3:
            logger.warning(f"Not enough frames for t-SNE visualization for video {video_id}")
            return
            
        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features.numpy())
        
        # Plot with timestamps as color gradient
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=timestamps, cmap='viridis', 
                            alpha=0.8)
        plt.colorbar(scatter, label='Time (seconds)')
        plt.title(f't-SNE Visualization of Features for Video {video_id}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_tsne.png"))
        plt.close()
    
    def create_feature_histograms(self, split_name, features_list, output_dir):
        """
        Create histograms of feature statistics for a split.
        
        Args:
            split_name: Name of the split
            features_list: List of feature tensors
            output_dir: Directory to save the visualization
        """
        # Get number of frames for each video
        n_frames = [features.shape[0] for features in features_list]
        
        plt.figure(figsize=(10, 6))
        plt.hist(n_frames, bins=30, alpha=0.7, color='blue')
        plt.title(f'Distribution of Sampled Frames per Video ({split_name})')
        plt.xlabel('Number of Frames')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{split_name}_frames_histogram.png"))
        plt.close()
    
    def process_split(self, split_name, split_json_path):
        """
        Process all videos in a split.
        
        Args:
            split_name: Name of the split (train, validation, test)
            split_json_path: Path to the split's JSON file
        """
        logger.info(f"Processing {split_name} split")
        
        # Load split data
        split_data = self.load_split_data(split_name, split_json_path)
        
        # Get videos for this split
        videos = self.split_info[split_name]["videos"]
        
        # Create output directory for this split
        split_output_dir = os.path.join(self.output_base_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Process each video
        features_list = []
        processed_count = 0
        failures = []
        
        # Select videos to visualize if enabled
        visualization_videos = []
        if self.visualize_subset:
            visualization_videos = random.sample(videos, min(self.visualization_samples, len(videos)))
        
        # Process videos with progress bar
        for video in tqdm(videos, desc=f"Processing {split_name} videos"):
            video_id = video["id"]
            video_path = video["path"]
            
            # Get subfolder from path
            subfolder = os.path.basename(os.path.dirname(video_path))
            
            # Create output directory for this subfolder
            subfolder_output_dir = os.path.join(split_output_dir, subfolder)
            os.makedirs(subfolder_output_dir, exist_ok=True)
            
            # Output path for features
            output_path = os.path.join(subfolder_output_dir, f"{video_id}.pt")
            
            # Skip if already processed
            if os.path.exists(output_path):
                logger.info(f"Skipping already processed video: {video_id}")
                continue
            
            try:
                # Get annotations for this video
                if video_id in split_data["database"]:
                    annotations = split_data["database"][video_id].get("annotation", [])
                else:
                    logger.warning(f"Video {video_id} not found in split data, using empty annotations")
                    annotations = []
                
                # Extract frames and compute features
                features, timestamps, sampled_frames = self.extract_frames_and_features(video_path, annotations)
                
                # Save features with metadata
                metadata = {
                    "video_id": video_id,
                    "timestamps": timestamps,
                    "num_frames": len(timestamps)
                }
                
                torch.save({
                    "features": features,
                    "metadata": metadata
                }, output_path)
                
                features_list.append(features)
                processed_count += 1
                
                # Create visualizations for selected videos
                should_visualize = any(v["id"] == video_id for v in visualization_videos)
                if should_visualize:
                    # Create visualization directory for this video
                    video_viz_dir = os.path.join(self.viz_dir, split_name, video_id)
                    os.makedirs(video_viz_dir, exist_ok=True)
                    
                    # Visualize sampled frames
                    self.visualize_sampled_frames(video_id, sampled_frames, timestamps, video_viz_dir)
                    
                    # Visualize feature space
                    self.visualize_feature_tsne(video_id, features, timestamps, video_viz_dir)
                
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {str(e)}")
                failures.append((video_id, str(e)))
        
        # Create feature histograms for this split
        if self.visualize_subset and len(features_list) > 0:
            self.create_feature_histograms(split_name, features_list, self.viz_dir)
        
        # Print summary for this split
        logger.info(f"Processed {processed_count} videos in {split_name} split")
        if failures:
            logger.warning(f"Failed to process {len(failures)} videos")
            for video_id, error in failures:
                logger.warning(f"  - {video_id}: {error}")
        
        return processed_count, len(failures)
    
    def run(self, train_json_path, val_json_path, test_json_path):
        """
        Run the feature extraction process for all splits.
        
        Args:
            train_json_path: Path to the train.json file
            val_json_path: Path to the validation.json file
            test_json_path: Path to the test.json file
        """
        start_time = time.time()
        logger.info("Starting feature extraction process")
        
        # Process each split
        train_processed, train_failures = self.process_split("train", train_json_path)
        val_processed, val_failures = self.process_split("validation", val_json_path)
        test_processed, test_failures = self.process_split("test", test_json_path)
        
        # Print overall summary
        total_processed = train_processed + val_processed + test_processed
        total_failures = train_failures + val_failures + test_failures
        
        logger.info("=" * 50)
        logger.info("Feature extraction complete")
        logger.info(f"Total videos processed: {total_processed}")
        logger.info(f"Total failures: {total_failures}")
        logger.info(f"Processing time: {(time.time() - start_time) / 60:.2f} minutes")
        logger.info("=" * 50)

def main():
    # Paths
    split_info_path = "Raw_Dataset/splits/split_info.json"
    train_json_path = "Raw_Dataset/splits/train.json"
    val_json_path = "Raw_Dataset/splits/validation.json"
    test_json_path = "Raw_Dataset/splits/test.json"
    output_base_dir = "Raw_Dataset/features"
    
    # Create feature extractor
    extractor = FeatureExtractor(
        split_info_path=split_info_path,
        output_base_dir=output_base_dir,
        visualize_subset=True,
        visualization_samples=2  # Visualize 2 videos per split
    )
    
    # Run feature extraction
    extractor.run(train_json_path, val_json_path, test_json_path)

if __name__ == "__main__":
    main()