import json
import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
from tqdm import tqdm

def split_dataset():
    # Paths
    json_file_path = "Raw_Dataset/StepsQA.json"
    output_dir = "Raw_Dataset/splits"
    video_base_dir = "Raw_Dataset/videos/videos"  # Base directory for videos
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
    
    # Extract video info for stratification
    video_info = []
    
    # Find all video files and their paths
    video_paths = {}
    print("Scanning video directories...")
    subfolders = [f for f in os.listdir(video_base_dir) if os.path.isdir(os.path.join(video_base_dir, f))]
    
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(video_base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.mp4'):
                    video_id = os.path.splitext(filename)[0]
                    video_paths[video_id] = os.path.join(subfolder_path, filename)
    
    print(f"Found {len(video_paths)} video files in the directories")
    
    # Process video metadata
    print("Processing video metadata...")
    for video_id, data in tqdm(dataset['database'].items(), desc="Extracting video info"):
        # Check if we have the video file
        if video_id not in video_paths:
            print(f"Warning: Video file for {video_id} not found")
            continue
            
        # Get video path
        video_path = video_paths[video_id]
        
        # Get class name
        class_name = data.get('class', 'unknown')
        
        # Get video duration
        duration = data.get('duration', 0)
        
        # Get number of steps (from annotation)
        num_steps = len(data.get('annotation', []))
        
        # Categorize duration
        if duration < 60:  # Less than 1 minute
            duration_category = "short"
        elif duration < 180:  # 1-3 minutes
            duration_category = "medium"
        else:  # 3+ minutes
            duration_category = "long"
        
        # Add to video info list
        video_info.append({
            'video_id': video_id,
            'video_path': video_path,
            'class': class_name,
            'duration': duration,
            'duration_category': duration_category,
            'num_steps': num_steps
        })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(video_info)
    
    # Get unique classes
    classes = df['class'].unique()
    
    # Initialize split dictionaries
    train_ids = []
    val_ids = []
    test_ids = []
    
    # For each class, stratify by duration category and number of steps
    print("Performing stratified split...")
    for class_name in tqdm(classes, desc="Processing classes"):
        class_df = df[df['class'] == class_name]
        
        # Group by duration category
        for duration_cat in class_df['duration_category'].unique():
            duration_df = class_df[class_df['duration_category'] == duration_cat]
            
            # Get video IDs
            class_duration_ids = duration_df['video_id'].tolist()
            n_samples = len(class_duration_ids)
            
            # Handle different cases based on number of samples
            if n_samples < 6:  # Too few for proper 70/15/15 split
                if n_samples <= 2:  # 1-2 samples: all to train
                    train_ids.extend(class_duration_ids)
                elif n_samples <= 4:  # 3-4 samples: train and val only
                    n_train = max(1, int(n_samples * 0.7))
                    train_subset, val_subset = train_test_split(
                        class_duration_ids,
                        train_size=n_train,
                        random_state=42
                    )
                    train_ids.extend(train_subset)
                    val_ids.extend(val_subset)
                else:  # 5 samples: 3 train, 1 val, 1 test
                    np.random.seed(42)
                    np.random.shuffle(class_duration_ids)
                    train_subset = class_duration_ids[:3]
                    val_subset = [class_duration_ids[3]]
                    test_subset = [class_duration_ids[4]]
                    train_ids.extend(train_subset)
                    val_ids.extend(val_subset)
                    test_ids.extend(test_subset)
            else:  # Enough samples for proper split
                # Split into train and temp (val+test)
                train_class_ids, temp_ids = train_test_split(
                    class_duration_ids, 
                    train_size=0.7,
                    random_state=42
                )
                
                # Split temp into val and test (50/50 split of the remaining 30%)
                if len(temp_ids) >= 2:  # At least 2 samples needed for splitting
                    val_class_ids, test_class_ids = train_test_split(
                        temp_ids,
                        train_size=0.5,  # 50% of temp is 15% of total
                        random_state=42
                    )
                    
                    # Add to our overall splits
                    train_ids.extend(train_class_ids)
                    val_ids.extend(val_class_ids)
                    test_ids.extend(test_class_ids)
                else:  # Only 1 sample after first split - add to validation
                    train_ids.extend(train_class_ids)
                    val_ids.extend(temp_ids)
    
    # Create split information with video paths
    split_info = {
        "train": {
            "videos": [],
            "statistics": {
                "count": len(train_ids)
            }
        },
        "validation": {
            "videos": [],
            "statistics": {
                "count": len(val_ids)
            }
        },
        "test": {
            "videos": [],
            "statistics": {
                "count": len(test_ids)
            }
        }
    }
    
    # Add video paths to split info
    print("Building split structure...")
    for video_item in tqdm(video_info, desc="Assigning videos to splits"):
        video_id = video_item['video_id']
        if video_id in train_ids:
            split_info["train"]["videos"].append({
                "id": video_id,
                "path": video_item['video_path']
            })
        elif video_id in val_ids:
            split_info["validation"]["videos"].append({
                "id": video_id,
                "path": video_item['video_path']
            })
        elif video_id in test_ids:
            split_info["test"]["videos"].append({
                "id": video_id,
                "path": video_item['video_path']
            })
    
    # Add more detailed statistics
    print("Calculating statistics...")
    for split_name in ['train', 'validation', 'test']:
        split_ids = [video['id'] for video in split_info[split_name]['videos']]
        split_df = df[df['video_id'].isin(split_ids)]
        
        # Add class distribution
        class_dist = split_df['class'].value_counts().to_dict()
        split_info[split_name]['statistics']['class_distribution'] = class_dist
        
        # Add duration statistics
        split_info[split_name]['statistics']['avg_duration'] = float(split_df['duration'].mean())
        split_info[split_name]['statistics']['duration_categories'] = split_df['duration_category'].value_counts().to_dict()
        
        # Add step count statistics
        split_info[split_name]['statistics']['avg_steps'] = float(split_df['num_steps'].mean())
    
    # Save the split information
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=4)
    
    # Create individual JSON files for each split
    print("Creating split files...")
    for split_name in ['train', 'validation', 'test']:
        split_ids = set(video['id'] for video in split_info[split_name]['videos'])
        split_data = {"database": {}}
        
        for video_id in tqdm(split_ids, desc=f"Processing {split_name} split"):
            if video_id in dataset['database']:
                split_data['database'][video_id] = dataset['database'][video_id]
        
        # Save the split-specific JSON
        with open(os.path.join(output_dir, f'{split_name}.json'), 'w') as f:
            json.dump(split_data, f, indent=4)
    
    # Update counts based on videos with paths
    train_count = len(split_info["train"]["videos"])
    val_count = len(split_info["validation"]["videos"])
    test_count = len(split_info["test"]["videos"])
    total_count = train_count + val_count + test_count
    
    print("\nDataset splitting completed!")
    print(f"Train: {train_count} videos ({train_count/len(dataset['database'])*100:.1f}%)")
    print(f"Validation: {val_count} videos ({val_count/len(dataset['database'])*100:.1f}%)")
    print(f"Test: {test_count} videos ({test_count/len(dataset['database'])*100:.1f}%)")
    print(f"Total: {total_count} videos")
    print(f"Original dataset: {len(dataset['database'])} videos")
    print(f"Split information saved to {os.path.join(output_dir, 'split_info.json')}")
    print(f"Individual split files saved to {output_dir}")

def verify_splits(output_dir, video_base_dir):
    """
    Verify that the splits are correctly created and all video files exist.
    
    Args:
        output_dir: Directory where split files are saved
        video_base_dir: Base directory for video files
    """
    print("\nVerifying splits...")
    
    # Load the split info
    with open(os.path.join(output_dir, 'split_info.json'), 'r') as f:
        split_info = json.load(f)
    
    # Check each split
    all_split_ids = set()
    
    for split_name in ['train', 'validation', 'test']:
        # Load split-specific JSON
        with open(os.path.join(output_dir, f'{split_name}.json'), 'r') as f:
            split_data = json.load(f)
        
        # Get video IDs from split info and JSON file
        info_ids = set(video['id'] for video in split_info[split_name]['videos'])
        json_ids = set(split_data['database'].keys())
        
        # Check counts
        print(f"{split_name.capitalize()} split:")
        print(f"  - Videos in split_info: {len(info_ids)}")
        print(f"  - Videos in JSON file: {len(json_ids)}")
        
        # Check for mismatches
        if info_ids != json_ids:
            print(f"  - WARNING: Mismatch between split_info and JSON file!")
            if len(info_ids - json_ids) > 0:
                print(f"    IDs in split_info but not in JSON: {len(info_ids - json_ids)}")
            if len(json_ids - info_ids) > 0:
                print(f"    IDs in JSON but not in split_info: {len(json_ids - info_ids)}")
        else:
            print(f"  - Video IDs match between split_info and JSON ✓")
        
        # Check if video files exist
        missing_videos = 0
        for video in split_info[split_name]['videos']:
            if not os.path.exists(video['path']):
                missing_videos += 1
        
        if missing_videos > 0:
            print(f"  - WARNING: {missing_videos} video files are missing!")
        else:
            print(f"  - All video files exist ✓")
        
        # Add to all split IDs
        all_split_ids.update(info_ids)
    
    # Check for duplicates across splits
    split_a_ids = set(video['id'] for video in split_info['train']['videos'])
    split_b_ids = set(video['id'] for video in split_info['validation']['videos'])
    split_c_ids = set(video['id'] for video in split_info['test']['videos'])
    
    ab_overlap = split_a_ids.intersection(split_b_ids)
    ac_overlap = split_a_ids.intersection(split_c_ids)
    bc_overlap = split_b_ids.intersection(split_c_ids)
    
    if len(ab_overlap) > 0:
        print(f"WARNING: {len(ab_overlap)} videos appear in both train and validation!")
    
    if len(ac_overlap) > 0:
        print(f"WARNING: {len(ac_overlap)} videos appear in both train and test!")
    
    if len(bc_overlap) > 0:
        print(f"WARNING: {len(bc_overlap)} videos appear in both validation and test!")
    
    if len(ab_overlap) == 0 and len(ac_overlap) == 0 and len(bc_overlap) == 0:
        print("No duplicates across splits ✓")
    
    # Final verification
    print(f"\nTotal unique videos across all splits: {len(all_split_ids)}")
    
    return all(len(x) == 0 for x in [ab_overlap, ac_overlap, bc_overlap])

if __name__ == "__main__":
    split_dataset()
    verify_splits("Raw_Dataset/splits", "Raw_Dataset/videos/videos")