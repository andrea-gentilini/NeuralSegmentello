import os
import json
import random
import shutil
from pathlib import Path
from data.config import DIR_COCO_DSET, REDUCED_DSET_SIZE, DIR_COCO_DSET_ADAPTATION

def create_reduced_coco_dataset(full_dataset_path, N):
    # Setup 
    
    new_dataset_path = DIR_COCO_DSET_ADAPTATION / f"reduced_dset_{N}"

    full_path = Path(full_dataset_path)
    new_path = Path(new_dataset_path)
    full_ann_path = full_path / "annotations" / "instances_train2014.json"
    full_img_path = full_path / "train2014"
    
    new_ann_path = new_path / "annotations"
    new_img_path = new_path / "train2014"
    
    # Create new dirs
    new_ann_path.mkdir(parents=True, exist_ok=True)
    new_img_path.mkdir(parents=True, exist_ok=True)
    
    # Load full annotations
    with open(full_ann_path, "r") as f:
        coco_data = json.load(f)
    
    # Randomly sample N images
    selected_images = random.sample(coco_data["images"], N)
    selected_image_ids = {img["id"] for img in selected_images}
    
    # Filter annotations for selected images
    selected_annotations = [
        ann for ann in coco_data["annotations"]
        if ann["image_id"] in selected_image_ids
    ]
    
    # Filter categories if needed (keep all for now)
    new_coco_data = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco_data["categories"]
    }
    
    # Save reduced annotation file
    with open(new_ann_path / "instances_train2014.json", "w") as f:
        json.dump(new_coco_data, f)
    
    # Copy selected images
    for img in selected_images:
        src_file = full_img_path / img["file_name"]
        dst_file = new_img_path / img["file_name"]
        shutil.copy(src_file, dst_file)

    annotation_file_size = os.path.getsize(new_ann_path / "instances_train2014.json") / (1024 * 1024)  # MB
    img_dir_size = sum(os.path.getsize(new_img_path / img["file_name"]) for img in selected_images) / (1024 * 1024)  # MB
    
    print(f"Created reduced dataset at: {new_dataset_path} with {N} images.")
    print(f"Annotation file size: {annotation_file_size:.2f} MB")
    print(f"Image directory size: {img_dir_size:.2f} MB")

if __name__ == "__main__":
    create_reduced_coco_dataset(DIR_COCO_DSET, REDUCED_DSET_SIZE)