import os
import json
import shutil
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Dict, List

class COCOImageProcessor:
    """
    A class to process images and create COCO dataset structures. 
    It processes images from a source directory, generates COCO annotations, 
    and saves them into separate JSON files for training, validation, and testing splits.
    """

    def __init__(self, source_directory: str, export_directory: str):
        """
        Initializes the COCOImageProcessor with source and export directories.
        :param source_directory: Directory containing the source images.
        :param export_directory: Directory where the processed data and COCO JSON files will be saved.
        """
        self.source_directory = source_directory
        self.export_directory = export_directory
        self.valid_classes = ["tanker", "tug", "other type", "passenger", "cargo"]
        self.class_to_id = {class_name: i+1 for i, class_name in enumerate(self.valid_classes)}
        self.image_id = 1
        self.annotation_id = 1

    def create_coco_structure(self) -> Dict[str, List]:
        """
        Creates the basic structure of a COCO dataset JSON file.
        :return: A dictionary with keys 'images', 'annotations', and 'categories', each mapping to an empty list.
        """
        return {"images": [], "annotations": [], "categories": []}

    def add_image_annotation(self, coco_structure: Dict[str, List], image_info: Dict, annotation_info: Dict) -> None:
        """
        Adds image and annotation information to the COCO dataset structure.
        :param coco_structure: The COCO dataset structure.
        :param image_info: Dictionary containing information about the image.
        :param annotation_info: Dictionary containing information about the annotation.
        """
        coco_structure["images"].append(image_info)
        coco_structure["annotations"].append(annotation_info)

    def create_image_info(self, image_path: str, image_id: int, split: str, class_name: str) -> Dict:
        """
        Creates information dictionary for a single image.
        :param image_path: Path to the image file.
        :param image_id: Unique identifier for the image.
        :param split: The dataset split ('train', 'val', or 'test') the image belongs to.
        :param class_name: The class name associated with the image.
        :return: Dictionary containing image information.
        """
        image = Image.open(image_path)
        width, height = image.size
        return {
            "file_name": os.path.join(self.export_directory, split, class_name, os.path.basename(image_path)),
            "height": height,
            "width": width,
            "id": image_id
        }

    def create_annotation_info(self, category_id: int, image_id: int, width: int, height: int, annotation_id: int) -> Dict:
        """
        Creates information dictionary for a single annotation.
        :param category_id: ID of the category the annotation belongs to.
        :param image_id: ID of the image the annotation is associated with.
        :param width: Width of the annotated object.
        :param height: Height of the annotated object.
        :param annotation_id: Unique identifier for the annotation.
        :return: Dictionary containing annotation information.
        """
        return {
            "id": annotation_id,
            "segmentation": [[0, 0, width, 0, width, height, 0, height]],
            "category_id": category_id,
            "image_id": image_id,
            "bbox": [0, 0, width, height],
            "area": width * height
        }

    def create_category_info(self) -> List[Dict]:
        """
        Creates a list of dictionaries, each representing a category.
        :return: List of dictionaries, where each dictionary has 'id' and 'name' keys.
        """
        return [{"id": id, "name": class_name} for class_name, id in self.class_to_id.items()]

    def process_images(self) -> None:
        """
        Processes all images in the source directory, splitting them into training, validation, and test sets.
        It creates COCO-style annotations and save them into JSON files.
        """
        files = [f for f in os.listdir(self.source_directory) if f.lower().endswith('.tif')]
        train_files, val_test_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=42)

        coco_structures = {
            "train": self.create_coco_structure(),
            "val": self.create_coco_structure(),
            "test": self.create_coco_structure()
        }

        for split in coco_structures:
            coco_structures[split]["categories"] = self.create_category_info()

        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            for file_name in tqdm(split_files, desc=f"Processing {split} images"):
                class_name = next((c for c in self.valid_classes if c in file_name.lower()), None)
                if class_name:
                    category_id = self.class_to_id[class_name]
                    image_path = os.path.join(self.source_directory, file_name)
                    image_info = self.create_image_info(image_path, self.image_id, split, class_name)
                    annotation_info = self.create_annotation_info(category_id, self.image_id, image_info["width"], image_info["height"], self.annotation_id)

                    self.add_image_annotation(coco_structures[split], image_info, annotation_info)

                    class_directory = os.path.join(self.export_directory, split, class_name)
                    if not os.path.exists(class_directory):
                        os.makedirs(class_directory)

                    shutil.copy(image_path, os.path.join(class_directory, file_name))

                    self.image_id += 1
                    self.annotation_id += 1

        # Save separate COCO files for each split
        annotations_dir = os.path.join(self.export_directory, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            json_path = os.path.join(annotations_dir, f'{split}.json')
            with open(json_path, 'w') as json_file:
                json.dump(coco_structures[split], json_file, indent=4)

