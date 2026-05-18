import pytest
from src.data.dataset import DiseaseDataset

def test_dataset_initialization():
    image_paths = ["img1.jpg", "img2.jpg"]
    labels = [0, 1]
    class_names = ["healthy", "disease"]
    
    dataset = DiseaseDataset.from_paths_and_labels(image_paths, labels, class_names)
    assert len(dataset) == 2
    assert dataset.class_names == class_names
    assert dataset.get_class_distribution() == {"healthy": 1, "disease": 1}
