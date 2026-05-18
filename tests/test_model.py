import pytest
import torch

from src.models.backbone import Backbone
from src.models.crop_head import CropHead
from src.models.disease_model import DiseaseModel

def test_backbone_output_shape():
    backbone = Backbone(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features = backbone(x)
    assert features.shape == (2, 1280)

def test_crop_head_forward():
    head = CropHead(num_classes=5, feature_dim=1280)
    features = torch.randn(2, 1280)
    logits = head(features)
    assert logits.shape == (2, 5)

def test_crop_head_expand():
    head = CropHead(num_classes=3, class_names=["a", "b", "c"], crop_name="test")
    expanded = head.expand(["d", "e"])
    
    assert expanded.num_classes == 5
    assert expanded.class_names == ["a", "b", "c", "d", "e"]
    
    # Check that old weights are preserved
    assert torch.allclose(head.classifier[0].weight, expanded.classifier[0].weight)
    assert torch.allclose(head.classifier[4].weight[:3, :], expanded.classifier[4].weight[:3, :])

def test_disease_model():
    backbone = Backbone(pretrained=False)
    head = CropHead(num_classes=3, crop_name="test")
    model = DiseaseModel(backbone=backbone, crop_head=head)
    
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 3)
