import pytest
import torch
from src.training.losses import CombinedLoss

def test_combined_loss():
    loss_fn = CombinedLoss(alpha=0.7, temperature=2.0, old_num_classes=3)
    
    logits = torch.randn(4, 5) # Batch of 4, 5 new classes
    targets = torch.tensor([0, 1, 3, 4])
    old_logits = torch.randn(4, 3) # Batch of 4, 3 old classes
    
    loss = loss_fn(logits, targets, old_logits)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    
def test_combined_loss_no_old_logits():
    loss_fn = CombinedLoss(alpha=0.7, temperature=2.0, old_num_classes=0)
    
    logits = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 3, 4])
    
    loss = loss_fn(logits, targets)
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
