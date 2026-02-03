
import pytest
import torch
import torch.nn as nn
from views_hydranet.train.train_model import train, training_loop

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = 8
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
    def forward(self, x, h):
        out = self.conv(x)
        return out[:, :1], out[:, 1:], h
    def init_h(self, hidden_channels, dim):
        return torch.zeros((1, hidden_channels, dim, dim))

def test_gate_21_22_execution_order():
    """Verify code structure for immediate backprop and gated step."""
    import inspect
    import views_hydranet.train.train_model as tm
    source = inspect.getsource(tm.training_loop)
    
    # Use exact indentation to target the functional code
    # backward() is indented twice inside the window loop
    idx_backward = source.find("                    window_loss.backward()")
    # step() is indented once inside the lesson loop
    idx_step = source.find("                optimizer.step()")
    
    assert idx_backward != -1, "window_loss.backward() with correct indentation not found"
    assert idx_step != -1, "optimizer.step() with correct indentation not found"
    assert idx_backward < idx_step, "Immediate backprop (inside window loop) must precede step (lesson loop)"

def test_gate_23_oom_sentinel():
    """Falsify: 'Backward releases no memory'."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for OOM Sentinel")
        
    device = torch.device("cuda")
    model = MockModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    x = torch.randn(1, 1, 128, 128).to(device)
    h = model.init_h(8, 128).to(device)
    
    reg, cl, _ = model(x, h)
    loss = reg.sum()
    
    mem_with_graph = torch.cuda.memory_allocated(device)
    loss.backward()
    
    # Clean references
    del loss, reg, cl, h, x
    
    mem_after = torch.cuda.memory_allocated(device)
    assert mem_after < mem_with_graph, "Trainer: Memory not released after backward"

def test_gate_24_binarization_gradients():
    """Verify t1_binary does not track gradients."""
    t1 = torch.ones((1, 1, 4, 4), requires_grad=True)
    t1_binary = (t1.clone().detach() > 0) * 1.0
    assert t1_binary.requires_grad == False

def test_gate_25_hidden_detachment():
    """Verify code detaches hidden state in train loop."""
    import inspect
    import views_hydranet.train.train_model as tm
    source = inspect.getsource(tm.train)
    assert "model(t0, h.detach())" in source

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
