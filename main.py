import torch
import my_extension

a = torch.tensor([1.0, 2.0, 3.0]).to('mps')
b = torch.tensor([4.0, 5.0, 6.0]).to('mps')
print(f"Input tensor a: {a}")
print(f"Input tensor b: {b}")
print(f"Input device: {a.device}")

result = my_extension.add_tensors(a, b)
print(f"Addition result: {result}")
print(f"Output device {result.device}")
assert result.device == torch.device('mps:0'), "Output tensor is (maybe?) not on the MPS device"