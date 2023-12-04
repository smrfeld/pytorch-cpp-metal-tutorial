import torch
import my_extension

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

result = my_extension.add_tensors(a, b)
print(result)