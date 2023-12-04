import torch

# Load the extension
my_extension_cpp = torch.ops.load_library("cpp_extension.so")

# Define a wrapper function
def add_tensors(a, b):
    return my_extension_cpp.add_tensors(a, b)