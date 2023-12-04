import torch

# Load the extension
my_extension_cpp = torch.ops.load_library("cpp_extension.so")

# Define a wrapper function
def initialize_metal():
    return my_extension_cpp.initialize_metal()