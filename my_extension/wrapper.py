import torch
import my_extension_cpp

# Define a wrapper function
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    # Find the shader file path
    import pkg_resources
    shader_file_path = pkg_resources.resource_filename('my_extension', 'add_tensors.metal')

    # Call the C++ function
    return my_extension_cpp.add_tensors_metal(a, b, shader_file_path)