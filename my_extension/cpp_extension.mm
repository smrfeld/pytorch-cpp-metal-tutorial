#include <torch/extension.h>
#include <Metal/Metal.h>

// Define a simple function
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

// Bind the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors, "Add two tensors");
}
