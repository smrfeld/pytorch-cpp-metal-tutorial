# Tutorial for custom Metal shaders using PyTorch & C++

This is a minimal example of a Python package calling a custom PyTorch C++ module that is using **Metal** shader (on Mac).

See also the associated [Medium](https://medium.com/practical-coding/metal-shaders-with-pytorch-from-end-to-end-c95370b3449b) article: [Metal shaders with PyTorch from end to end](https://medium.com/practical-coding/metal-shaders-with-pytorch-from-end-to-end-c95370b3449b)

## Installing & running

0. (Optional) Create a conda environment:

    ```bash
    conda create -n test-pytorch-cpp python=3.11
    conda activate test-pytorch-cpp
    ```

1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

2. Install package using `setup.py`:
    ```bash
    pip install -e .
    ```

3. Run the test:
    ```bash
    python main.py
    ```
    Expected result:
    ```
    tensor([5., 7., 9.])
    ```

## Other good examples

* [https://github.com/open-mmlab/mmcv/blob/main/setup.py](https://github.com/open-mmlab/mmcv/blob/main/setup.py)

## About

Goal: We will write from scratch a Python library that compiles a Metal shader using `C++`/`Objective-C` and lets you call the method from Python using `pybind11`.

### Project setup

We will create from scratch a new Python package called my_extension. This package will expose a method to add two `PyTorch` Tensors together which are on `MPS` device using a custom Metal shader. Create a new directory with the following structure:
```
my_extension/
my_extension/__init__.py
my_extension/add_tensors.metal
my_extension/cpp_extension.mm
my_extension/wrapper.py
setup.py
```

Here the package is build out of the `my_extension` folder. The wrapper.py contains the wrapper code that will call the compiled `C++` library. This is defined in the `cpp_extension.mm`, which mixes `C++` and `Objective-C` to call the shader `add_tensors.metal`.

### setup.py file

Let’s take a look at the [setup.py](setup.py) file. The main action happens in this if statement:
```python
    if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
```
where we check if `mps` is available. If so, we define how to handle `.mm` files which mix `Objective-C` and `C++`:
```python
        from distutils.unixccompiler import UnixCCompiler
        if '.mm' not in UnixCCompiler.src_extensions:
            UnixCCompiler.src_extensions.append('.mm')
            UnixCCompiler.language_map['.mm'] = 'objc'
and add the Metal framework:
        extra_compile_args = {}
        extra_compile_args['cxx'] = [
            '-Wall', 
            '-std=c++17',
            '-framework', 
            'Metal', 
            '-framework', 
            'Foundation',
            '-ObjC++'
            ]
```

There are two packages being defined here:
1. `my_extension` — this is the final Python package that we want to create. It is defined by the setup command in the last line:
    ```python
    setup(
        name='my_extension',
        version="0.0.1",
        packages=find_packages(),
        include_package_data=True,
        python_requires='>=3.11',
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
    )
    ```
2. `my_extension_cpp` — this is a `C++` library that will call the metal shader. It is defined through the `ext_modules` argument in the setup method, specifically in this line:
    ```python
    ext_ops = CppExtension(
        name='my_extension_cpp',
        sources=['my_extension/cpp_extension.mm'],
        include_dirs=[],
        extra_objects=[],
        extra_compile_args=extra_compile_args,
        library_dirs=[],
        libraries=[],
        extra_link_args=[]
        )
    ```

### Python Wrapper

We now have a project structure that creates a `C++` library called `my_extension_cpp` and a Python package called `my_extension`.

Next, let’s look at the Python wrapper `wrapper.py` defined in `my_extension/wrapper.py`:
```python
import torch
import my_extension_cpp

# Define a wrapper function
def add_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    # Find the shader file path
    import pkg_resources
    shader_file_path = pkg_resources.resource_filename('my_extension', 'add_tensors.metal')

    # Call the C++ function
    return my_extension_cpp.add_tensors_metal(a, b, shader_file_path)
```

Here we just expose the methods defined in `my_extension_cpp` to the `Python` interface. This adds one extra layer between the `C++` interface and the `Python` interface, which is often very useful as the usage can be quite different. For example, here we locate the .metal shader file using `Python` and pass it as argument to the `C++` extension function `add_tensors_metal(...)`.

Don’t forget to also expose this in the `__init__.py`:
```python
from .wrapper import add_tensors
```

### Metal shader

Let’s take a look at the actual Metal shader we want to use — `add_tensors.metal`:
```metal
#include <metal_stdlib>
using namespace metal;

// Define a simple kernel function to add two tensors
kernel void addTensors(device float *a [[buffer(0)]],
                       device float *b [[buffer(1)]],
                       device float *result [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    // Perform addition if within tensor bounds
    result[id] = a[id] + b[id];
}
```

We import the metal framework `#include <metal_stdlib>`. In the `addTensors` method, we have the `[[thread_position_in_grid]]`. Straight from the [Apple docs](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups):

> [[thread_position_in_grid]] is an attribute qualifier. Attribute qualifiers, identifiable by their double square-bracket syntax, allow kernel parameters to be bound to resources and built-in variables, in this case the thread’s position in the grid to the kernel function.

It is the position of the thread in the threadgroup (threads make up thread groups; thread groups make up grids).

The result is written to the output buffer `float *result`. The device qualifier indicates that the pointer refers to memory on the GPU.

### Calling the Metal shader from C++

Finally, let’s write the [my_extension/cpp_extension.mm](my_extension/cpp_extension.mm) file in `C++` and `Objective-C` that calls the `.metal` shader.

There’s a lot to unpack here — let’s start at the very bottom:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors_metal", &add_tensors_metal, "Add two tensors using Metal");
}
```

This uses `pybind11` to expose the `add_tensors_metal` function to `Python`, so that we can call it in the `wrapper.py`.

In the actual function, we load the shader file and compile it:

```cpp
    // Get the default Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Load the Metal shader from the specified path
    NSError* error = nil;
    NSString* shaderSource = [
        NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shaderFilePath.c_str()]
        encoding:NSUTF8StringEncoding 
        error:&error];
    if (error) {
        throw std::runtime_error("Failed to load Metal shader: " + std::string(error.localizedDescription.UTF8String));
    }

    // Compile the Metal shader source
    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        throw std::runtime_error("Error compiling Metal shader: " + std::string(error.localizedDescription.UTF8String));
    }
```

Ensure that the function exists

```cpp
    id<MTLFunction> function = [library newFunctionWithName:@"addTensors"];
    if (!function) {
        throw std::runtime_error("Error: Metal function addTensors not found.");
    }
```

Convert the torch Tensors into buffers

```cpp
    // Create a Metal compute pipeline state
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:nil];

    // Create Metal buffers for the tensors
    id<MTLBuffer> aBuffer = [device newBufferWithBytes:a.data_ptr() length:(numElements * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuffer = [device newBufferWithBytes:b.data_ptr() length:(numElements * sizeof(float)) options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:(numElements * sizeof(float)) options:MTLResourceStorageModeShared];

    // Create a command queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // Create a command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Create a compute command encoder
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // Set the compute pipeline state
    [encoder setComputePipelineState:pipelineState];

    // Set the buffers
    [encoder setBuffer:aBuffer offset:0 atIndex:0];
    [encoder setBuffer:bBuffer offset:0 atIndex:1];
    [encoder setBuffer:resultBuffer offset:0 atIndex:2];
```

We set the grid size and thread group size as the max allowed:

```cpp
    // Dispatch the compute kernel
    MTLSize gridSize = MTLSizeMake(numElements, 1, 1);
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > numElements) {
        threadGroupSize = numElements;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
```

Execute the shader

```cpp
    // Commit the command buffer and wait for it to complete
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
```

And finally copy the result back to a `torch` Tensor:

```cpp
    // Create an empty tensor on the MPS device to hold the result
    torch::Tensor result = torch::empty({numElements}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kMPS));

    // Copy the result from the Metal buffer to the MPS tensor
    id<MTLBuffer> resultBufferMPS = [device newBufferWithBytesNoCopy:result.data_ptr()
                                                                length:(numElements * sizeof(float))
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];

    return result;
```

### Test run

Let’s create a test file `main.py` to execute the shader:

```python
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
```

which uses input Tensors on the `MPS` device, and should give the following output to verify that the result is on the `MPS` device:

```
Input tensor a: tensor([1., 2., 3.], device='mps:0')
Input tensor b: tensor([4., 5., 6.], device='mps:0')
Input device: mps:0
Addition result: tensor([4., 5., 6.], device='mps:0')
Output device mps:0
```

### Closing thoughts

Thanks for reading! I had a lot of fun learning about metal shaders and hope you did as well. I found this [example](https://github.com/open-mmlab/mmcv/blob/main/setup.py) `setup.py` file pretty useful to look at, as well as the [official Apple docs](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups) to understand threads, threadgroups and grids.