#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <iostream>

// Define a function to add tensors using Metal
torch::Tensor add_tensors_metal(torch::Tensor a, torch::Tensor b, const std::string& shaderFilePath) {
  
    // Check that device is MPS
    if (a.device().type() != torch::kMPS || b.device().type() != torch::kMPS) {
        throw std::runtime_error("Error: tensors must be on MPS device.");
    }

    // Check that tensors are contiguous
    // Contiguous means that the memory is contiguous
    a = a.contiguous();
    b = b.contiguous();

    // Get the total number of elements in the tensors
    int numElements = a.numel();

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

    id<MTLFunction> function = [library newFunctionWithName:@"addTensors"];
    if (!function) {
        throw std::runtime_error("Error: Metal function addTensors not found.");
    }

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

    // Dispatch the compute kernel
    MTLSize gridSize = MTLSizeMake(numElements, 1, 1);
    NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > numElements) {
        threadGroupSize = numElements;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    // Commit the command buffer and wait for it to complete
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Create an empty tensor on the MPS device to hold the result
    torch::Tensor result = torch::empty({numElements}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kMPS));

    // Copy the result from the Metal buffer to the MPS tensor
    id<MTLBuffer> resultBufferMPS = [device newBufferWithBytesNoCopy:result.data_ptr()
                                                                length:(numElements * sizeof(float))
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];

    // Copy the result back to a PyTorch tensor
    // The .clone() function is used to create a deep copy of the tensor. This is necessary because torch::from_blob creates a tensor that directly uses the memory from resultBuffer.contents, and this memory might be released or go out of scope. The cloned tensor still resides on the CPU.
    // torch::Tensor result = torch::from_blob(resultBuffer.contents, {numElements}, torch::kFloat).clone();
    // result = result.to(torch::kMPS);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors_metal", &add_tensors_metal, "Add two tensors using Metal");
}