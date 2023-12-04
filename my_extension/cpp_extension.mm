#include <iostream>
#include <Metal/Metal.h>
#include <torch/extension.h>

// Define a simple function to add tensors using Metal shader
torch::Tensor add_tensors_metal(torch::Tensor a, torch::Tensor b) {
    // Get the Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal is not supported on this device.");
    }

    // Load the Metal shader from a .metal file (assuming add_tensors.metal is in the same directory)
    NSError* error = nil;
    NSString* shaderSource = [NSString stringWithContentsOfFile:@"add_tensors.metal" 
                                                       encoding:NSUTF8StringEncoding 
                                                          error:&error];
    if (error) {
        throw std::runtime_error("Failed to load Metal shader: " + std::string(error.localizedDescription.UTF8String));
    }

    MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
    compileOptions.preprocessorMacros = @{@"MTL_LANGUAGE_VERSION": @2};

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:compileOptions error:&error];
    if (error) {
        throw std::runtime_error("Failed to compile Metal shader: " + std::string(error.localizedDescription.UTF8String));
    }

    id<MTLFunction> function = [library newFunctionWithName:@"add_tensors"];

    // Create buffers for input and output data
    int dataSize = a.numel() * sizeof(float); // Assuming a and b are float tensors
    id<MTLBuffer> bufferA = [device newBufferWithBytes:a.data_ptr(), dataSize, MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:b.data_ptr(), dataSize, MTLResourceStorageModeShared];
    id<MTLBuffer> bufferResult = [device newBufferWithLength:dataSize options:MTLResourceStorageModeShared];

    // Create a compute pipeline
    MTLComputePipelineDescriptor* pipelineDesc = [[MTLComputePipelineDescriptor alloc] init];
    pipelineDesc.computeFunction = function;

    NSError* pipelineError = nil;
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithDescriptor:pipelineDesc error:&pipelineError];
    if (pipelineError) {
        throw std::runtime_error("Failed to create compute pipeline: " + std::string(pipelineError.localizedDescription.UTF8String));
    }

    // Create a compute command buffer
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Set input and output buffers
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferResult offset:0 atIndex:2];

    // Calculate threadgroup and grid sizes based on tensor size
    MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
    MTLSize threadgroupCount = MTLSizeMake((a.numel() + threadgroupSize.width - 1) / threadgroupSize.width, 1, 1);

    // Dispatch the compute kernel
    [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    // Commit the command buffer and wait for completion
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Create a new torch::Tensor from the result buffer
    torch::Tensor result = torch::empty({a.size(0)}, torch::dtype(torch::kFloat32));
    memcpy(result.data_ptr(), [bufferResult contents], dataSize);

    return result;
}

// Bind the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors", &add_tensors_metal, "Add two tensors using Metal shader");
}
