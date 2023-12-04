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
