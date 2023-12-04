#include <metal_stdlib>

using namespace metal;

kernel void add_tensors(
    device float* a [[ buffer(0) ]],
    device float* b [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]]) {
    
    result[gid] = a[gid] + b[gid];
}
