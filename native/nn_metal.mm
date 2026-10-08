// NN-owned Metal kernels. Core lends the device queue and tensor buffers
// through the stable native-extension ABI; every neural-network formula,
// launch shape and autograd rule in this file belongs to NN.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <quidra/native_extension.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <atomic>
#include <map>
#include <mutex>
#include <vector>

namespace {

// Kernels are compiled in Metal's IEEE-preserving "safe" math mode so that
// infinities, NaNs and signed zeros follow the same rules as the CPU kernels.
// Convolution sums use explicit fma() in the same tap order as the CPU loops.
NSString* nn_metal_source() {
    static NSString* source = [[NSString alloc] initWithUTF8String:R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
// Every rounding step is written out explicitly (fma() where fused), so the
// activation expressions round exactly like the NN package source.
#pragma METAL fp contract(off)

// NCHW input, OIHW weight, NCHW output. Every extent and element count fits
// in 32 bits; the host validates that before launching.
struct ConvShape {
    uint batches;
    uint channels_in;
    uint height;
    uint width;
    uint channels_out;
    uint weight_channels;
    uint kernel_height;
    uint kernel_width;
    uint output_height;
    uint output_width;
    uint stride;
    uint padding;
    uint groups;
    uint outputs_per_group;
    uint output_count;
    uint input_count;
};

// Kernel templates take a square kernel size K as a compile-time constant
// for the common 1x1 and 3x3 cases (fully unrolled taps); K = 0 reads the
// (possibly non-square) size from the shape. Every variant visits taps in
// the same order, so the specialization never changes a result.

// Output channels computed per forward thread and input channels per dInput
// thread: neighbouring channels share every input/gradient load.
constant uint channel_block = 4;

// Direct convolution: one thread per (n, block of output channels, oy, ox).
// Each output value is a single fma chain over (input channel, ky, kx) in
// the CPU loop order, then + bias.
template <uint K>
static void conv2d_forward(
    device const float* input,
    device const float* weight,
    device const float* bias,
    device float* output,
    constant ConvShape& s,
    uint use_bias,
    uint gid) {
    const uint kernel_height = K != 0 ? K : s.kernel_height;
    const uint kernel_width = K != 0 ? K : s.kernel_width;
    const uint taps = kernel_height * kernel_width;
    const uint blocks = (s.outputs_per_group + channel_block - 1) / channel_block;
    const uint ox = gid % s.output_width;
    uint rest = gid / s.output_width;
    const uint oy = rest % s.output_height;
    rest /= s.output_height;
    const uint block = rest % (s.groups * blocks);
    const uint n = rest / (s.groups * blocks);
    if (n >= s.batches) return;
    const uint group = block / blocks;
    const uint first_oc = group * s.outputs_per_group + (block % blocks) * channel_block;
    const uint count = min(channel_block, group * s.outputs_per_group + s.outputs_per_group - first_oc);
    const int top = int(oy * s.stride) - int(s.padding);
    const int left = int(ox * s.stride) - int(s.padding);
    const uint ky_begin = uint(max(0, -top));
    const uint ky_end = uint(clamp(int(s.height) - top, 0, int(kernel_height)));
    const uint kx_begin = uint(max(0, -left));
    const uint kx_end = uint(clamp(int(s.width) - left, 0, int(kernel_width)));
    float sum[channel_block] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint first_channel = group * s.weight_channels;
    for (uint local = 0; local < s.weight_channels; ++local) {
        device const float* plane =
            input + (n * s.channels_in + first_channel + local) * s.height * s.width;
        device const float* filter = weight + (first_oc * s.weight_channels + local) * taps;
        const uint filter_step = s.weight_channels * taps;
        for (uint ky = ky_begin; ky < ky_end; ++ky) {
            device const float* row = plane + uint(top + int(ky)) * s.width;
            for (uint kx = kx_begin; kx < kx_end; ++kx) {
                const float value = row[left + int(kx)];
                const uint tap = ky * kernel_width + kx;
                for (uint j = 0; j < channel_block; ++j) {
                    if (j < count)
                        sum[j] = fma(value, filter[j * filter_step + tap], sum[j]);
                }
            }
        }
    }
    const uint plane_size = s.output_height * s.output_width;
    const uint position = oy * s.output_width + ox;
    for (uint j = 0; j < channel_block; ++j) {
        if (j < count) {
            const uint oc = first_oc + j;
            output[(n * s.channels_out + oc) * plane_size + position] =
                use_bias != 0u ? sum[j] + bias[oc] : sum[j];
        }
    }
}

#define NN_CONV2D_FORWARD(NAME, K)                                        \
kernel void NAME(                                                         \
    device const float* input [[buffer(0)]],                              \
    device const float* weight [[buffer(1)]],                             \
    device const float* bias [[buffer(2)]],                               \
    device float* output [[buffer(3)]],                                   \
    constant ConvShape& s [[buffer(4)]],                                  \
    constant uint& use_bias [[buffer(5)]],                                \
    uint gid [[thread_position_in_grid]]) {                               \
    conv2d_forward<K>(input, weight, bias, output, s, use_bias, gid);     \
}
NN_CONV2D_FORWARD(nn_conv2d_forward, 0)
NN_CONV2D_FORWARD(nn_conv2d_forward_k1, 1)
NN_CONV2D_FORWARD(nn_conv2d_forward_k3, 3)

// Largest tap index <= size - 1 congruent to `residue` modulo `stride`
// (negative when there is none).
static int last_tap(int size, int residue, int stride) {
    return size - 1 - ((size - 1 - residue) % stride + stride) % stride;
}

// dInput: one thread per (n, block of input channels, iy, ix) gathers the
// output positions its values reached, so no two threads write the same
// gradient element. Taps are visited in the CPU loop's (output channel, oy,
// ox) order: ky and kx descend in steps of the stride.
template <uint K>
static void conv2d_backward_data(
    device const float* weight,
    device const float* gradient_output,
    device float* gradient_input,
    constant ConvShape& s,
    uint gid) {
    const uint kernel_height = K != 0 ? K : s.kernel_height;
    const uint kernel_width = K != 0 ? K : s.kernel_width;
    const uint taps = kernel_height * kernel_width;
    const uint blocks = (s.weight_channels + channel_block - 1) / channel_block;
    const uint ix = gid % s.width;
    uint rest = gid / s.width;
    const uint iy = rest % s.height;
    rest /= s.height;
    const uint block = rest % (s.groups * blocks);
    const uint n = rest / (s.groups * blocks);
    if (n >= s.batches) return;
    const uint group = block / blocks;
    const uint first_local = (block % blocks) * channel_block;
    const uint count = min(channel_block, s.weight_channels - first_local);
    const uint plane = s.output_height * s.output_width;
    const int stride = int(s.stride);
    const int padded_y = int(iy + s.padding);
    const int padded_x = int(ix + s.padding);
    const int ky_first = last_tap(int(kernel_height), padded_y % stride, stride);
    const int kx_first = last_tap(int(kernel_width), padded_x % stride, stride);
    float sum[channel_block] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint local_output = 0; local_output < s.outputs_per_group; ++local_output) {
        const uint oc = group * s.outputs_per_group + local_output;
        device const float* gradient = gradient_output + (n * s.channels_out + oc) * plane;
        device const float* filter = weight + (oc * s.weight_channels + first_local) * taps;
        for (int ky = ky_first; ky >= 0; ky -= stride) {
            const int window_y = padded_y - ky;
            if (window_y < 0) continue;
            const uint oy = uint(window_y / stride);
            if (oy >= s.output_height) break;
            for (int kx = kx_first; kx >= 0; kx -= stride) {
                const int window_x = padded_x - kx;
                if (window_x < 0) continue;
                const uint ox = uint(window_x / stride);
                if (ox >= s.output_width) break;
                const float value = gradient[oy * s.output_width + ox];
                const uint tap = uint(ky) * kernel_width + uint(kx);
                for (uint j = 0; j < channel_block; ++j) {
                    if (j < count)
                        sum[j] = fma(value, filter[j * taps + tap], sum[j]);
                }
            }
        }
    }
    const uint input_plane = s.height * s.width;
    const uint position = iy * s.width + ix;
    const uint first_channel = group * s.weight_channels + first_local;
    for (uint j = 0; j < channel_block; ++j) {
        if (j < count)
            gradient_input[(n * s.channels_in + first_channel + j) * input_plane + position] = sum[j];
    }
}

#define NN_CONV2D_BACKWARD_DATA(NAME, K)                                  \
kernel void NAME(                                                         \
    device const float* weight [[buffer(0)]],                             \
    device const float* gradient_output [[buffer(1)]],                    \
    device float* gradient_input [[buffer(2)]],                           \
    constant ConvShape& s [[buffer(3)]],                                  \
    uint gid [[thread_position_in_grid]]) {                               \
    conv2d_backward_data<K>(weight, gradient_output, gradient_input, s, gid); \
}
NN_CONV2D_BACKWARD_DATA(nn_conv2d_backward_data, 0)
NN_CONV2D_BACKWARD_DATA(nn_conv2d_backward_data_k1, 1)
NN_CONV2D_BACKWARD_DATA(nn_conv2d_backward_data_k3, 3)

// Deterministic threadgroup tree reduction. `width` is a power of two no
// larger than 256 and equals the threadgroup size.
static float reduce_group(
    threadgroup float* partial, float value, uint lane, uint width) {
    partial[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_width = width / 2; half_width > 0; half_width /= 2) {
        if (lane < half_width) partial[lane] += partial[lane + half_width];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partial[0];
}

// dWeight: sums input * gradient_output over batch and output positions.
// Each lane walks a fixed strided slice of the (n, oy, ox) positions and the
// lanes are combined by a tree reduction, so the order depends only on the
// shape. K > 0: one threadgroup per (output channel, input channel) pair
// accumulates all K*K taps at once, sharing each gradient load and the
// position arithmetic. K = 0: one threadgroup per weight element. Both
// produce the same per-tap chain and reduction.
template <uint K>
static void conv2d_backward_filter(
    device const float* input,
    device const float* gradient_output,
    device float* gradient_weight,
    constant ConvShape& s,
    uint width,
    uint element,
    uint lane,
    threadgroup float* partial) {
    const uint taps = K * K;
    const uint local = element % s.weight_channels;
    const uint oc = element / s.weight_channels;
    const uint ic = (oc / s.outputs_per_group) * s.weight_channels + local;
    const uint plane = s.output_height * s.output_width;
    const uint total = s.batches * plane;
    float sum[taps];
    for (uint tap = 0; tap < taps; ++tap) sum[tap] = 0.0f;
    for (uint index = lane; index < total; index += width) {
        const uint n = index / plane;
        const uint position = index % plane;
        const uint oy = position / s.output_width;
        const uint ox = position % s.output_width;
        const float gradient =
            gradient_output[(n * s.channels_out + oc) * plane + position];
        device const float* source = input + (n * s.channels_in + ic) * s.height * s.width;
        const int top = int(oy * s.stride) - int(s.padding);
        const int left = int(ox * s.stride) - int(s.padding);
        for (uint ky = 0; ky < K; ++ky) {
            const int iy = top + int(ky);
            if (iy < 0 || iy >= int(s.height)) continue;
            for (uint kx = 0; kx < K; ++kx) {
                const int ix = left + int(kx);
                if (ix < 0 || ix >= int(s.width)) continue;
                sum[ky * K + kx] = fma(gradient, source[uint(iy) * s.width + uint(ix)], sum[ky * K + kx]);
            }
        }
    }
    for (uint tap = 0; tap < taps; ++tap) partial[tap * width + lane] = sum[tap];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_width = width / 2; half_width > 0; half_width /= 2) {
        if (lane < half_width) {
            for (uint tap = 0; tap < taps; ++tap)
                partial[tap * width + lane] += partial[tap * width + lane + half_width];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint tap = lane; tap < taps; tap += width)
        gradient_weight[element * taps + tap] = partial[tap * width];
}

kernel void nn_conv2d_backward_filter_k1(
    device const float* input [[buffer(0)]],
    device const float* gradient_output [[buffer(1)]],
    device float* gradient_weight [[buffer(2)]],
    constant ConvShape& s [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint element [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float partial[256];
    conv2d_backward_filter<1>(input, gradient_output, gradient_weight, s, width, element, lane, partial);
}

kernel void nn_conv2d_backward_filter_k3(
    device const float* input [[buffer(0)]],
    device const float* gradient_output [[buffer(1)]],
    device float* gradient_weight [[buffer(2)]],
    constant ConvShape& s [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint element [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float partial[9 * 256];
    conv2d_backward_filter<3>(input, gradient_output, gradient_weight, s, width, element, lane, partial);
}

kernel void nn_conv2d_backward_filter(
    device const float* input [[buffer(0)]],
    device const float* gradient_output [[buffer(1)]],
    device float* gradient_weight [[buffer(2)]],
    constant ConvShape& s [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint element [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float partial[256];
    const uint kx = element % s.kernel_width;
    uint rest = element / s.kernel_width;
    const uint ky = rest % s.kernel_height;
    rest /= s.kernel_height;
    const uint local = rest % s.weight_channels;
    const uint oc = rest / s.weight_channels;
    const uint ic = (oc / s.outputs_per_group) * s.weight_channels + local;
    const uint plane = s.output_height * s.output_width;
    const uint total = s.batches * plane;
    float sum = 0.0f;
    for (uint index = lane; index < total; index += width) {
        const uint n = index / plane;
        const uint position = index % plane;
        const uint oy = position / s.output_width;
        const uint ox = position % s.output_width;
        const int iy = int(oy * s.stride + ky) - int(s.padding);
        const int ix = int(ox * s.stride + kx) - int(s.padding);
        if (iy < 0 || iy >= int(s.height) || ix < 0 || ix >= int(s.width))
            continue;
        sum = fma(gradient_output[(n * s.channels_out + oc) * plane + position],
                  input[((n * s.channels_in + ic) * s.height + uint(iy)) *
                            s.width + uint(ix)],
                  sum);
    }
    const float total_sum = reduce_group(partial, sum, lane, width);
    if (lane == 0) gradient_weight[element] = total_sum;
}

// dBias: one threadgroup per output channel.
kernel void nn_conv2d_backward_bias(
    device const float* gradient_output [[buffer(0)]],
    device float* gradient_bias [[buffer(1)]],
    constant ConvShape& s [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint oc [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float partial[256];
    const uint plane = s.output_height * s.output_width;
    const uint total = s.batches * plane;
    float sum = 0.0f;
    for (uint index = lane; index < total; index += width) {
        const uint n = index / plane;
        const uint position = index % plane;
        sum += gradient_output[(n * s.channels_out + oc) * plane + position];
    }
    const float total_sum = reduce_group(partial, sum, lane, width);
    if (lane == 0) gradient_bias[oc] = total_sum;
}

// NN activation ids: 1 = ReLU expression (x + |x|) / 2, 2 = GELU expression
// x / 2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))) with nn.tanh's clamp of
// the argument to [-20, 20] and tanh(z) = (e^(2z) - 1) / (e^(2z) + 1).
constant float gelu_cubic = 0.044715f;
constant float gelu_scale = 0.7978845608028654f;

// math.abs derivative, including 0 at 0.
static float abs_derivative(float value) {
    return value < 0.0f ? -1.0f : (value > 0.0f ? 1.0f : 0.0f);
}

static float gelu_value(float value) {
    const float square = value * value;
    const float cubic = square * value;
    const float inner = (value + cubic * gelu_cubic) * gelu_scale;
    const float lifted = ((inner - 20.0f) + fabs(inner + 20.0f)) * 0.5f;
    const float limited = ((lifted + 20.0f) - fabs(lifted - 20.0f)) * 0.5f;
    const float exponent = exp(limited * 2.0f);
    const float hyperbolic = (exponent - 1.0f) / (exponent + 1.0f);
    return (value * 0.5f) * (1.0f + hyperbolic);
}

// First and second derivative of gelu_value. The clamp contributes the same
// piecewise factor as the compositional graph: 1 inside, 0 outside and 1/2
// exactly on a boundary (math.abs has derivative 0 at 0 and no curvature).
static float2 gelu_derivatives(float value) {
    const float square = value * value;
    const float cubic = square * value;
    const float inner = (value + cubic * gelu_cubic) * gelu_scale;
    const float lower = inner + 20.0f;
    const float lifted = ((inner - 20.0f) + fabs(lower)) * 0.5f;
    const float upper = lifted - 20.0f;
    const float limited = ((lifted + 20.0f) - fabs(upper)) * 0.5f;
    const float exponent = exp(limited * 2.0f);
    const float denominator = exponent + 1.0f;
    const float hyperbolic = (exponent - 1.0f) / denominator;
    const float clamp = ((1.0f + abs_derivative(lower)) * 0.5f) *
                        ((1.0f - abs_derivative(upper)) * 0.5f);
    const float slope = (4.0f * exponent) / (denominator * denominator);
    const float inner_slope = slope * clamp;
    const float inner_derivative =
        gelu_scale * (1.0f + (3.0f * gelu_cubic) * square);
    const float first = 0.5f * (1.0f + hyperbolic) +
                        ((0.5f * value) * inner_slope) * inner_derivative;
    // tanh'' times the square of d limited / dx = clamp * inner_derivative:
    // exactly 0 outside the clamp window, where inner_derivative^2 alone
    // overflows for |x| >= 1.3128e10 and clamp^2 * inf would be NaN; inside
    // it (clamp 1, or 1/2 on a boundary) the roundings are unchanged.
    const float curvature = (-2.0f * hyperbolic) * slope;
    const float limited_derivative = clamp * inner_derivative;
    const float inner_curvature = gelu_scale * ((6.0f * gelu_cubic) * value);
    const float second =
        inner_slope * inner_derivative +
        (0.5f * value) *
            (curvature * (limited_derivative * limited_derivative) +
             inner_slope * inner_curvature);
    return float2(first, second);
}

kernel void nn_activation_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& activation [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const float value = input[gid];
    output[gid] = activation == 1u ? (value + fabs(value)) * 0.5f
                                   : gelu_value(value);
}

kernel void nn_activation_backward(
    device const float* input [[buffer(0)]],
    device const float* gradient_output [[buffer(1)]],
    device float* gradient_input [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& activation [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const float value = input[gid];
    const float upstream = gradient_output[gid];
    if (activation == 1u) {
        // The compositional ReLU graph: s = g / 2; dx = s + s * abs'(x).
        const float half_upstream = upstream * 0.5f;
        gradient_input[gid] =
            half_upstream + half_upstream * abs_derivative(value);
    } else {
        gradient_input[gid] = upstream * gelu_derivatives(value).x;
    }
}

// Adjoint of dx = g * f'(x) with respect to x and g.
kernel void nn_activation_second_backward(
    device const float* input [[buffer(0)]],
    device const float* first_gradient [[buffer(1)]],
    device const float* gradient_output [[buffer(2)]],
    device float* gradient_input [[buffer(3)]],
    device float* gradient_first [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& activation [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const float value = input[gid];
    const float upstream = gradient_output[gid];
    if (activation == 1u) {
        const float first_half = first_gradient[gid] * 0.5f;
        gradient_input[gid] = (upstream * first_half) * 0.0f;
        gradient_first[gid] =
            upstream * 0.5f + (upstream * abs_derivative(value)) * 0.5f;
    } else {
        const float2 derivatives = gelu_derivatives(value);
        gradient_input[gid] =
            (upstream * first_gradient[gid]) * derivatives.y;
        gradient_first[gid] = upstream * derivatives.x;
    }
}

// Global average pooling over each NCHW plane: one threadgroup per (n, c)
// row. Small planes (<= 64 values) use one lane, i.e. the sequential
// left-to-right sum of the CPU kernel; larger planes use the fixed-shape
// strided reduction.
kernel void nn_global_average_pool(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& plane [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float partial[256];
    device const float* values = input + row * plane;
    float sum = 0.0f;
    if (width == 1u) {
        sum = values[0];
        for (uint index = 1; index < plane; ++index) sum = sum + values[index];
    } else {
        for (uint index = lane; index < plane; index += width)
            sum = sum + values[index];
    }
    const float total = width == 1u ? sum : reduce_group(partial, sum, lane, width);
    if (lane == 0) output[row] = total / float(plane);
}

// Adjoint of global average pooling: every plane element receives its row
// value divided by the plane size.
kernel void nn_global_average_unpool(
    device const float* gradient [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& plane [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    output[gid] = gradient[gid / plane] / float(plane);
}

// Fused Adam update with bias correction folded into epsilon and scale by
// the caller. The operations are those of nn.Adam's element-wise update, in
// its order and each rounded separately: m = m * b1 + g * (1 - b1);
// v = v * b2 + g * g * (1 - b2); p - (m / (sqrt(v) + eps)) * scale. The
// element-wise update runs on Core's and Math's Metal kernels, which are
// compiled with Metal's default fast math; fast::sqrt and fast::divide are
// those kernels' square root and division, so this kernel reproduces the
// element-wise Metal update bit for bit (tests/metal_adam.qui). Everything
// else is exact IEEE arithmetic in both compilations.
struct AdamCoefficients {
    float beta1;
    float beta2;
    float one_minus_beta1;
    float one_minus_beta2;
    float epsilon;
    float scale;
    uint count;
};

kernel void nn_adam_step(
    device const float* parameter [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    device const float* first [[buffer(2)]],
    device const float* second [[buffer(3)]],
    device float* next_parameter [[buffer(4)]],
    device float* next_first [[buffer(5)]],
    device float* next_second [[buffer(6)]],
    constant AdamCoefficients& c [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= c.count) return;
    const float g = gradient[gid];
    const float m = first[gid] * c.beta1 + g * c.one_minus_beta1;
    const float v = second[gid] * c.beta2 + (g * g) * c.one_minus_beta2;
    const float denominator = fast::sqrt(v) + c.epsilon;
    next_first[gid] = m;
    next_second[gid] = v;
    next_parameter[gid] =
        parameter[gid] - fast::divide(m, denominator) * c.scale;
}

// Second-order dBias adjoint: broadcast a per-channel value over NCHW.
kernel void nn_conv2d_bias_broadcast(
    device const float* bias_gradient [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ConvShape& s [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= s.output_count) return;
    const uint plane = s.output_height * s.output_width;
    output[gid] = bias_gradient[(gid / plane) % s.channels_out];
}
)MSL"];
    return source;
}

// NN's main kernels request IEEE-preserving math through the compile options
// as well as the in-source pragma, so no toolchain that ignores the pragma
// can fall back to fast math.
MTLCompileOptions* nn_metal_safe_options() {
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, iOS 18.0, *)) {
        options.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    return options;
}

// Pipelines are compiled once per MTLDevice. The enum order must match
// kernel_names.
enum NnMetalKernel : std::size_t {
    kernel_conv2d_forward,
    kernel_conv2d_forward_k1,
    kernel_conv2d_forward_k3,
    kernel_conv2d_backward_data,
    kernel_conv2d_backward_data_k1,
    kernel_conv2d_backward_data_k3,
    kernel_conv2d_backward_filter,
    kernel_conv2d_backward_filter_k1,
    kernel_conv2d_backward_filter_k3,
    kernel_conv2d_backward_bias,
    kernel_conv2d_bias_broadcast,
    kernel_activation_forward,
    kernel_activation_backward,
    kernel_activation_second_backward,
    kernel_global_average_pool,
    kernel_global_average_unpool,
    kernel_adam_step,
    nn_metal_kernel_count
};

const char* const kernel_names[nn_metal_kernel_count] = {
    "nn_conv2d_forward",
    "nn_conv2d_forward_k1",
    "nn_conv2d_forward_k3",
    "nn_conv2d_backward_data",
    "nn_conv2d_backward_data_k1",
    "nn_conv2d_backward_data_k3",
    "nn_conv2d_backward_filter",
    "nn_conv2d_backward_filter_k1",
    "nn_conv2d_backward_filter_k3",
    "nn_conv2d_backward_bias",
    "nn_conv2d_bias_broadcast",
    "nn_activation_forward",
    "nn_activation_backward",
    "nn_activation_second_backward",
    "nn_global_average_pool",
    "nn_global_average_unpool",
    "nn_adam_step",
};

struct NnMetalPrograms {
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pipelines[nn_metal_kernel_count] = {};
    bool ready = false;
};

std::mutex programs_mutex;
std::map<std::uintptr_t, NnMetalPrograms> programs_by_device;

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    const char* name) {
    NSString* function_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (!function) return nil;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    return pipeline;
}

NnMetalPrograms* programs_for(id<MTLDevice> device) {
    if (!device) return nullptr;
    const auto key = reinterpret_cast<std::uintptr_t>((__bridge void*)device);
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto [it, inserted] = programs_by_device.try_emplace(key);
    auto& programs = it->second;
    if (inserted) {
        NSError* error = nil;
        MTLCompileOptions* options = nn_metal_safe_options();
        programs.library = [device newLibraryWithSource:nn_metal_source()
                                                options:options
                                                  error:&error];
        [options release];
        bool ready = programs.library != nil;
        for (std::size_t index = 0; ready && index < nn_metal_kernel_count;
             ++index) {
            programs.pipelines[index] =
                make_pipeline(device, programs.library, kernel_names[index]);
            ready = programs.pipelines[index] != nil;
        }
        programs.ready = ready;
    }
    return programs.ready ? &programs : nullptr;
}

id<MTLCommandQueue> command_queue(const void* tensor) {
    const auto device = qcore_tensor_device(tensor);
    if (device < 0) return nil;
    const auto handle = qcore_device_queue_handle(device);
    if (handle == 0) return nil;
    return (__bridge id<MTLCommandQueue>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

id<MTLBuffer> const_buffer(const void* tensor) {
    const auto handle = qcore_tensor_device_handle_const(tensor);
    if (handle == 0) return nil;
    return (__bridge id<MTLBuffer>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

id<MTLBuffer> mutable_buffer(void* tensor) {
    const auto handle = qcore_tensor_device_handle(tensor);
    if (handle == 0) return nil;
    return (__bridge id<MTLBuffer>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

// Dense float32 tensor on a Metal device. Views with offsets are fine: the
// buffer offset is applied when binding.
bool metal_f32(const void* tensor) {
    return tensor &&
           qcore_tensor_dtype(tensor) == QCORE_DTYPE_FLOAT32 &&
           qcore_tensor_backend(tensor) == QCORE_BACKEND_METAL &&
           qcore_tensor_is_contiguous(tensor) != 0;
}

bool same_device(const void* left, const void* right) {
    return qcore_tensor_device(left) == qcore_tensor_device(right);
}

bool fits_u32(unsigned long long value) {
    return value <= std::numeric_limits<std::uint32_t>::max();
}

bool bind_const(
    id<MTLComputeCommandEncoder> encoder,
    const void* tensor,
    NSUInteger index) {
    id<MTLBuffer> buffer = const_buffer(tensor);
    if (!buffer) return false;
    [encoder setBuffer:buffer
                offset:static_cast<NSUInteger>(
                    qcore_tensor_device_offset_bytes(tensor))
               atIndex:index];
    return true;
}

bool bind_mutable(
    id<MTLComputeCommandEncoder> encoder,
    void* tensor,
    NSUInteger index) {
    id<MTLBuffer> buffer = mutable_buffer(tensor);
    if (!buffer) return false;
    [encoder setBuffer:buffer
                offset:static_cast<NSUInteger>(
                    qcore_tensor_device_offset_bytes(tensor))
               atIndex:index];
    return true;
}

NSUInteger group_width(id<MTLComputePipelineState> pipeline) {
    NSUInteger width = 256;
    const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
    if (maximum != 0 && width > maximum) width = maximum;
    return width == 0 ? 1 : width;
}

void dispatch_elements(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t count) {
    [encoder setComputePipelineState:pipeline];
    if (count == 0) return;
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(group_width(pipeline), 1, 1)];
}

// Threadgroup width for a deterministic reduction over `total` terms: about
// 64 terms per lane, a power of two in [1, 256]. It depends only on the shape.
std::uint32_t reduction_width(
    id<MTLComputePipelineState> pipeline, std::uint32_t total) {
    std::uint32_t width = 1;
    while (width < 256 && static_cast<std::uint64_t>(width) * 64 < total)
        width *= 2;
    const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
    while (width > 1 && maximum != 0 && width > maximum) width /= 2;
    return width;
}

void dispatch_groups(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t groups,
    std::uint32_t width) {
    [encoder setComputePipelineState:pipeline];
    if (groups == 0) return;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

// NN Metal dispatches by kernel family. The native kernels and the portable
// fallback give the same values, so tests read these counts
// (nn_metal_dispatch_count) to prove that the native path ran.
enum DispatchKind : int {
    dispatch_conv2d_forward = 1,
    dispatch_conv2d_backward = 2,
    dispatch_conv2d_backward_data = 3,
    dispatch_conv2d_backward_filter = 4,
    dispatch_conv2d_bias_broadcast = 5,
    dispatch_activation_forward = 6,
    dispatch_activation_backward = 7,
    dispatch_activation_backward_internal = 8,
    dispatch_activation_second_backward = 9,
    dispatch_global_average_pool = 10,
    dispatch_global_average_unpool = 11,
    dispatch_adam_update = 12,
    dispatch_conv2d_activation_forward = 13,
    dispatch_kinds = 14,
};

std::atomic<std::uint64_t> dispatch_counts[dispatch_kinds];

void count_dispatch(DispatchKind kind, std::uint64_t count = 1) {
    dispatch_counts[kind].fetch_add(count, std::memory_order_relaxed);
}

// Command buffers this thread committed without waiting (submit(..., false))
// whose completion status nobody has checked yet. Each holds a retain.
thread_local std::vector<id<MTLCommandBuffer>> unchecked_commands;

// Waits for every unchecked command of this thread and releases it. Returns
// false when any of them did not complete successfully, so a GPU failure of
// an unwaited command is reported by the next waiting NN command on the
// thread (at the latest the NN kernel that reads its result) instead of
// being lost.
bool settle_unchecked_commands() {
    bool completed = true;
    for (id<MTLCommandBuffer> command : unchecked_commands) {
        [command waitUntilCompleted];
        if ([command status] != MTLCommandBufferStatusCompleted)
            completed = false;
        [command release];
    }
    unchecked_commands.clear();
    return completed;
}

// Encodes one compute pass on Core's queue for `tensor`'s device, commits it
// and waits for completion. Core's host-read synchronization only waits for
// command buffers that Core itself committed (and skips its queue barrier
// when none are pending), so an asynchronous package command could still be
// in flight when Core copies the result to the host. Waiting here keeps
// every NN Metal result visible to Core as soon as the call returns.
//
// `wait = false` is reserved for results that only NN kernels on the same
// queue ever read (see nn_metal_activation_backward_internal): queue order
// then guarantees they are complete before any reader runs. Such a command
// is kept as unchecked; the reader is a waiting submit on the same thread,
// which also checks the earlier command's status and fails if it failed.
// A command that was committed counts as one dispatch of `kind` once it
// completed (or, without waiting, once it was committed).
template <typename Encode>
int submit(
    DispatchKind kind, const void* tensor, Encode&& encode, bool wait = true) {
    id<MTLCommandQueue> queue = command_queue(tensor);
    if (!queue) return 3;
    NnMetalPrograms* programs = programs_for([queue device]);
    if (!programs) return 5;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command) return 5;
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) return 5;
    const int status = encode(encoder, *programs);
    [encoder endEncoding];
    if (status != 0) return status;
    [command commit];
    if (!wait) {
        unchecked_commands.push_back([command retain]);
        count_dispatch(kind);
        return 0;
    }
    [command waitUntilCompleted];
    const bool earlier = settle_unchecked_commands();
    if (!earlier || [command status] != MTLCommandBufferStatusCompleted)
        return 6;
    count_dispatch(kind);
    return 0;
}

// Host mirror of the MSL ConvShape.
struct ConvShape {
    std::uint32_t batches;
    std::uint32_t channels_in;
    std::uint32_t height;
    std::uint32_t width;
    std::uint32_t channels_out;
    std::uint32_t weight_channels;
    std::uint32_t kernel_height;
    std::uint32_t kernel_width;
    std::uint32_t output_height;
    std::uint32_t output_width;
    std::uint32_t stride;
    std::uint32_t padding;
    std::uint32_t groups;
    std::uint32_t outputs_per_group;
    std::uint32_t output_count;
    std::uint32_t input_count;
};

// Validates NCHW/OIHW convolution geometry exactly like the CPU kernels and
// additionally requires every tensor to be dense float32 on the same Metal
// device with 32-bit addressable extents.
bool conv_shape(
    const void* input,
    const void* weight,
    const void* output,
    long long stride,
    long long padding,
    long long groups,
    ConvShape& shape) {
    if (!metal_f32(input) || !metal_f32(weight) || !metal_f32(output) ||
        !same_device(input, weight) || !same_device(input, output) ||
        qcore_tensor_rank(input) != 4 || qcore_tensor_rank(weight) != 4 ||
        qcore_tensor_rank(output) != 4 ||
        stride <= 0 || padding < 0 || groups <= 0) {
        return false;
    }
    const long long batches = qcore_tensor_extent(input, 0);
    const long long channels_in = qcore_tensor_extent(input, 1);
    const long long height = qcore_tensor_extent(input, 2);
    const long long width = qcore_tensor_extent(input, 3);
    const long long channels_out = qcore_tensor_extent(weight, 0);
    const long long weight_channels = qcore_tensor_extent(weight, 1);
    const long long kernel_height = qcore_tensor_extent(weight, 2);
    const long long kernel_width = qcore_tensor_extent(weight, 3);
    const long long output_height = qcore_tensor_extent(output, 2);
    const long long output_width = qcore_tensor_extent(output, 3);
    const long long limit = std::numeric_limits<std::int32_t>::max();
    const long long extents[] = {
        batches, channels_in, height, width, channels_out, weight_channels,
        kernel_height, kernel_width, output_height, output_width,
        stride, groups
    };
    for (const long long extent : extents) {
        if (extent <= 0 || extent > limit) return false;
    }
    if (padding > limit || height + 2 * padding > limit ||
        width + 2 * padding > limit ||
        channels_in % groups != 0 || channels_out % groups != 0 ||
        weight_channels != channels_in / groups ||
        qcore_tensor_extent(output, 0) != batches ||
        qcore_tensor_extent(output, 1) != channels_out) {
        return false;
    }
    const long long padded_height = height + 2 * padding;
    const long long padded_width = width + 2 * padding;
    if (padded_height < kernel_height || padded_width < kernel_width ||
        (padded_height - kernel_height) / stride + 1 != output_height ||
        (padded_width - kernel_width) / stride + 1 != output_width) {
        return false;
    }
    const auto input_count = qcore_tensor_element_count(input);
    const auto weight_count = qcore_tensor_element_count(weight);
    const auto output_count = qcore_tensor_element_count(output);
    if (!fits_u32(input_count) || !fits_u32(weight_count) ||
        !fits_u32(output_count) || input_count == 0 || output_count == 0) {
        return false;
    }
    shape = ConvShape{
        static_cast<std::uint32_t>(batches),
        static_cast<std::uint32_t>(channels_in),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(channels_out),
        static_cast<std::uint32_t>(weight_channels),
        static_cast<std::uint32_t>(kernel_height),
        static_cast<std::uint32_t>(kernel_width),
        static_cast<std::uint32_t>(output_height),
        static_cast<std::uint32_t>(output_width),
        static_cast<std::uint32_t>(stride),
        static_cast<std::uint32_t>(padding),
        static_cast<std::uint32_t>(groups),
        static_cast<std::uint32_t>(channels_out / groups),
        static_cast<std::uint32_t>(output_count),
        static_cast<std::uint32_t>(input_count)
    };
    return true;
}

// Kernel-size specialization: the k1/k3 pipelines follow the generic one in
// the kernel table and serve square 1x1/3x3 kernels; every other shape uses
// the generic pipeline.
std::size_t conv_variant(std::size_t generic, const ConvShape& shape) {
    if (shape.kernel_height != shape.kernel_width) return generic;
    if (shape.kernel_height == 1) return generic + 1;
    if (shape.kernel_height == 3) return generic + 2;
    return generic;
}

std::uint32_t channel_blocks(std::uint32_t channels) {
    constexpr std::uint32_t channel_block = 4;
    return (channels + channel_block - 1) / channel_block;
}

int encode_forward(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvShape& shape) {
    const std::uint32_t use_bias = bias ? 1u : 0u;
    if (!bind_const(encoder, input, 0) ||
        !bind_const(encoder, weight, 1) ||
        !bind_const(encoder, bias ? bias : weight, 2) ||
        !bind_mutable(encoder, output, 3))
        return 3;
    [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
    [encoder setBytes:&use_bias length:sizeof(use_bias) atIndex:5];
    // One thread per (n, group, block of output channels, oy, ox); the count
    // never exceeds the (32-bit) output element count.
    const std::uint32_t threads =
        shape.batches * shape.groups *
        channel_blocks(shape.outputs_per_group) *
        shape.output_height * shape.output_width;
    dispatch_elements(
        encoder,
        programs.pipelines[conv_variant(kernel_conv2d_forward, shape)],
        threads);
    return 0;
}

int encode_backward_data(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const void* weight,
    const void* gradient_output,
    void* gradient_input,
    const ConvShape& shape) {
    if (!bind_const(encoder, weight, 0) ||
        !bind_const(encoder, gradient_output, 1) ||
        !bind_mutable(encoder, gradient_input, 2))
        return 3;
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    // One thread per (n, group, block of input channels, iy, ix).
    const std::uint32_t threads =
        shape.batches * shape.groups *
        channel_blocks(shape.weight_channels) * shape.height * shape.width;
    dispatch_elements(
        encoder,
        programs.pipelines[conv_variant(kernel_conv2d_backward_data, shape)],
        threads);
    return 0;
}

int encode_backward_filter(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const void* input,
    const void* gradient_output,
    void* gradient_weight,
    const ConvShape& shape) {
    std::size_t variant = conv_variant(kernel_conv2d_backward_filter, shape);
    const auto elements = qcore_tensor_element_count(gradient_weight);
    if (!fits_u32(elements)) return 2;
    // The lane count depends only on the reduction length, so the k1/k3
    // pipelines (one threadgroup per channel pair) and the generic one (one
    // per weight element) produce identical per-tap sums; a specialized
    // pipeline that cannot run that many lanes yields to the generic one.
    const std::uint32_t width = reduction_width(
        programs.pipelines[kernel_conv2d_backward_filter],
        shape.batches * shape.output_height * shape.output_width);
    if (width > programs.pipelines[variant].maxTotalThreadsPerThreadgroup)
        variant = kernel_conv2d_backward_filter;
    id<MTLComputePipelineState> pipeline = programs.pipelines[variant];
    const std::uint32_t taps = shape.kernel_height * shape.kernel_width;
    const std::uint32_t groups =
        variant == kernel_conv2d_backward_filter
            ? static_cast<std::uint32_t>(elements)
            : static_cast<std::uint32_t>(elements) / taps;
    if (!bind_const(encoder, input, 0) ||
        !bind_const(encoder, gradient_output, 1) ||
        !bind_mutable(encoder, gradient_weight, 2))
        return 3;
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    [encoder setBytes:&width length:sizeof(width) atIndex:4];
    dispatch_groups(encoder, pipeline, groups, width);
    return 0;
}

int encode_backward_bias(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const void* gradient_output,
    void* gradient_bias,
    const ConvShape& shape) {
    id<MTLComputePipelineState> pipeline =
        programs.pipelines[kernel_conv2d_backward_bias];
    const std::uint32_t width = reduction_width(
        pipeline, shape.batches * shape.output_height * shape.output_width);
    if (!bind_const(encoder, gradient_output, 0) ||
        !bind_mutable(encoder, gradient_bias, 1))
        return 3;
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    dispatch_groups(encoder, pipeline, shape.channels_out, width);
    return 0;
}

bool same_extents(const void* left, const void* right) {
    const auto rank = qcore_tensor_rank(left);
    if (rank != qcore_tensor_rank(right)) return false;
    for (unsigned long long axis = 0; axis < rank; ++axis) {
        if (qcore_tensor_extent(left, axis) != qcore_tensor_extent(right, axis))
            return false;
    }
    return true;
}

bool bias_shape_ok(const void* bias, const void* reference, const ConvShape& shape) {
    return metal_f32(bias) && same_device(bias, reference) &&
           qcore_tensor_rank(bias) == 1 &&
           qcore_tensor_extent(bias, 0) ==
               static_cast<long long>(shape.channels_out);
}

} // namespace

// Device index of a float32 tensor on a Metal device (any layout), or -1.
// Kernels themselves still require dense tensors.
extern "C" long long nn_metal_device_f32(const void* tensor) {
    if (!tensor || qcore_tensor_dtype(tensor) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(tensor) != QCORE_BACKEND_METAL)
        return -1;
    const auto device = qcore_tensor_device(tensor);
    if (device < 0 ||
        device > static_cast<long long>(std::numeric_limits<std::int32_t>::max()))
        return -1;
    return device;
}

// output = conv2d(input, weight) (+ bias when bias is not null). Writes every
// output element; no autograd is attached here.
extern "C" int nn_metal_conv2d_forward(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride,
    long long padding,
    long long groups) {
    @autoreleasepool {
        ConvShape shape{};
        if (!conv_shape(input, weight, output, stride, padding, groups, shape))
            return 2;
        if (bias && !bias_shape_ok(bias, input, shape)) return 2;
        return submit(dispatch_conv2d_forward, input,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            return encode_forward(
                encoder, programs, input, weight, bias, output, shape);
        });
    }
}

// First-order Conv2D gradients in one command buffer: dInput, dWeight and
// dBias, each written by exactly one thread or threadgroup per element.
extern "C" int nn_metal_conv2d_backward(
    const void* input,
    const void* weight,
    const void* gradient_output,
    void* gradient_input,
    void* gradient_weight,
    void* gradient_bias,
    long long stride,
    long long padding,
    long long groups) {
    @autoreleasepool {
        ConvShape shape{};
        if (!conv_shape(input, weight, gradient_output, stride, padding,
                        groups, shape) ||
            !metal_f32(gradient_input) || !metal_f32(gradient_weight) ||
            !same_device(input, gradient_input) ||
            !same_device(input, gradient_weight) ||
            !same_extents(input, gradient_input) ||
            !same_extents(weight, gradient_weight) ||
            !bias_shape_ok(gradient_bias, input, shape))
            return 2;
        return submit(dispatch_conv2d_backward, input,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            int status = encode_backward_data(
                encoder, programs, weight, gradient_output, gradient_input,
                shape);
            if (status == 0)
                status = encode_backward_filter(
                    encoder, programs, input, gradient_output,
                    gradient_weight, shape);
            if (status == 0)
                status = encode_backward_bias(
                    encoder, programs, gradient_output, gradient_bias, shape);
            return status;
        });
    }
}

// gradient_input = conv2d_transpose(gradient_output, weight); the geometry is
// taken from gradient_input's NCHW shape.
extern "C" int nn_metal_conv2d_backward_data(
    const void* weight,
    const void* gradient_output,
    void* gradient_input,
    long long stride,
    long long padding,
    long long groups) {
    @autoreleasepool {
        ConvShape shape{};
        if (!conv_shape(gradient_input, weight, gradient_output, stride,
                        padding, groups, shape))
            return 2;
        return submit(dispatch_conv2d_backward_data, weight,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            return encode_backward_data(
                encoder, programs, weight, gradient_output, gradient_input,
                shape);
        });
    }
}

// gradient_weight = correlation of input with gradient_output; the geometry
// is taken from gradient_weight's OIHW shape.
extern "C" int nn_metal_conv2d_backward_filter(
    const void* input,
    const void* gradient_output,
    void* gradient_weight,
    long long stride,
    long long padding,
    long long groups) {
    @autoreleasepool {
        ConvShape shape{};
        if (!conv_shape(input, gradient_weight, gradient_output, stride,
                        padding, groups, shape))
            return 2;
        return submit(dispatch_conv2d_backward_filter, input,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            return encode_backward_filter(
                encoder, programs, input, gradient_output, gradient_weight,
                shape);
        });
    }
}

// output[n, c, y, x] = bias_gradient[c] for an NCHW output.
extern "C" int nn_metal_conv2d_bias_broadcast(
    const void* bias_gradient,
    void* output) {
    @autoreleasepool {
        if (!metal_f32(bias_gradient) || !metal_f32(output) ||
            !same_device(bias_gradient, output) ||
            qcore_tensor_rank(bias_gradient) != 1 ||
            qcore_tensor_rank(output) != 4 ||
            qcore_tensor_extent(bias_gradient, 0) !=
                qcore_tensor_extent(output, 1))
            return 2;
        const auto count = qcore_tensor_element_count(output);
        if (!fits_u32(count) || count == 0) return 2;
        ConvShape shape{};
        shape.channels_out =
            static_cast<std::uint32_t>(qcore_tensor_extent(output, 1));
        shape.output_height =
            static_cast<std::uint32_t>(qcore_tensor_extent(output, 2));
        shape.output_width =
            static_cast<std::uint32_t>(qcore_tensor_extent(output, 3));
        shape.output_count = static_cast<std::uint32_t>(count);
        return submit(dispatch_conv2d_bias_broadcast, output,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            if (!bind_const(encoder, bias_gradient, 0) ||
                !bind_mutable(encoder, output, 1))
                return 3;
            [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
            dispatch_elements(
                encoder, programs.pipelines[kernel_conv2d_bias_broadcast],
                shape.output_count);
            return 0;
        });
    }
}

namespace {

// Same-device dense float32 Metal tensors with one element count.
bool activation_tensors(
    std::initializer_list<const void*> tensors, std::uint32_t& count) {
    const void* first = *tensors.begin();
    if (!metal_f32(first)) return false;
    const auto elements = qcore_tensor_element_count(first);
    if (!fits_u32(elements)) return false;
    for (const void* tensor : tensors) {
        if (!metal_f32(tensor) || !same_device(first, tensor) ||
            qcore_tensor_element_count(tensor) != elements)
            return false;
    }
    count = static_cast<std::uint32_t>(elements);
    return true;
}

bool activation_code(long long activation, std::uint32_t& code) {
    if (activation != 1 && activation != 2) return false;
    code = static_cast<std::uint32_t>(activation);
    return true;
}

} // namespace

// output = activation(input) with NN's exact ReLU/GELU expressions.
extern "C" int nn_metal_activation_forward(
    const void* input,
    void* output,
    long long activation) {
    @autoreleasepool {
        std::uint32_t count = 0;
        std::uint32_t code = 0;
        if (!activation_code(activation, code) ||
            !activation_tensors({input, output}, count))
            return 2;
        if (count == 0) return 0;
        return submit(dispatch_activation_forward, input,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            if (!bind_const(encoder, input, 0) ||
                !bind_mutable(encoder, output, 1))
                return 3;
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
            [encoder setBytes:&code length:sizeof(code) atIndex:3];
            dispatch_elements(
                encoder, programs.pipelines[kernel_activation_forward], count);
            return 0;
        });
    }
}

namespace {

int activation_backward(
    const void* input,
    const void* gradient_output,
    void* gradient_input,
    long long activation,
    bool wait) {
    std::uint32_t count = 0;
    std::uint32_t code = 0;
    if (!activation_code(activation, code) ||
        !activation_tensors({input, gradient_output, gradient_input}, count))
        return 2;
    if (count == 0) return 0;
    const DispatchKind kind = wait ? dispatch_activation_backward
                                   : dispatch_activation_backward_internal;
    return submit(kind, input,
                  [&](id<MTLComputeCommandEncoder> encoder,
                      NnMetalPrograms& programs) {
        if (!bind_const(encoder, input, 0) ||
            !bind_const(encoder, gradient_output, 1) ||
            !bind_mutable(encoder, gradient_input, 2))
            return 3;
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
        [encoder setBytes:&code length:sizeof(code) atIndex:4];
        dispatch_elements(
            encoder, programs.pipelines[kernel_activation_backward], count);
        return 0;
    }, wait);
}

} // namespace

extern "C" int nn_metal_activation_backward(
    const void* input,
    const void* gradient_output,
    void* gradient_input,
    long long activation) {
    @autoreleasepool {
        return activation_backward(
            input, gradient_output, gradient_input, activation, true);
    }
}

// Activation backward whose gradient_input is read only by NN's Metal
// Conv2D backward (the activation of a Conv2D fusion target, whose
// pre-activation tensor never leaves NN). Its command buffer is committed
// without waiting: the Conv2D backward that consumes the gradient runs later
// on the same queue and thread, waits for completion itself and then checks
// this command's status too (settle_unchecked_commands).
extern "C" int nn_metal_activation_backward_internal(
    const void* input,
    const void* gradient_output,
    void* gradient_input,
    long long activation) {
    @autoreleasepool {
        return activation_backward(
            input, gradient_output, gradient_input, activation, false);
    }
}

extern "C" int nn_metal_activation_second_backward(
    const void* input,
    const void* first_gradient,
    const void* gradient_output,
    void* gradient_input,
    void* gradient_first,
    long long activation) {
    @autoreleasepool {
        std::uint32_t count = 0;
        std::uint32_t code = 0;
        if (!activation_code(activation, code) ||
            !activation_tensors({input, first_gradient, gradient_output,
                                 gradient_input, gradient_first}, count))
            return 2;
        if (count == 0) return 0;
        return submit(dispatch_activation_second_backward, input,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            if (!bind_const(encoder, input, 0) ||
                !bind_const(encoder, first_gradient, 1) ||
                !bind_const(encoder, gradient_output, 2) ||
                !bind_mutable(encoder, gradient_input, 3) ||
                !bind_mutable(encoder, gradient_first, 4))
                return 3;
            [encoder setBytes:&count length:sizeof(count) atIndex:5];
            [encoder setBytes:&code length:sizeof(code) atIndex:6];
            dispatch_elements(
                encoder, programs.pipelines[kernel_activation_second_backward],
                count);
            return 0;
        });
    }
}

namespace {

// [N, C, H, W] and [N, C] dense float32 tensors on one Metal device.
bool pooling_tensors(
    const void* planes,
    const void* rows,
    std::uint32_t& row_count,
    std::uint32_t& plane) {
    if (!metal_f32(planes) || !metal_f32(rows) || !same_device(planes, rows) ||
        qcore_tensor_rank(planes) != 4 || qcore_tensor_rank(rows) != 2 ||
        qcore_tensor_extent(planes, 0) != qcore_tensor_extent(rows, 0) ||
        qcore_tensor_extent(planes, 1) != qcore_tensor_extent(rows, 1))
        return false;
    const auto total = qcore_tensor_element_count(planes);
    const auto rows_total = qcore_tensor_element_count(rows);
    if (!fits_u32(total) || rows_total == 0 || total % rows_total != 0)
        return false;
    const auto plane_size = total / rows_total;
    if (plane_size == 0 || !fits_u32(plane_size)) return false;
    row_count = static_cast<std::uint32_t>(rows_total);
    plane = static_cast<std::uint32_t>(plane_size);
    return true;
}

} // namespace

// output[n, c] = mean of input[n, c, :, :].
extern "C" int nn_metal_global_average_pool(
    const void* input,
    void* output) {
    @autoreleasepool {
        std::uint32_t rows = 0;
        std::uint32_t plane = 0;
        if (!pooling_tensors(input, output, rows, plane)) return 2;
        return submit(dispatch_global_average_pool, input,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            id<MTLComputePipelineState> pipeline =
                programs.pipelines[kernel_global_average_pool];
            const std::uint32_t width = reduction_width(pipeline, plane);
            if (!bind_const(encoder, input, 0) ||
                !bind_mutable(encoder, output, 1))
                return 3;
            [encoder setBytes:&plane length:sizeof(plane) atIndex:2];
            [encoder setBytes:&width length:sizeof(width) atIndex:3];
            dispatch_groups(encoder, pipeline, rows, width);
            return 0;
        });
    }
}

// output[n, c, y, x] = gradient[n, c] / (H * W).
extern "C" int nn_metal_global_average_unpool(
    const void* gradient,
    void* output) {
    @autoreleasepool {
        std::uint32_t rows = 0;
        std::uint32_t plane = 0;
        if (!pooling_tensors(output, gradient, rows, plane)) return 2;
        const std::uint32_t count = rows * plane;
        return submit(dispatch_global_average_unpool, gradient,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            if (!bind_const(encoder, gradient, 0) ||
                !bind_mutable(encoder, output, 1))
                return 3;
            [encoder setBytes:&plane length:sizeof(plane) atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            dispatch_elements(
                encoder, programs.pipelines[kernel_global_average_unpool],
                count);
            return 0;
        });
    }
}

namespace {

// Adam updates of one optimizer step are encoded into one command buffer and
// committed together, so a step costs one GPU round trip instead of one per
// Parameter. Every batch belongs to the Adam.step call that began it: the
// token returned by nn_metal_adam_begin names it in encode and commit, so
// concurrent optimizers (for example under task.all) never share, reset or
// commit each other's work.
//
// Lifetime: an encode binds Core's buffers to the batch's command buffer,
// which keeps strong references to them (Metal's default retained
// references); NN never reads the borrowed handles after the encode call
// returns. The Quidra caller additionally keeps every bound tensor alive and
// untouched until the commit returns, and the outputs are fresh tensors that
// nothing reads before then. The inputs were written by commands Core
// committed earlier on the same queue, so queue order makes them visible.
struct AdamBatch {
    id<MTLCommandBuffer> command = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    long long device = -1;
    std::uint64_t dispatches = 0;
};

std::mutex adam_mutex;
std::map<std::int32_t, AdamBatch> adam_batches;
std::int32_t adam_next_token = 1;
// A caller that never commits (it cannot happen in NN's own Adam.step) leaks
// its batch; the cap bounds that.
constexpr std::size_t adam_batch_limit = 4096;

void adam_batch_release(AdamBatch& batch) {
    if (batch.encoder) {
        [batch.encoder endEncoding];
        [batch.encoder release];
    }
    if (batch.command) [batch.command release];
    batch = AdamBatch{};
}

} // namespace

// Starts a new, empty Adam batch owned by the caller and returns its token
// (> 0), or 0 when no batch can be started.
extern "C" std::int32_t nn_metal_adam_begin() {
    std::lock_guard<std::mutex> lock(adam_mutex);
    if (adam_batches.size() >= adam_batch_limit) return 0;
    for (;;) {
        const std::int32_t token = adam_next_token;
        adam_next_token = token == std::numeric_limits<std::int32_t>::max()
                              ? 1
                              : token + 1;
        if (adam_batches.try_emplace(token).second) return token;
    }
}

// Encodes one Adam update for dense float32 Metal tensors of one shape
// (next parameter and both next moments) into batch `batch`. Nothing runs
// until commit. A refused encode (non-zero) adds nothing to the batch.
extern "C" int nn_metal_adam_encode(
    std::int32_t batch,
    const void* parameter,
    const void* gradient,
    const void* first,
    const void* second,
    void* next_parameter,
    void* next_first,
    void* next_second,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float epsilon,
    float scale) {
    @autoreleasepool {
        std::uint32_t count = 0;
        if (!activation_tensors({parameter, gradient, first, second,
                                 next_parameter, next_first, next_second},
                                count))
            return 2;
        const void* others[] = {gradient, first, second, next_parameter,
                                next_first, next_second};
        for (const void* tensor : others) {
            if (!same_extents(parameter, tensor)) return 2;
        }
        std::lock_guard<std::mutex> lock(adam_mutex);
        const auto found = adam_batches.find(batch);
        if (found == adam_batches.end()) return 7;
        AdamBatch& owned = found->second;
        const long long device = qcore_tensor_device(parameter);
        if (owned.command && owned.device != device) return 9;
        if (!owned.command) {
            id<MTLCommandQueue> queue = command_queue(parameter);
            if (!queue) return 3;
            if (!programs_for([queue device])) return 5;
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!command) return 5;
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (!encoder) return 5;
            owned.command = [command retain];
            owned.encoder = [encoder retain];
            owned.device = device;
        }
        NnMetalPrograms* programs = programs_for([owned.command device]);
        if (!programs) return 5;
        struct {
            float beta1;
            float beta2;
            float one_minus_beta1;
            float one_minus_beta2;
            float epsilon;
            float scale;
            std::uint32_t count;
        } coefficients{beta1, beta2, one_minus_beta1, one_minus_beta2,
                       epsilon, scale, count};
        id<MTLComputeCommandEncoder> encoder = owned.encoder;
        // Every dispatch rebinds all of its slots, so a bind that fails part
        // way leaves nothing behind: the dispatch is simply not issued and
        // the updates already in the batch are unaffected.
        if (!bind_const(encoder, parameter, 0) ||
            !bind_const(encoder, gradient, 1) ||
            !bind_const(encoder, first, 2) ||
            !bind_const(encoder, second, 3) ||
            !bind_mutable(encoder, next_parameter, 4) ||
            !bind_mutable(encoder, next_first, 5) ||
            !bind_mutable(encoder, next_second, 6))
            return 3;
        [encoder setBytes:&coefficients length:sizeof(coefficients) atIndex:7];
        dispatch_elements(encoder, programs->pipelines[kernel_adam_step], count);
        ++owned.dispatches;
        return 0;
    }
}

// Commits batch `batch` and waits for it (see submit() for why NN waits),
// then releases the token. The batch must hold exactly `expected` encoded
// updates; otherwise, or for an unknown token, nothing is committed and the
// caller must not use any output of the batch. `expected == 0` with no
// encoded update just releases the batch.
extern "C" int nn_metal_adam_commit(
    std::int32_t batch,
    std::uint64_t expected) {
    @autoreleasepool {
        AdamBatch owned;
        {
            std::lock_guard<std::mutex> lock(adam_mutex);
            const auto found = adam_batches.find(batch);
            if (found == adam_batches.end()) return 7;
            owned = found->second;
            adam_batches.erase(found);
        }
        if (owned.dispatches != expected) {
            adam_batch_release(owned);
            return 8;
        }
        if (!owned.command) return 0;
        [owned.encoder endEncoding];
        [owned.encoder release];
        owned.encoder = nil;
        id<MTLCommandBuffer> command = owned.command;
        owned.command = nil;
        [command commit];
        [command waitUntilCompleted];
        const bool earlier = settle_unchecked_commands();
        const bool completed =
            [command status] == MTLCommandBufferStatusCompleted;
        [command release];
        if (!earlier || !completed) return 6;
        count_dispatch(dispatch_adam_update, owned.dispatches);
        return 0;
    }
}

// Number of successful NN Metal dispatches of one DispatchKind in this
// process (0 for an unknown kind). Test probe only; see DispatchKind.
extern "C" unsigned long long nn_metal_dispatch_count(int kind) {
    if (kind <= 0 || kind >= dispatch_kinds) return 0;
    return dispatch_counts[kind].load(std::memory_order_relaxed);
}

// Conv2D followed by an NN activation in one command buffer (one GPU round
// trip): conv_output = conv2d(input, weight) + bias, then
// activation_output = activation(conv_output). Both results are written.
extern "C" int nn_metal_conv2d_activation_forward(
    const void* input,
    const void* weight,
    const void* bias,
    void* conv_output,
    void* activation_output,
    long long stride,
    long long padding,
    long long groups,
    long long activation) {
    @autoreleasepool {
        ConvShape shape{};
        std::uint32_t count = 0;
        std::uint32_t code = 0;
        if (!conv_shape(input, weight, conv_output, stride, padding, groups,
                        shape) ||
            !bias_shape_ok(bias, input, shape) ||
            !activation_code(activation, code) ||
            !activation_tensors({conv_output, activation_output}, count) ||
            !same_extents(conv_output, activation_output))
            return 2;
        return submit(dispatch_conv2d_activation_forward, input,
                      [&](id<MTLComputeCommandEncoder> encoder,
                          NnMetalPrograms& programs) {
            const int status = encode_forward(
                encoder, programs, input, weight, bias, conv_output, shape);
            if (status != 0) return status;
            // The activation reads what the convolution wrote.
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (!bind_const(encoder, conv_output, 0) ||
                !bind_mutable(encoder, activation_output, 1))
                return 3;
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
            [encoder setBytes:&code length:sizeof(code) atIndex:3];
            dispatch_elements(
                encoder, programs.pipelines[kernel_activation_forward], count);
            return 0;
        });
    }
}
