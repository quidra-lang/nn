// NN-owned Metal kernels. Core lends its command stream and tensor buffers
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
// element-wise update uses the same separately rounded operations as Core's
// and Math's Metal kernels. This library disables fast math and uses ordinary
// square root and division; tests/metal_adam.qui checks their bit identity.
struct AdamCoefficients {
    float beta1;
    float beta2;
    float one_minus_beta1;
    float one_minus_beta2;
    float epsilon;
    float scale;
    uint count;
};

// The current Parameter and moments are only read; the next ones are
// written to buffers of their own.
kernel void nn_adam_update(
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
    const float denominator = sqrt(v) + c.epsilon;
    next_first[gid] = m;
    next_second[gid] = v;
    next_parameter[gid] =
        parameter[gid] - (m / denominator) * c.scale;
}

struct SgdCoefficients { float rate; uint count; };
kernel void nn_sgd_update(device const float* parameter [[buffer(0)]],
                           device const float* gradient [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           constant SgdCoefficients& c [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= c.count) return;
    const float product = gradient[gid] * c.rate;
    output[gid] = parameter[gid] - product;
}

ulong nn_qjam(ulong m,int n){if(n<=0)return m;if(n>=64)return ulong(m!=0);return (m>>n)|ulong((m&((1ul<<n)-1ul))!=0);}
ulong nn_qnearest(ulong m,int n){if(n<=0)return m<<(-n);if(n>64)return 0;if(n==64)return ulong(m>0x8000000000000000ul);ulong v=m>>n,r=m&((1ul<<n)-1ul),h=1ul<<(n-1);return v+ulong(r>h||(r==h&&(v&1ul)));}
uint nn_qpack(ulong m,int e,uint sign){
 if(m==0)return sign;
 int top=0;for(ulong v=m;v>1;v>>=1)++top;
 int exponent=top+e;
 if(exponent>127)return sign|0x7f800000u;
 if(exponent< -126){ulong v=nn_qnearest(m,-149-e);return sign|uint(v);}
 ulong v=nn_qnearest(m,top-23);if(v>=0x1000000ul){v>>=1;++exponent;}
 if(exponent>127)return sign|0x7f800000u;
 return sign|(uint(exponent+127)<<23)|(uint(v)&0x7fffffu);
}
ulong nn_qmant(uint b){return ulong(b&0x7fffffu)|((b&0x7f800000u)?0x800000ul:0ul);}
int nn_qexp(uint b){int e=int((b>>23)&255u);return e?e-150:-149;}
uint nn_qadd_bits(uint a,uint b){
 uint aa=a&0x7fffffffu,bb=b&0x7fffffffu,sa=a&0x80000000u,sb=b&0x80000000u;
 if(aa>=0x7f800000u||bb>=0x7f800000u){if(aa>0x7f800000u||bb>0x7f800000u||(aa==bb&&sa!=sb))return 0x7fc00000u;return aa==0x7f800000u?a:b;}
 if(aa==0&&bb==0)return sa&sb;
 if(aa==bb&&sa!=sb)return 0u;
 if(aa<bb){uint t=a;a=b;b=t;t=aa;aa=bb;bb=t;t=sa;sa=sb;sb=t;}
 int e=nn_qexp(a);ulong x=nn_qmant(a)<<3,y=nn_qjam(nn_qmant(b)<<3,e-nn_qexp(b));
 return nn_qpack(sa==sb?x+y:x-y,e-3,sa);
}
uint nn_qmul_bits(uint a,uint b){
 uint aa=a&0x7fffffffu,bb=b&0x7fffffffu,s=(a^b)&0x80000000u;
 if(aa>=0x7f800000u||bb>=0x7f800000u){if(aa>0x7f800000u||bb>0x7f800000u||aa==0||bb==0)return 0x7fc00000u;return s|0x7f800000u;}
 return nn_qpack(nn_qmant(a)*nn_qmant(b),nn_qexp(a)+nn_qexp(b),s);
}
uint nn_qdiv_bits(uint a,uint b){
 uint aa=a&0x7fffffffu,bb=b&0x7fffffffu,s=(a^b)&0x80000000u;
 if(aa>0x7f800000u||bb>0x7f800000u||(aa==0&&bb==0)||(aa==0x7f800000u&&bb==0x7f800000u))return 0x7fc00000u;
 if(aa==0x7f800000u||bb==0)return s|0x7f800000u;
 if(aa==0||bb==0x7f800000u)return s;
 ulong am=nn_qmant(a),bm=nn_qmant(b);int ae=nn_qexp(a),be=nn_qexp(b);
 while(am<0x800000ul){am<<=1;--ae;}while(bm<0x800000ul){bm<<=1;--be;}
 ulong numerator=am<<32,denominator=bm,v=numerator/denominator;
 v|=ulong(numerator%denominator!=0);
 return nn_qpack(v,ae-be-32,s);
}

uint nn_sqrt_bits(uint raw) {
    if ((raw & 0x7fffffffu) == 0) return raw;
    if ((raw & 0x80000000u) || raw > 0x7f800000u) return 0x7fc00000u;
    if (raw == 0x7f800000u) return raw;
    ulong m = nn_qmant(raw); int e = nn_qexp(raw);
    while (m < 0x800000ul) { m <<= 1; --e; }
    if (e % 2 != 0) { m <<= 1; --e; }
    ulong remainder = m << 32, root = 0, bit = 1ul << 62;
    while (bit > remainder) bit >>= 2;
    while (bit) {
        if (remainder >= root + bit) {
            remainder -= root + bit; root = (root >> 1) + bit;
        } else root >>= 1;
        bit >>= 2;
    }
    return nn_qpack(root | ulong(remainder != 0), e / 2 - 16, 0);
}
uint nn_half_widen(ushort raw, uint bfloat_mode) {
    if (bfloat_mode) return uint(raw) << 16;
    uint sign = uint(raw & 0x8000u) << 16, exponent = (raw >> 10) & 31u;
    uint fraction = raw & 1023u;
    if (exponent == 31) return sign | 0x7f800000u | (fraction << 13);
    if (exponent) return sign | ((exponent + 112u) << 23) | (fraction << 13);
    if (!fraction) return sign;
    int e = -14;
    while (fraction < 1024u) { fraction <<= 1; --e; }
    return sign | (uint(e + 127) << 23) | ((fraction & 1023u) << 13);
}
bool nn_half_narrow(uint raw, uint bfloat_mode, thread ushort& result) {
    if ((raw & 0x7f800000u) == 0x7f800000u) return false;
    if (bfloat_mode) {
        result = ushort((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
        return (result & 0x7f80u) != 0x7f80u;
    }
    ushort sign = ushort((raw >> 16) & 0x8000u);
    int e = int((raw >> 23) & 255u) - 127;
    uint mantissa = (raw & 0x7fffffu) | 0x800000u;
    if (e < -25) { result = sign; return true; }
    if (e < -14) { result = sign | ushort(nn_qnearest(mantissa, -e - 1)); return true; }
    ulong rounded = nn_qnearest(mantissa, 13); int exponent = e + 15;
    if (rounded >= 2048) { rounded >>= 1; ++exponent; }
    if (exponent >= 31) return false;
    result = sign | ushort((exponent << 10) | (rounded & 1023ul));
    return true;
}
bool nn_half_native_safe(uint raw) {
    uint magnitude = raw & 0x7fffffffu;
    return magnitude == 0 || (magnitude >= 0x0d800000u && magnitude < 0x7f800000u);
}
float nn_half_add(float a, float b, thread bool& safe) {
    float r = a + b;
    safe = safe && nn_half_native_safe(as_type<uint>(a)) && nn_half_native_safe(as_type<uint>(b)) && nn_half_native_safe(as_type<uint>(r));
    return r;
}
float nn_half_mul(float a, float b, thread bool& safe) {
    float r = a * b;
    safe = safe && nn_half_native_safe(as_type<uint>(a)) && nn_half_native_safe(as_type<uint>(b)) && nn_half_native_safe(as_type<uint>(r));
    if ((as_type<uint>(a) & 0x7fffffffu) && (as_type<uint>(b) & 0x7fffffffu) && !(as_type<uint>(r) & 0x7fffffffu)) safe = false;
    return r;
}
float nn_half_div(float a, float b, thread bool& safe) {
    float r = a / b;
    safe = safe && nn_half_native_safe(as_type<uint>(a)) && nn_half_native_safe(as_type<uint>(b)) && nn_half_native_safe(as_type<uint>(r));
    if ((as_type<uint>(a) & 0x7fffffffu) && !(as_type<uint>(r) & 0x7fffffffu)) safe = false;
    return r;
}
struct HalfCoefficients {
    float beta1, beta2, one_minus_beta1, one_minus_beta2, epsilon, scale;
    uint count, bfloat_mode, adam, count_exact;
};
kernel void nn_half_update(
    device const ushort* parameter [[buffer(0)]],
    device const ushort* gradient [[buffer(1)]],
    device uint* first [[buffer(2)]], device uint* second [[buffer(3)]],
    device ushort* output [[buffer(4)]], constant HalfCoefficients& c [[buffer(5)]],
    device atomic_uint* conversion [[buffer(6)]], device atomic_uint* nonfinite [[buffer(7)]],
    device atomic_uint* exact_count [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= c.count) return;
    const uint pb = nn_half_widen(parameter[gid], c.bfloat_mode);
    const uint gb = nn_half_widen(gradient[gid], c.bfloat_mode);
    const uint mb = c.adam ? first[gid] : 0u, vb = c.adam ? second[gid] : 0u;
    bool safe = nn_half_native_safe(pb) && nn_half_native_safe(gb);
    const float p = as_type<float>(pb), g = as_type<float>(gb);
    uint nm = 0, nv = 0, updated;
    if (c.adam) {
        float m0 = nn_half_mul(as_type<float>(mb), c.beta1, safe);
        float m1 = nn_half_mul(g, c.one_minus_beta1, safe);
        float m = nn_half_add(m0, m1, safe);
        float v0 = nn_half_mul(as_type<float>(vb), c.beta2, safe);
        float g2 = nn_half_mul(g, g, safe);
        float v1 = nn_half_mul(g2, c.one_minus_beta2, safe);
        float v = nn_half_add(v0, v1, safe);
        float root = sqrt(v);
        safe = safe && nn_half_native_safe(as_type<uint>(root));
        float denominator = nn_half_add(root, c.epsilon, safe);
        float quotient = nn_half_div(m, denominator, safe);
        float product = nn_half_mul(quotient, c.scale, safe);
        updated = as_type<uint>(nn_half_add(p, -product, safe));
        nm = as_type<uint>(m); nv = as_type<uint>(v);
    } else {
        float product = nn_half_mul(g, c.scale, safe);
        updated = as_type<uint>(nn_half_add(p, -product, safe));
    }
    if (!safe) {
        if (c.count_exact) atomic_fetch_add_explicit(exact_count, 1u, memory_order_relaxed);
        if (c.adam) {
            nm = nn_qadd_bits(nn_qmul_bits(mb, as_type<uint>(c.beta1)), nn_qmul_bits(gb, as_type<uint>(c.one_minus_beta1)));
            nv = nn_qadd_bits(nn_qmul_bits(vb, as_type<uint>(c.beta2)), nn_qmul_bits(nn_qmul_bits(gb, gb), as_type<uint>(c.one_minus_beta2)));
            uint denominator = nn_qadd_bits(nn_sqrt_bits(nv), as_type<uint>(c.epsilon));
            uint product = nn_qmul_bits(nn_qdiv_bits(nm, denominator), as_type<uint>(c.scale));
            updated = nn_qadd_bits(pb, product ^ 0x80000000u);
        } else {
            updated = nn_qadd_bits(pb, nn_qmul_bits(gb, as_type<uint>(c.scale)) ^ 0x80000000u);
        }
    }
    if (c.adam) {
        first[gid] = nm; second[gid] = nv;
        if ((nm & 0x7fffffffu) > 0x7f800000u || (nv & 0x7fffffffu) > 0x7f800000u)
            atomic_store_explicit(nonfinite, 1u, memory_order_relaxed);
    }
    ushort narrowed = 0;
    if (!nn_half_narrow(updated, c.bfloat_mode, narrowed)) {
        atomic_store_explicit(conversion, 1u, memory_order_relaxed);
    } else output[gid] = narrowed;
}


// Raw-bit fixture entry point. Integer outputs deliberately retain IEEE
// exceptional encodings without making them observable as real values.
kernel void nn_half_probe_words(device const uint* left [[buffer(0)]],
    device const uint* right [[buffer(1)]], device uint* output [[buffer(2)]],
    constant uint4& options [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= options.x) return;
    const uint a = left[gid], b = right[gid], operation = options.y;
    if (operation == 1) { output[gid] = nn_half_widen(ushort(a), options.z); return; }
    if (operation == 2) {
        ushort value = 0;
        output[gid] = nn_half_narrow(a, options.z, value) ? uint(value) : 0x10000u;
        return;
    }
    bool safe = nn_half_native_safe(a) && nn_half_native_safe(b);
    float result;
    if (operation == 3) result = nn_half_add(as_type<float>(a), as_type<float>(b), safe);
    else if (operation == 4) result = nn_half_mul(as_type<float>(a), as_type<float>(b), safe);
    else if (operation == 5) result = nn_half_div(as_type<float>(a), as_type<float>(b), safe);
    else { result = sqrt(as_type<float>(a)); safe = safe && nn_half_native_safe(as_type<uint>(result)); }
    if (safe) output[gid] = as_type<uint>(result);
    else if (operation == 3) output[gid] = nn_qadd_bits(a, b);
    else if (operation == 4) output[gid] = nn_qmul_bits(a, b);
    else if (operation == 5) output[gid] = nn_qdiv_bits(a, b);
    else output[gid] = nn_sqrt_bits(a);
}

// Only public real32 outputs are checked; internal masks are not inputs.
kernel void nn_validate_finite(device const uint* values [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]], constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count && (values[gid] & 0x7fffffffu) > 0x7f800000u)
        atomic_store_explicit(status, 1u, memory_order_relaxed);
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
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
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
    kernel_adam_update,
    kernel_sgd_update,
    kernel_half_update,
    kernel_validate_finite,
    kernel_half_probe_words,
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
    "nn_adam_update",
    "nn_sgd_update",
    "nn_half_update",
    "nn_validate_finite",
    "nn_half_probe_words",
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

template <typename T>
T bridge_handle(std::uint64_t handle) {
    return (__bridge T)(reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(handle)));
}

// The device's MTLDevice, for compiling NN's pipelines. A lookup that lends
// no queue, so it is allowed anywhere, also while Core's stream is held.
id<MTLDevice> native_device(long long device) {
    const auto handle = qcore_device_native_device(device);
    return handle == 0 ? nil : bridge_handle<id<MTLDevice>>(handle);
}

// A tensor buffer lent by Core, taken before the encoder scope opens: taking
// it may make Core work for it (a copy-on-write detach, an upload, storage of
// its own), which Core refuses inside the scope.
struct Borrowed {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;

    explicit operator bool() const {
        return buffer != nil;
    }
};

Borrowed borrowed(const void* tensor, std::uint64_t handle) {
    if (handle == 0) return {};
    return {bridge_handle<id<MTLBuffer>>(handle),
            static_cast<NSUInteger>(qcore_tensor_device_offset_bytes(tensor))};
}

Borrowed borrow_const(const void* tensor) {
    return borrowed(tensor, qcore_tensor_device_handle_const(tensor));
}

Borrowed borrow_mutable(void* tensor) {
    return borrowed(tensor, qcore_tensor_device_handle(tensor));
}

// A write-only output: the kernel stores every element, so Core lends
// uninitialized storage with no fill (qcore_tensor_output_handle), and the
// output is marked written once the kernel is encoded.
Borrowed borrow_output(void* tensor) {
    return borrowed(tensor, qcore_tensor_output_handle(tensor));
}

// Forward kernels also serve as adjoints inside backward callbacks, where
// they write a gradient Core lends; only a forward output is write-only.
Borrowed borrow_written(void* tensor, bool write_only) {
    return write_only ? borrow_output(tensor) : borrow_mutable(tensor);
}

bool mark_written(std::initializer_list<void*> tensors) {
    bool written = true;
    for (void* tensor : tensors)
        written = qcore_tensor_mark_written(tensor) == 0 && written;
    return written;
}

void bind(id<MTLComputeCommandEncoder> encoder, const Borrowed& value,
          NSUInteger index) {
    [encoder setBuffer:value.buffer offset:value.offset atIndex:index];
}

// One NN submission on a Metal device.
//
// On Core's command stream it follows the encode-scope protocol of
// quidra/native_extension.h: the constructor holds the device's stream, the
// caller then takes every tensor handle, opens the scope (compute_encoder()),
// sets state and dispatches, and close() ends scope and hold with one
// qcore_metal_note_work. The work joins Core's open command buffer, which
// Core commits with its own work and waits for only where the host reads, so
// NN never commits or waits. Core's encoder dispatches serially, so a
// dispatch reads what the dispatches before it wrote.
//
// Without a Core stream (QUIDRA_METAL_STREAM=0) the same calls fill a command
// buffer of NN's own on Core's queue, and close() commits it without waiting;
// Core's synchronization covers command buffers committed on its queue.
//
// A submission that is not closed (an early return) ends the hold with no
// work, and an own command buffer is dropped uncommitted.
class MetalSubmission {
public:
    explicit MetalSubmission(long long device)
        : device_(device), held_(qcore_device_encode_begin(device) != 0) {}

    MetalSubmission(const MetalSubmission&) = delete;
    MetalSubmission& operator=(const MetalSubmission&) = delete;

    ~MetalSubmission() {
        if (closed_) return;
        if (held_) {
            qcore_metal_note_work(device_, 0, 0);
        } else if (own_encoder_) {
            [encoder_ endEncoding];
        }
    }

    // Opens the compute-encoder scope. On Core's stream this is Core's open
    // encoder (serial dispatch, tracked resources), which NN never ends.
    id<MTLComputeCommandEncoder> compute_encoder() {
        if (encoder_) return encoder_;
        if (held_) {
            const auto handle = qcore_device_compute_encoder(device_);
            if (handle != 0)
                encoder_ = bridge_handle<id<MTLComputeCommandEncoder>>(handle);
            return encoder_;
        }
        if (!command_) {
            const auto handle = qcore_device_queue_handle(device_);
            if (handle == 0) return nil;
            command_ =
                [bridge_handle<id<MTLCommandQueue>>(handle) commandBuffer];
            if (!command_) return nil;
        }
        encoder_ = [command_ computeCommandEncoder];
        own_encoder_ = encoder_ != nil;
        return encoder_;
    }

    // Ends the submission after a successful encode. `dispatches` and
    // `bytes` feed Core's commit policy.
    int close(std::uint32_t dispatches, std::uint64_t bytes) {
        if (closed_) return 5;
        closed_ = true;
        if (held_) {
            qcore_metal_note_work(device_, dispatches, bytes);
            return 0;
        }
        if (own_encoder_) [encoder_ endEncoding];
        if (command_) [command_ commit];
        return 0;
    }

private:
    long long device_;
    bool held_;
    bool closed_ = false;
    bool own_encoder_ = false;
    id<MTLCommandBuffer> command_ = nil;
    id<MTLComputeCommandEncoder> encoder_ = nil;
};

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

std::uint64_t float_bytes(std::uint64_t count) {
    return count * sizeof(float);
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
    // Gradient kernels encoded by first-order Conv2D backward calls (one per
    // requested gradient), so tests can see the masked request.
    dispatch_conv2d_backward_gradients = 14,
    dispatch_sgd_update = 15,
    dispatch_half_update = 16,
    dispatch_validate_finite = 17,
    dispatch_kinds = 18,
};

std::atomic<std::uint64_t> dispatch_counts[dispatch_kinds];

void count_dispatch(DispatchKind kind, std::uint64_t count = 1) {
    dispatch_counts[kind].fetch_add(count, std::memory_order_relaxed);
}

// Encodes one NN call on the device of `tensor`: compiles NN's pipelines for
// the device before the stream is held, then holds it, lets `take` borrow
// every handle the call binds (returning false refuses the call before any
// scope opens), opens the encoder scope, lets `encode` set state and
// dispatch, closes, and marks the write-only outputs in `written` (taken
// with borrow_output) as written. Nothing is committed or waited for here:
// Core runs the work in program order with its own and synchronizes where
// the host reads. An encoded call counts as one call of `kind`.
template <typename Take, typename Encode>
int encode_call(
    DispatchKind kind,
    const void* tensor,
    std::uint32_t dispatches,
    std::uint64_t bytes,
    Take&& take,
    Encode&& encode,
    std::initializer_list<void*> written = {}) {
    const auto device = qcore_tensor_device(tensor);
    if (device < 0) return 3;
    NnMetalPrograms* programs = programs_for(native_device(device));
    if (!programs) return 5;
    MetalSubmission submission(device);
    if (!take()) return 3;
    id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
    if (!encoder) return 5;
    const int status = encode(encoder, *programs);
    if (status != 0) return status;
    const int closed = submission.close(dispatches, bytes);
    if (closed != 0) return closed;
    if (!mark_written(written)) return 5;
    count_dispatch(kind);
    return 0;
}


// Defer owns a slot only on success. After an encoded refusal, wait consumes
// the caller-owned slot before returning failure; it is never released twice.
class NnDeviceStatus {
public:
    explicit NnDeviceStatus(long long device) : device_(device) {}
    NnDeviceStatus(const NnDeviceStatus&) = delete;
    NnDeviceStatus& operator=(const NnDeviceStatus&) = delete;
    ~NnDeviceStatus() { if (taken_) qcore_device_status_release(device_, slot_); }
    bool take() {
        std::uint64_t handle = 0;
        if (qcore_device_status_slot(device_, &handle, &offset_, &slot_) != 0) return false;
        taken_ = true; buffer_ = bridge_handle<id<MTLBuffer>>(handle);
        return buffer_ != nil;
    }
    void bind_to(id<MTLComputeCommandEncoder> encoder, NSUInteger index) const {
        [encoder setBuffer:buffer_ offset:static_cast<NSUInteger>(offset_) atIndex:index];
    }
    bool defer(int dtype, const char* message) {
        const int result = dtype ? qcore_device_defer_conversion_status(device_, slot_, dtype)
                                 : qcore_device_defer_status(device_, slot_, message);
        if (result == 0) { taken_ = false; return true; }
        waited_ = qcore_device_status_wait(device_, slot_, &value_) == 0;
        taken_ = false;
        return false;
    }
    bool failed_check() const { return waited_ && value_ != 0; }
private:
    bool waited_ = false;
    std::uint32_t value_ = 0;
    long long device_; bool taken_ = false;
    std::uint64_t slot_ = 0, offset_ = 0;
    id<MTLBuffer> buffer_ = nil;
};

// This storage exists only while a test explicitly enables the probe. Reading
// or resetting it waits; normal half updates never synchronize for a counter.
std::mutex half_probe_mutex;
std::map<long long, void*> half_probe_buffers;
Borrowed half_probe_buffer(long long device) {
    std::lock_guard<std::mutex> lock(half_probe_mutex);
    const auto it = half_probe_buffers.find(device);
    if (it == half_probe_buffers.end()) return {};
    return {bridge_handle<id<MTLBuffer>>(qcore_device_buffer_handle(it->second)), 0};
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

std::uint64_t conv_bytes(const ConvShape& shape) {
    return float_bytes(static_cast<std::uint64_t>(shape.input_count) +
                       shape.output_count);
}

void encode_forward(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const Borrowed& input,
    const Borrowed& weight,
    const Borrowed& bias,
    const Borrowed& output,
    bool use_bias,
    const ConvShape& shape) {
    const std::uint32_t bias_flag = use_bias ? 1u : 0u;
    bind(encoder, input, 0);
    bind(encoder, weight, 1);
    bind(encoder, use_bias ? bias : weight, 2);
    bind(encoder, output, 3);
    [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
    [encoder setBytes:&bias_flag length:sizeof(bias_flag) atIndex:5];
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
}

void encode_backward_data(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const Borrowed& weight,
    const Borrowed& gradient_output,
    const Borrowed& gradient_input,
    const ConvShape& shape) {
    bind(encoder, weight, 0);
    bind(encoder, gradient_output, 1);
    bind(encoder, gradient_input, 2);
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    // One thread per (n, group, block of input channels, iy, ix).
    const std::uint32_t threads =
        shape.batches * shape.groups *
        channel_blocks(shape.weight_channels) * shape.height * shape.width;
    dispatch_elements(
        encoder,
        programs.pipelines[conv_variant(kernel_conv2d_backward_data, shape)],
        threads);
}

// The dWeight dispatch: its pipeline, threadgroup count and lane width. The
// lane count depends only on the reduction length, so the k1/k3 pipelines
// (one threadgroup per channel pair) and the generic one (one per weight
// element) produce identical per-tap sums; a specialized pipeline that
// cannot run that many lanes yields to the generic one.
struct FilterPlan {
    id<MTLComputePipelineState> pipeline = nil;
    std::uint32_t groups = 0;
    std::uint32_t width = 0;
};

FilterPlan filter_plan(NnMetalPrograms& programs, const ConvShape& shape) {
    std::size_t variant = conv_variant(kernel_conv2d_backward_filter, shape);
    const std::uint32_t elements =
        shape.channels_out * shape.weight_channels *
        shape.kernel_height * shape.kernel_width;
    FilterPlan plan;
    plan.width = reduction_width(
        programs.pipelines[kernel_conv2d_backward_filter],
        shape.batches * shape.output_height * shape.output_width);
    if (plan.width > programs.pipelines[variant].maxTotalThreadsPerThreadgroup)
        variant = kernel_conv2d_backward_filter;
    plan.pipeline = programs.pipelines[variant];
    const std::uint32_t taps = shape.kernel_height * shape.kernel_width;
    plan.groups = variant == kernel_conv2d_backward_filter
                      ? elements
                      : elements / taps;
    return plan;
}

void encode_backward_filter(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const Borrowed& input,
    const Borrowed& gradient_output,
    const Borrowed& gradient_weight,
    const ConvShape& shape) {
    const FilterPlan plan = filter_plan(programs, shape);
    bind(encoder, input, 0);
    bind(encoder, gradient_output, 1);
    bind(encoder, gradient_weight, 2);
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    [encoder setBytes:&plan.width length:sizeof(plan.width) atIndex:4];
    dispatch_groups(encoder, plan.pipeline, plan.groups, plan.width);
}

void encode_backward_bias(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const Borrowed& gradient_output,
    const Borrowed& gradient_bias,
    const ConvShape& shape) {
    id<MTLComputePipelineState> pipeline =
        programs.pipelines[kernel_conv2d_backward_bias];
    const std::uint32_t width = reduction_width(
        pipeline, shape.batches * shape.output_height * shape.output_width);
    bind(encoder, gradient_output, 0);
    bind(encoder, gradient_bias, 1);
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    dispatch_groups(encoder, pipeline, shape.channels_out, width);
}

void encode_activation(
    id<MTLComputeCommandEncoder> encoder,
    NnMetalPrograms& programs,
    const Borrowed& input,
    const Borrowed& output,
    std::uint32_t count,
    std::uint32_t code) {
    bind(encoder, input, 0);
    bind(encoder, output, 1);
    [encoder setBytes:&count length:sizeof(count) atIndex:2];
    [encoder setBytes:&code length:sizeof(code) atIndex:3];
    dispatch_elements(
        encoder, programs.pipelines[kernel_activation_forward], count);
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
// output element; `write_only` takes a forward output write-only (a gradient
// written by an adjoint is borrowed as lent). No autograd is attached here.
extern "C" int nn_metal_conv2d_forward(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride,
    long long padding,
    long long groups,
    int write_only) {
    @autoreleasepool {
        ConvShape shape{};
        if (!conv_shape(input, weight, output, stride, padding, groups, shape))
            return 2;
        if (bias && !bias_shape_ok(bias, input, shape)) return 2;
        Borrowed x, w, b, y;
        return encode_call(
            dispatch_conv2d_forward, input, 1, conv_bytes(shape),
            [&] {
                x = borrow_const(input);
                w = borrow_const(weight);
                b = bias ? borrow_const(bias) : w;
                y = borrow_written(output, write_only != 0);
                return x && w && b && y;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                encode_forward(encoder, programs, x, w, b, y, bias != nullptr,
                               shape);
                return 0;
            },
            write_only != 0 ? std::initializer_list<void*>{output}
                            : std::initializer_list<void*>{});
    }
}

// First-order Conv2D gradients in one call: dInput, dWeight and dBias, each
// written by exactly one thread or threadgroup per element. A null gradient
// is not computed (the masked autograd ABI requests only the gradients a
// backward target needs); at least one must be requested.
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
            (!gradient_input && !gradient_weight && !gradient_bias) ||
            (gradient_input &&
             (!metal_f32(gradient_input) ||
              !same_device(input, gradient_input) ||
              !same_extents(input, gradient_input))) ||
            (gradient_weight &&
             (!metal_f32(gradient_weight) ||
              !same_device(input, gradient_weight) ||
              !same_extents(weight, gradient_weight))) ||
            (gradient_bias && !bias_shape_ok(gradient_bias, input, shape)))
            return 2;
        const std::uint32_t dispatches =
            (gradient_input ? 1u : 0u) + (gradient_weight ? 1u : 0u) +
            (gradient_bias ? 1u : 0u);
        Borrowed x, w, upstream, dx, dw, db;
        return encode_call(
            dispatch_conv2d_backward, input, dispatches,
            dispatches * conv_bytes(shape),
            [&] {
                upstream = borrow_const(gradient_output);
                if (gradient_input) {
                    w = borrow_const(weight);
                    dx = borrow_mutable(gradient_input);
                    if (!w || !dx) return false;
                }
                if (gradient_weight) {
                    x = borrow_const(input);
                    dw = borrow_mutable(gradient_weight);
                    if (!x || !dw) return false;
                }
                if (gradient_bias) {
                    db = borrow_mutable(gradient_bias);
                    if (!db) return false;
                }
                return static_cast<bool>(upstream);
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                if (gradient_input)
                    encode_backward_data(encoder, programs, w, upstream, dx, shape);
                if (gradient_weight)
                    encode_backward_filter(encoder, programs, x, upstream, dw, shape);
                if (gradient_bias)
                    encode_backward_bias(encoder, programs, upstream, db, shape);
                count_dispatch(dispatch_conv2d_backward_gradients, dispatches);
                return 0;
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
        Borrowed w, upstream, dx;
        return encode_call(
            dispatch_conv2d_backward_data, weight, 1, conv_bytes(shape),
            [&] {
                w = borrow_const(weight);
                upstream = borrow_const(gradient_output);
                dx = borrow_mutable(gradient_input);
                return w && upstream && dx;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                encode_backward_data(encoder, programs, w, upstream, dx, shape);
                return 0;
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
        Borrowed x, upstream, dw;
        return encode_call(
            dispatch_conv2d_backward_filter, input, 1, conv_bytes(shape),
            [&] {
                x = borrow_const(input);
                upstream = borrow_const(gradient_output);
                dw = borrow_mutable(gradient_weight);
                return x && upstream && dw;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                encode_backward_filter(encoder, programs, x, upstream, dw, shape);
                return 0;
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
        Borrowed bias, y;
        return encode_call(
            dispatch_conv2d_bias_broadcast, output, 1, float_bytes(count),
            [&] {
                bias = borrow_const(bias_gradient);
                y = borrow_mutable(output);
                return bias && y;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, bias, 0);
                bind(encoder, y, 1);
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
        Borrowed x, y;
        return encode_call(
            dispatch_activation_forward, input, 1, 2 * float_bytes(count),
            [&] {
                x = borrow_const(input);
                y = borrow_output(output);
                return x && y;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                encode_activation(encoder, programs, x, y, count, code);
                return 0;
            },
            {output});
    }
}

namespace {

int activation_backward(
    const void* input,
    const void* gradient_output,
    void* gradient_input,
    long long activation,
    DispatchKind kind) {
    std::uint32_t count = 0;
    std::uint32_t code = 0;
    if (!activation_code(activation, code) ||
        !activation_tensors({input, gradient_output, gradient_input}, count))
        return 2;
    if (count == 0) return 0;
    Borrowed x, upstream, dx;
    return encode_call(
        kind, input, 1, 3 * float_bytes(count),
        [&] {
            x = borrow_const(input);
            upstream = borrow_const(gradient_output);
            dx = borrow_mutable(gradient_input);
            return x && upstream && dx;
        },
        [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
            bind(encoder, x, 0);
            bind(encoder, upstream, 1);
            bind(encoder, dx, 2);
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder setBytes:&code length:sizeof(code) atIndex:4];
            dispatch_elements(
                encoder, programs.pipelines[kernel_activation_backward], count);
            return 0;
        });
}

} // namespace

extern "C" int nn_metal_activation_backward(
    const void* input,
    const void* gradient_output,
    void* gradient_input,
    long long activation) {
    @autoreleasepool {
        return activation_backward(
            input, gradient_output, gradient_input, activation,
            dispatch_activation_backward);
    }
}

// The activation backward of a Conv2D fusion target, whose pre-activation
// input only NN's Conv2D node reads. It runs exactly like
// nn_metal_activation_backward and is counted separately, so tests can tell
// the fused target from the separate calls.
extern "C" int nn_metal_activation_backward_internal(
    const void* input,
    const void* gradient_output,
    void* gradient_input,
    long long activation) {
    @autoreleasepool {
        return activation_backward(
            input, gradient_output, gradient_input, activation,
            dispatch_activation_backward_internal);
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
        Borrowed x, first, upstream, dx, dfirst;
        return encode_call(
            dispatch_activation_second_backward, input, 1,
            5 * float_bytes(count),
            [&] {
                x = borrow_const(input);
                first = borrow_const(first_gradient);
                upstream = borrow_const(gradient_output);
                dx = borrow_mutable(gradient_input);
                dfirst = borrow_mutable(gradient_first);
                return x && first && upstream && dx && dfirst;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, x, 0);
                bind(encoder, first, 1);
                bind(encoder, upstream, 2);
                bind(encoder, dx, 3);
                bind(encoder, dfirst, 4);
                [encoder setBytes:&count length:sizeof(count) atIndex:5];
                [encoder setBytes:&code length:sizeof(code) atIndex:6];
                dispatch_elements(
                    encoder,
                    programs.pipelines[kernel_activation_second_backward],
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

// output[n, c] = mean of input[n, c, :, :]; `write_only` as for
// nn_metal_conv2d_forward (the pooling is also the adjoint of the unpooling).
extern "C" int nn_metal_global_average_pool(
    const void* input,
    void* output,
    int write_only) {
    @autoreleasepool {
        std::uint32_t rows = 0;
        std::uint32_t plane = 0;
        if (!pooling_tensors(input, output, rows, plane)) return 2;
        Borrowed x, y;
        return encode_call(
            dispatch_global_average_pool, input, 1,
            float_bytes(static_cast<std::uint64_t>(rows) * plane + rows),
            [&] {
                x = borrow_const(input);
                y = borrow_written(output, write_only != 0);
                return x && y;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                id<MTLComputePipelineState> pipeline =
                    programs.pipelines[kernel_global_average_pool];
                const std::uint32_t width = reduction_width(pipeline, plane);
                bind(encoder, x, 0);
                bind(encoder, y, 1);
                [encoder setBytes:&plane length:sizeof(plane) atIndex:2];
                [encoder setBytes:&width length:sizeof(width) atIndex:3];
                dispatch_groups(encoder, pipeline, rows, width);
                return 0;
            },
            write_only != 0 ? std::initializer_list<void*>{output}
                            : std::initializer_list<void*>{});
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
        Borrowed upstream, y;
        return encode_call(
            dispatch_global_average_unpool, gradient, 1,
            float_bytes(static_cast<std::uint64_t>(count) + rows),
            [&] {
                upstream = borrow_const(gradient);
                y = borrow_mutable(output);
                return upstream && y;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, upstream, 0);
                bind(encoder, y, 1);
                [encoder setBytes:&plane length:sizeof(plane) atIndex:2];
                [encoder setBytes:&count length:sizeof(count) atIndex:3];
                dispatch_elements(
                    encoder, programs.pipelines[kernel_global_average_unpool],
                    count);
                return 0;
            });
    }
}

// One fused Adam update for dense float32 Metal tensors of one shape,
// encoded into Core's command stream like every other NN kernel: it reads the
// Parameter and its moments `first` and `second`, and writes the next
// Parameter and moments to `next_parameter`, `next_first` and `next_second`,
// write-only outputs, so the current state is never written. Updates of
// several Parameters share Core's command buffer, which runs them in program
// order. A refused call (non-zero) encodes nothing.
extern "C" int nn_metal_adam_update(
    const void* parameter,
    const void* gradient,
    void* first,
    void* second,
    void* next_parameter,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float epsilon,
    float scale) {
    void* next_first = first;
    void* next_second = second;
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
        Borrowed p, g, m, v, next_p, next_m, next_v;
        return encode_call(
            dispatch_adam_update, parameter, 1, 6 * float_bytes(count),
            [&] {
                p = borrow_const(parameter);
                g = borrow_const(gradient);
                m = borrow_mutable(first);
                v = borrow_mutable(second);
                next_p = borrow_output(next_parameter);
                next_m = m;
                next_v = v;
                // Moment borrows preserve their old contents on detach. The
                // same buffers supply both reads and writes for each element.
                return p && g && m && v && next_p && next_m && next_v;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, p, 0);
                bind(encoder, g, 1);
                bind(encoder, m, 2);
                bind(encoder, v, 3);
                bind(encoder, next_p, 4);
                bind(encoder, next_m, 5);
                bind(encoder, next_v, 6);
                [encoder setBytes:&coefficients length:sizeof(coefficients)
                          atIndex:7];
                dispatch_elements(
                    encoder, programs.pipelines[kernel_adam_update], count);
                return 0;
            },
            {next_parameter, next_first, next_second});
    }
}

// Number of NN Metal calls of one DispatchKind encoded in this process (0
// for an unknown kind). Test probe only; see DispatchKind.
extern "C" unsigned long long nn_metal_dispatch_count(int kind) {
    if (kind <= 0 || kind >= dispatch_kinds) return 0;
    return dispatch_counts[kind].load(std::memory_order_relaxed);
}

// Conv2D followed by an NN activation in one call: conv_output =
// conv2d(input, weight) + bias, then activation_output =
// activation(conv_output). Both results are write-only outputs.
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
        Borrowed x, w, b, convolved, activated;
        return encode_call(
            dispatch_conv2d_activation_forward, input, 2,
            conv_bytes(shape) + 2 * float_bytes(count),
            [&] {
                x = borrow_const(input);
                w = borrow_const(weight);
                b = borrow_const(bias);
                convolved = borrow_output(conv_output);
                activated = borrow_output(activation_output);
                return x && w && b && convolved && activated;
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                encode_forward(encoder, programs, x, w, b, convolved, true,
                               shape);
                // Core's encoder dispatches serially, so the activation reads
                // what the convolution wrote.
                encode_activation(encoder, programs, convolved, activated,
                                  count, code);
                return 0;
            },
            {conv_output, activation_output});
    }
}

extern "C" int nn_metal_sgd_update(const void* parameter, const void* gradient,
                                     void* output, float rate) {
    @autoreleasepool {
        std::uint32_t count = 0;
        if (!activation_tensors({parameter, gradient, output}, count) ||
            !same_extents(parameter, gradient) || !same_extents(parameter, output)) return 2;
        struct { float rate; std::uint32_t count; } coefficients{rate, count};
        Borrowed p, g, next;
        return encode_call(dispatch_sgd_update, parameter, 1, 2 * float_bytes(count),
            [&] { p = borrow_const(parameter); g = borrow_const(gradient);
                  next = borrow_output(output); return p && g && next; },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, p, 0); bind(encoder, g, 1); bind(encoder, next, 2);
                [encoder setBytes:&coefficients length:sizeof(coefficients) atIndex:3];
                dispatch_elements(encoder, programs.pipelines[kernel_sgd_update], count);
                return 0;
            }, {output});
    }
}

extern "C" long long nn_metal_half_exact_probe(long long device, int mode) {
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(half_probe_mutex);
        auto it = half_probe_buffers.find(device);
        if (mode == 1 && it == half_probe_buffers.end()) {
            void* storage = qcore_device_buffer_allocate(device, sizeof(std::uint32_t));
            if (!storage) return -1;
            it = half_probe_buffers.emplace(device, storage).first;
        } else if (it == half_probe_buffers.end()) return mode == 0 ? 0 : -1;
        if (qcore_device_wait(device) != 0) return -1;
        id<MTLBuffer> buffer = bridge_handle<id<MTLBuffer>>(qcore_device_buffer_handle(it->second));
        if (!buffer || !buffer.contents) return -1;
        auto* count = static_cast<std::uint32_t*>(buffer.contents);
        if (mode == 2) return *count;
        if (mode == 1) { *count = 0; return 0; }
        if (mode == 0) {
            qcore_device_buffer_release(it->second); half_probe_buffers.erase(it); return 0;
        }
        return -1;
    }
}

extern "C" int nn_metal_half_update(const void* parameter, const void* gradient,
    void* first, void* second, void* output, float beta1, float beta2,
    float one_minus_beta1, float one_minus_beta2, float epsilon, float scale, int adam) {
    @autoreleasepool {
        const int dtype = qcore_tensor_dtype(parameter);
        const auto device = qcore_tensor_device(parameter);
        if ((dtype != QCORE_DTYPE_FLOAT16 && dtype != QCORE_DTYPE_BFLOAT16) ||
            device < 0 || !fits_u32(qcore_tensor_element_count(parameter))) return 2;
        for (const void* tensor : {parameter, gradient, static_cast<const void*>(output)}) {
            if (!tensor || qcore_tensor_dtype(tensor) != dtype ||
                qcore_tensor_backend(tensor) != QCORE_BACKEND_METAL ||
                !qcore_tensor_is_contiguous(tensor) || !same_device(parameter, tensor) ||
                !same_extents(parameter, tensor)) return 2;
        }
        if (adam) for (const void* tensor : {first, second}) {
            if (!metal_f32(tensor) || !same_device(parameter, tensor) ||
                !same_extents(parameter, tensor)) return 2;
        }
        const auto count = static_cast<std::uint32_t>(qcore_tensor_element_count(parameter));
        struct {
            float beta1, beta2, one_minus_beta1, one_minus_beta2, epsilon, scale;
            std::uint32_t count, bfloat, adam, count_exact;
        } c{beta1, beta2, one_minus_beta1, one_minus_beta2, epsilon, scale,
            count, dtype == QCORE_DTYPE_BFLOAT16 ? 1u : 0u, adam ? 1u : 0u, 0};
        NnDeviceStatus conversion(device), nonfinite(device);
        Borrowed p, g, m, v, next, probe;
        bool encoded = false;
        const int result = encode_call(dispatch_half_update, parameter, 1,
            2 * count + (adam ? 8ull * count : 0),
            [&] {
                p = borrow_const(parameter); g = borrow_const(gradient);
                next = borrow_output(output);
                if (adam) { m = borrow_mutable(first); v = borrow_mutable(second); }
                else { m = next; v = next; }
                probe = half_probe_buffer(device); c.count_exact = probe ? 1u : 0u;
                return p && g && m && v && next && conversion.take() && nonfinite.take();
            },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, p, 0); bind(encoder, g, 1); bind(encoder, m, 2); bind(encoder, v, 3);
                bind(encoder, next, 4); [encoder setBytes:&c length:sizeof(c) atIndex:5];
                conversion.bind_to(encoder, 6); nonfinite.bind_to(encoder, 7);
                if (probe) bind(encoder, probe, 8); else conversion.bind_to(encoder, 8);
                encoded = true;
                dispatch_elements(encoder, programs.pipelines[kernel_half_update], count);
                return 0;
            }, {output});
        if (!encoded) return result;
        // Both ownership handoffs run even when one fails.
        const bool conversion_deferred = conversion.defer(dtype, nullptr);
        const bool nonfinite_deferred = nonfinite.defer(0, "package operation nn.Adam.update produced a NaN real32 value");
        // Both slot handoffs/consumes finish before a diagnostic may throw.
        if (conversion.failed_check()) qcore_tensor_report_conversion_status(output, 1);
        if (nonfinite.failed_check()) qcore_tensor_report_status(first, "nn.Adam.update", 1);
        return result == 0 && conversion_deferred && nonfinite_deferred ? 0 : 6;
    }
}

extern "C" int nn_metal_validate_output(const void* output, const char* operation, const char* message) {
    @autoreleasepool {
        if (!metal_f32(output) || !fits_u32(qcore_tensor_element_count(output))) return 6;
        const auto count = static_cast<std::uint32_t>(qcore_tensor_element_count(output));
        NnDeviceStatus status(qcore_tensor_device(output));
        Borrowed values;
        bool encoded = false;
        const int result = encode_call(dispatch_validate_finite, output, 1, 0,
            [&] { values = borrow_const(output); return values && status.take(); },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, values, 0); status.bind_to(encoder, 1);
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
                encoded = true;
                dispatch_elements(encoder, programs.pipelines[kernel_validate_finite], count);
                return 0;
            });
        if (!encoded) return result == 0 ? 6 : result;
        const bool deferred = status.defer(0, message);
        if (status.failed_check()) qcore_tensor_report_status(output, operation, 1);
        return result == 0 && deferred ? 0 : 6;
    }
}

extern "C" int nn_metal_half_probe_words(const void* left, const void* right,
    void* output, int operation, int bfloat) {
    @autoreleasepool {
        if (operation < 1 || operation > 6 || (bfloat != 0 && bfloat != 1) ||
            !left || !fits_u32(qcore_tensor_element_count(left))) return 2;
        for (const void* tensor : {left, right, static_cast<const void*>(output)}) {
            if (!tensor || qcore_tensor_dtype(tensor) != QCORE_DTYPE_UINT32 ||
                qcore_tensor_backend(tensor) != QCORE_BACKEND_METAL ||
                !qcore_tensor_is_contiguous(tensor) || !same_device(left, tensor) ||
                !same_extents(left, tensor)) return 2;
        }
        const auto count = static_cast<std::uint32_t>(qcore_tensor_element_count(left));
        std::uint32_t options[4] = {count, static_cast<std::uint32_t>(operation), static_cast<std::uint32_t>(bfloat), 0};
        Borrowed a, b, out;
        return encode_call(dispatch_half_update, left, 1, 4ull * count,
            [&] { a = borrow_const(left); b = borrow_const(right); out = borrow_output(output); return a && b && out; },
            [&](id<MTLComputeCommandEncoder> encoder, NnMetalPrograms& programs) {
                bind(encoder, a, 0); bind(encoder, b, 1); bind(encoder, out, 2);
                [encoder setBytes:options length:sizeof(options) atIndex:3];
                dispatch_elements(encoder, programs.pipelines[kernel_half_probe_words], count);
                return 0;
            }, {output});
    }
}
