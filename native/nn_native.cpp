#include <quidra/native_extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef __APPLE__
// NN-owned Metal kernels live in native/nn_metal.mm, which the package
// manifest declares for macOS only. Every other platform keeps the portable
// Quidra fallbacks for Metal-less builds.
extern "C" long long nn_metal_device_f32(const void* tensor);
extern "C" int nn_metal_conv2d_forward(
    const void* input, const void* weight, const void* bias, void* output,
    long long stride, long long padding, long long groups);
extern "C" int nn_metal_conv2d_backward(
    const void* input, const void* weight, const void* gradient_output,
    void* gradient_input, void* gradient_weight, void* gradient_bias,
    long long stride, long long padding, long long groups);
extern "C" int nn_metal_conv2d_backward_data(
    const void* weight, const void* gradient_output, void* gradient_input,
    long long stride, long long padding, long long groups);
extern "C" int nn_metal_conv2d_backward_filter(
    const void* input, const void* gradient_output, void* gradient_weight,
    long long stride, long long padding, long long groups);
extern "C" int nn_metal_conv2d_bias_broadcast(
    const void* bias_gradient, void* output);
extern "C" int nn_metal_activation_forward(
    const void* input, void* output, long long activation);
extern "C" int nn_metal_activation_backward(
    const void* input, const void* gradient_output, void* gradient_input,
    long long activation);
extern "C" int nn_metal_activation_backward_internal(
    const void* input, const void* gradient_output, void* gradient_input,
    long long activation);
extern "C" int nn_metal_activation_second_backward(
    const void* input, const void* first_gradient, const void* gradient_output,
    void* gradient_input, void* gradient_first, long long activation);
extern "C" int nn_metal_global_average_pool(const void* input, void* output);
extern "C" int nn_metal_global_average_unpool(
    const void* gradient, void* output);
extern "C" int nn_metal_conv2d_activation_forward(
    const void* input, const void* weight, const void* bias,
    void* conv_output, void* activation_output, long long stride,
    long long padding, long long groups, long long activation);
extern "C" std::int32_t nn_metal_adam_begin();
extern "C" int nn_metal_adam_encode(
    std::int32_t batch, const void* parameter, const void* gradient, const void* first,
    const void* second, void* next_parameter, void* next_first,
    void* next_second, float beta1, float beta2, float one_minus_beta1,
    float one_minus_beta2, float epsilon, float scale);
extern "C" int nn_metal_adam_commit(
    std::int32_t batch, std::uint64_t expected_dispatches);
extern "C" unsigned long long nn_metal_dispatch_count(int kind);
#endif

namespace {

bool positive(long long value) {
    return value > 0;
}

bool multiply_ok(std::size_t left, std::size_t right) {
    return right == 0 || left <= std::numeric_limits<std::size_t>::max() / right;
}

class DynamicLibrary {
public:
    DynamicLibrary() = default;
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    DynamicLibrary(DynamicLibrary&& other) noexcept
        : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
        if (this == &other) return *this;
        close();
        handle_ = other.handle_;
        other.handle_ = nullptr;
        return *this;
    }

    ~DynamicLibrary() {
        close();
    }

    bool open(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_ = LoadLibraryW(path.c_str());
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        return handle_ != nullptr;
    }

    template <typename T>
    T symbol(const char* name) const {
        if (!handle_) return nullptr;
#ifdef _WIN32
        return reinterpret_cast<T>(GetProcAddress(handle_, name));
#else
        return reinterpret_cast<T>(dlsym(handle_, name));
#endif
    }

private:
    void close() noexcept {
#ifdef _WIN32
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
        handle_ = nullptr;
    }

#ifdef _WIN32
    HMODULE handle_{};
#else
    void* handle_{};
#endif
};

using CudnnStatus = int;
using CudnnHandle = void*;
using CudnnTensorDescriptor = void*;
using CudnnFilterDescriptor = void*;
using CudnnConvolutionDescriptor = void*;

constexpr CudnnStatus cudnn_success = 0;
constexpr int cudnn_data_float = 0;
constexpr int cudnn_tensor_nchw = 0;
constexpr int cudnn_cross_correlation = 1;
constexpr int cudnn_conv_fwd_implicit_gemm = 0;
constexpr int cudnn_conv_bwd_data_algo_0 = 0;
constexpr int cudnn_conv_bwd_filter_algo_0 = 0;
constexpr int cudnn_default_math = 0;
constexpr std::size_t cudnn_fast_workspace_limit =
    std::size_t{64} * 1024 * 1024;
constexpr int cudnn_heuristic_result_count = 8;

struct CudnnAlgoPerf {
    int algorithm{};
    CudnnStatus status{};
    float time{};
    std::size_t memory{};
    int determinism{};
    int math_type{};
    int reserved[3]{};
};

struct CudnnApi {
    DynamicLibrary library;
    CudnnStatus (*create)(CudnnHandle*){};
    CudnnStatus (*destroy)(CudnnHandle){};
    CudnnStatus (*create_tensor)(CudnnTensorDescriptor*){};
    CudnnStatus (*destroy_tensor)(CudnnTensorDescriptor){};
    CudnnStatus (*set_tensor4d)(
        CudnnTensorDescriptor, int, int, int, int, int, int){};
    CudnnStatus (*create_filter)(CudnnFilterDescriptor*){};
    CudnnStatus (*destroy_filter)(CudnnFilterDescriptor){};
    CudnnStatus (*set_filter4d)(
        CudnnFilterDescriptor, int, int, int, int, int, int){};
    CudnnStatus (*create_convolution)(CudnnConvolutionDescriptor*){};
    CudnnStatus (*destroy_convolution)(CudnnConvolutionDescriptor){};
    CudnnStatus (*set_convolution2d)(
        CudnnConvolutionDescriptor, int, int, int, int, int, int, int, int){};
    CudnnStatus (*set_group_count)(CudnnConvolutionDescriptor, int){};
    CudnnStatus (*set_math_type)(CudnnConvolutionDescriptor, int){};
    CudnnStatus (*workspace_size)(
        CudnnHandle, CudnnTensorDescriptor, CudnnFilterDescriptor,
        CudnnConvolutionDescriptor, CudnnTensorDescriptor, int,
        std::size_t*){};
    CudnnStatus (*forward_algorithms)(
        CudnnHandle, CudnnTensorDescriptor, CudnnFilterDescriptor,
        CudnnConvolutionDescriptor, CudnnTensorDescriptor, int, int*,
        CudnnAlgoPerf*){};
    CudnnStatus (*convolution_forward)(
        CudnnHandle, const void*, CudnnTensorDescriptor, const void*,
        CudnnFilterDescriptor, const void*, CudnnConvolutionDescriptor, int,
        void*, std::size_t, const void*, CudnnTensorDescriptor, void*){};
    CudnnStatus (*backward_data_workspace_size)(
        CudnnHandle, CudnnFilterDescriptor, CudnnTensorDescriptor,
        CudnnConvolutionDescriptor, CudnnTensorDescriptor, int,
        std::size_t*){};
    CudnnStatus (*backward_data_algorithms)(
        CudnnHandle, CudnnFilterDescriptor, CudnnTensorDescriptor,
        CudnnConvolutionDescriptor, CudnnTensorDescriptor, int, int*,
        CudnnAlgoPerf*){};
    CudnnStatus (*convolution_backward_data)(
        CudnnHandle, const void*, CudnnFilterDescriptor, const void*,
        CudnnTensorDescriptor, const void*, CudnnConvolutionDescriptor, int,
        void*, std::size_t, const void*, CudnnTensorDescriptor, void*){};
    CudnnStatus (*backward_filter_workspace_size)(
        CudnnHandle, CudnnTensorDescriptor, CudnnTensorDescriptor,
        CudnnConvolutionDescriptor, CudnnFilterDescriptor, int,
        std::size_t*){};
    CudnnStatus (*backward_filter_algorithms)(
        CudnnHandle, CudnnTensorDescriptor, CudnnTensorDescriptor,
        CudnnConvolutionDescriptor, CudnnFilterDescriptor, int, int*,
        CudnnAlgoPerf*){};
    CudnnStatus (*convolution_backward_filter)(
        CudnnHandle, const void*, CudnnTensorDescriptor, const void*,
        CudnnTensorDescriptor, const void*, CudnnConvolutionDescriptor, int,
        void*, std::size_t, const void*, CudnnFilterDescriptor, void*){};
    CudnnStatus (*convolution_backward_bias)(
        CudnnHandle, const void*, CudnnTensorDescriptor, const void*,
        const void*, CudnnTensorDescriptor, void*){};
    CudnnStatus (*add_tensor)(
        CudnnHandle, const void*, CudnnTensorDescriptor, const void*,
        const void*, CudnnTensorDescriptor, void*){};
    bool ready() const {
        return create && destroy && create_tensor && destroy_tensor &&
               set_tensor4d && create_filter && destroy_filter &&
               set_filter4d && create_convolution && destroy_convolution &&
               set_convolution2d && set_group_count && workspace_size &&
               convolution_forward && add_tensor;
    }

    bool backward_ready() const {
        return ready() && backward_data_workspace_size &&
               convolution_backward_data && backward_filter_workspace_size &&
               convolution_backward_filter && convolution_backward_bias;
    }
};

std::filesystem::path package_root() {
    auto source = std::filesystem::path(__FILE__);
    if (!source.is_absolute())
        source = std::filesystem::absolute(source);
    // .../<package>/native/nn_native.cpp
    return source.parent_path().parent_path().lexically_normal();
}

std::filesystem::path nvidia_library_directory() {
    if (const char* configured =
            std::getenv("QUIDRA_NN_NVIDIA_LIBRARY_PATH");
        configured && *configured) {
        return std::filesystem::path(configured).lexically_normal();
    }
    return package_root() / "nvidia" / "lib";
}

CudnnApi load_cudnn() {
    CudnnApi api;
    const auto directory = nvidia_library_directory();
#ifdef _WIN32
    const auto primary = directory / "cudnn64_9.dll";
#else
    const auto primary = directory / "libcudnn.so.9";
#endif
    if (!std::filesystem::is_regular_file(primary) ||
        !api.library.open(primary)) {
        return api;
    }

    api.create = api.library.symbol<decltype(api.create)>("cudnnCreate");
    api.destroy = api.library.symbol<decltype(api.destroy)>("cudnnDestroy");
    api.create_tensor =
        api.library.symbol<decltype(api.create_tensor)>(
            "cudnnCreateTensorDescriptor");
    api.destroy_tensor =
        api.library.symbol<decltype(api.destroy_tensor)>(
            "cudnnDestroyTensorDescriptor");
    api.set_tensor4d =
        api.library.symbol<decltype(api.set_tensor4d)>(
            "cudnnSetTensor4dDescriptor");
    api.create_filter =
        api.library.symbol<decltype(api.create_filter)>(
            "cudnnCreateFilterDescriptor");
    api.destroy_filter =
        api.library.symbol<decltype(api.destroy_filter)>(
            "cudnnDestroyFilterDescriptor");
    api.set_filter4d =
        api.library.symbol<decltype(api.set_filter4d)>(
            "cudnnSetFilter4dDescriptor");
    api.create_convolution =
        api.library.symbol<decltype(api.create_convolution)>(
            "cudnnCreateConvolutionDescriptor");
    api.destroy_convolution =
        api.library.symbol<decltype(api.destroy_convolution)>(
            "cudnnDestroyConvolutionDescriptor");
    api.set_convolution2d =
        api.library.symbol<decltype(api.set_convolution2d)>(
            "cudnnSetConvolution2dDescriptor");
    api.set_group_count =
        api.library.symbol<decltype(api.set_group_count)>(
            "cudnnSetConvolutionGroupCount");
    api.set_math_type =
        api.library.symbol<decltype(api.set_math_type)>(
            "cudnnSetConvolutionMathType");
    api.workspace_size =
        api.library.symbol<decltype(api.workspace_size)>(
            "cudnnGetConvolutionForwardWorkspaceSize");
    api.forward_algorithms =
        api.library.symbol<decltype(api.forward_algorithms)>(
            "cudnnGetConvolutionForwardAlgorithm_v7");
    api.convolution_forward =
        api.library.symbol<decltype(api.convolution_forward)>(
            "cudnnConvolutionForward");
    api.backward_data_workspace_size =
        api.library.symbol<decltype(api.backward_data_workspace_size)>(
            "cudnnGetConvolutionBackwardDataWorkspaceSize");
    api.backward_data_algorithms =
        api.library.symbol<decltype(api.backward_data_algorithms)>(
            "cudnnGetConvolutionBackwardDataAlgorithm_v7");
    api.convolution_backward_data =
        api.library.symbol<decltype(api.convolution_backward_data)>(
            "cudnnConvolutionBackwardData");
    api.backward_filter_workspace_size =
        api.library.symbol<decltype(api.backward_filter_workspace_size)>(
            "cudnnGetConvolutionBackwardFilterWorkspaceSize");
    api.backward_filter_algorithms =
        api.library.symbol<decltype(api.backward_filter_algorithms)>(
            "cudnnGetConvolutionBackwardFilterAlgorithm_v7");
    api.convolution_backward_filter =
        api.library.symbol<decltype(api.convolution_backward_filter)>(
            "cudnnConvolutionBackwardFilter");
    api.convolution_backward_bias =
        api.library.symbol<decltype(api.convolution_backward_bias)>(
            "cudnnConvolutionBackwardBias");
    api.add_tensor =
        api.library.symbol<decltype(api.add_tensor)>("cudnnAddTensor");
    return api;
}

CudnnApi& cudnn() {
    static CudnnApi api = load_cudnn();
    return api;
}

int select_fast_cudnn_algorithm(
    CudnnStatus (*query)(
        CudnnHandle, CudnnTensorDescriptor, CudnnFilterDescriptor,
        CudnnConvolutionDescriptor, CudnnTensorDescriptor, int, int*,
        CudnnAlgoPerf*),
    CudnnHandle handle,
    CudnnTensorDescriptor x,
    CudnnFilterDescriptor weight,
    CudnnConvolutionDescriptor convolution,
    CudnnTensorDescriptor y,
    int fallback) {
    if ((qcore_execution_policy_get() == QCORE_EXECUTION_DETERMINISTIC) || !query) return fallback;

    std::array<CudnnAlgoPerf, cudnn_heuristic_result_count> results{};
    int returned = 0;
    if (query(
            handle, x, weight, convolution, y,
            static_cast<int>(results.size()), &returned,
            results.data()) != cudnn_success) {
        return fallback;
    }
    const int available =
        std::min(returned, static_cast<int>(results.size()));
    for (int index = 0; index < available; ++index) {
        const auto& candidate = results[static_cast<std::size_t>(index)];
        if (candidate.status == cudnn_success &&
            candidate.math_type == cudnn_default_math &&
            candidate.memory <= cudnn_fast_workspace_limit) {
            return candidate.algorithm;
        }
    }
    return fallback;
}

int select_fast_cudnn_backward_data_algorithm(
    CudnnApi& api, CudnnHandle handle,
    CudnnFilterDescriptor weight, CudnnTensorDescriptor output_gradient,
    CudnnConvolutionDescriptor convolution, CudnnTensorDescriptor input_gradient) {
    if ((qcore_execution_policy_get() == QCORE_EXECUTION_DETERMINISTIC) || !api.backward_data_algorithms)
        return cudnn_conv_bwd_data_algo_0;

    std::array<CudnnAlgoPerf, cudnn_heuristic_result_count> results{};
    int returned = 0;
    if (api.backward_data_algorithms(
            handle, weight, output_gradient, convolution, input_gradient,
            static_cast<int>(results.size()), &returned,
            results.data()) != cudnn_success) {
        return cudnn_conv_bwd_data_algo_0;
    }
    const int available =
        std::min(returned, static_cast<int>(results.size()));
    for (int index = 0; index < available; ++index) {
        const auto& candidate = results[static_cast<std::size_t>(index)];
        if (candidate.status == cudnn_success &&
            candidate.math_type == cudnn_default_math &&
            candidate.memory <= cudnn_fast_workspace_limit) {
            return candidate.algorithm;
        }
    }
    return cudnn_conv_bwd_data_algo_0;
}

int select_fast_cudnn_backward_filter_algorithm(
    CudnnApi& api, CudnnHandle handle,
    CudnnTensorDescriptor input, CudnnTensorDescriptor output_gradient,
    CudnnConvolutionDescriptor convolution, CudnnFilterDescriptor weight_gradient) {
    if ((qcore_execution_policy_get() == QCORE_EXECUTION_DETERMINISTIC) || !api.backward_filter_algorithms)
        return cudnn_conv_bwd_filter_algo_0;

    std::array<CudnnAlgoPerf, cudnn_heuristic_result_count> results{};
    int returned = 0;
    if (api.backward_filter_algorithms(
            handle, input, output_gradient, convolution, weight_gradient,
            static_cast<int>(results.size()), &returned,
            results.data()) != cudnn_success) {
        return cudnn_conv_bwd_filter_algo_0;
    }
    const int available =
        std::min(returned, static_cast<int>(results.size()));
    for (int index = 0; index < available; ++index) {
        const auto& candidate = results[static_cast<std::size_t>(index)];
        if (candidate.status == cudnn_success &&
            candidate.math_type == cudnn_default_math &&
            candidate.memory <= cudnn_fast_workspace_limit) {
            return candidate.algorithm;
        }
    }
    return cudnn_conv_bwd_filter_algo_0;
}

struct CudnnTensor {
    CudnnApi* api{};
    CudnnTensorDescriptor value{};
    ~CudnnTensor() {
        if (api && value && api->destroy_tensor)
            (void)api->destroy_tensor(value);
    }
};

struct CudnnFilter {
    CudnnApi* api{};
    CudnnFilterDescriptor value{};
    ~CudnnFilter() {
        if (api && value && api->destroy_filter)
            (void)api->destroy_filter(value);
    }
};

struct CudnnConvolution {
    CudnnApi* api{};
    CudnnConvolutionDescriptor value{};
    ~CudnnConvolution() {
        if (api && value && api->destroy_convolution)
            (void)api->destroy_convolution(value);
    }
};

struct CudnnSession {
    CudnnApi* api{};
    CudnnHandle value{};
    ~CudnnSession() {
        if (api && value && api->destroy)
            (void)api->destroy(value);
    }
};

const void* cuda_pointer_const(const void* tensor) {
    const auto base = qcore_tensor_device_handle_const(tensor);
    if (base == 0) return nullptr;
    const auto offset = qcore_tensor_device_offset_bytes(tensor);
    if (offset > std::numeric_limits<std::uint64_t>::max() - base)
        return nullptr;
    return reinterpret_cast<const void*>(base + offset);
}

void* cuda_pointer(void* tensor) {
    const auto base = qcore_tensor_device_handle(tensor);
    if (base == 0) return nullptr;
    const auto offset = qcore_tensor_device_offset_bytes(tensor);
    if (offset > std::numeric_limits<std::uint64_t>::max() - base)
        return nullptr;
    return reinterpret_cast<void*>(base + offset);
}


using NcclStatus = int;
using NcclComm = void*;
using NcclStream = void*;

struct NcclUniqueId {
    char internal[128]{};
};

constexpr NcclStatus nccl_success = 0;
constexpr int nccl_sum = 0;
constexpr int nccl_float32 = 7;
constexpr int nccl_float64 = 8;
constexpr std::size_t nccl_no_selection =
    std::numeric_limits<std::size_t>::max();

struct NcclApi {
    DynamicLibrary library;
    NcclStatus (*get_unique_id)(NcclUniqueId*){};
    NcclStatus (*comm_init_rank)(NcclComm*, int, NcclUniqueId, int){};
    NcclStatus (*comm_destroy)(NcclComm){};
    NcclStatus (*group_start)(){};
    NcclStatus (*group_end)(){};
    NcclStatus (*all_reduce)(
        const void*, void*, std::size_t, int, int, NcclComm, NcclStream){};

    bool ready() const {
        return get_unique_id && comm_init_rank && comm_destroy &&
               group_start && group_end && all_reduce;
    }
};

NcclApi load_nccl() {
    NcclApi api;
#ifndef _WIN32
    const auto primary = nvidia_library_directory() / "libnccl.so.2";
    if (!std::filesystem::is_regular_file(primary) ||
        !api.library.open(primary)) {
        return api;
    }
    api.get_unique_id =
        api.library.symbol<decltype(api.get_unique_id)>("ncclGetUniqueId");
    api.comm_init_rank =
        api.library.symbol<decltype(api.comm_init_rank)>("ncclCommInitRank");
    api.comm_destroy =
        api.library.symbol<decltype(api.comm_destroy)>("ncclCommDestroy");
    api.group_start =
        api.library.symbol<decltype(api.group_start)>("ncclGroupStart");
    api.group_end =
        api.library.symbol<decltype(api.group_end)>("ncclGroupEnd");
    api.all_reduce =
        api.library.symbol<decltype(api.all_reduce)>("ncclAllReduce");
#endif
    return api;
}

NcclApi& nccl() {
    // Keep the dynamic library alive through thread-local communicator teardown.
    static auto* api = new NcclApi(load_nccl());
    return *api;
}

struct NcclCommunicatorSet {
    std::vector<long long> devices;
    std::vector<long long> backend_devices;
    std::vector<NcclComm> communicators;
};

struct NcclState {
    std::vector<NcclCommunicatorSet> cache;
    int expected{};
    std::vector<long long> pending_devices;
    std::vector<long long> pending_backend_devices;
    std::vector<unsigned char> registered;
    int dtype{};
    std::uint64_t count{};
    std::size_t selected{nccl_no_selection};
    bool group_open{};

    void reset_pending() {
        expected = 0;
        pending_devices.clear();
        pending_backend_devices.clear();
        registered.clear();
        dtype = 0;
        count = 0;
        selected = nccl_no_selection;
    }

    ~NcclState() {
        auto& api = nccl();
        if (!api.comm_destroy) return;
        for (auto& set : cache) {
            for (std::size_t index = 0;
                 index < set.communicators.size(); ++index) {
                if (index < set.devices.size())
                    (void)qcore_device_activate(set.devices[index]);
                if (set.communicators[index])
                    (void)api.comm_destroy(set.communicators[index]);
            }
        }
    }
};

thread_local NcclState nccl_state;

bool nccl_select_pending_communicators() {
    auto& state = nccl_state;
    auto& api = nccl();
    if (!api.ready() || state.expected <= 1 ||
        state.pending_devices.size() != static_cast<std::size_t>(state.expected) ||
        state.pending_backend_devices.size() != static_cast<std::size_t>(state.expected) ||
        state.registered.size() != static_cast<std::size_t>(state.expected)) {
        return false;
    }
    for (const auto value : state.registered)
        if (value == 0) return false;

    for (std::size_t index = 0; index < state.cache.size(); ++index) {
        if (state.cache[index].devices == state.pending_devices &&
            state.cache[index].backend_devices == state.pending_backend_devices) {
            state.selected = index;
            return true;
        }
    }

    for (const auto device : state.pending_devices)
        if (!qcore_device_activate(device)) return false;

    NcclUniqueId unique{};
    if (api.get_unique_id(&unique) != nccl_success ||
        api.group_start() != nccl_success) {
        return false;
    }

    std::vector<NcclComm> communicators(
        static_cast<std::size_t>(state.expected), nullptr);
    bool ok = true;
    for (int rank = 0; rank < state.expected; ++rank) {
        if (!qcore_device_activate(
                state.pending_devices[static_cast<std::size_t>(rank)])) {
            ok = false;
            continue;
        }
        if (api.comm_init_rank(
                &communicators[static_cast<std::size_t>(rank)],
                state.expected, unique, rank) != nccl_success) {
            ok = false;
        }
    }
    if (api.group_end() != nccl_success) ok = false;

    if (!ok) {
        for (std::size_t index = 0; index < communicators.size(); ++index) {
            if (communicators[index]) {
                (void)qcore_device_activate(state.pending_devices[index]);
                (void)api.comm_destroy(communicators[index]);
            }
        }
        return false;
    }

    state.cache.push_back(NcclCommunicatorSet{
        state.pending_devices,
        state.pending_backend_devices,
        std::move(communicators)});
    state.selected = state.cache.size() - 1;
    return true;
}

struct Conv2dAutogradMetadata {
    std::int64_t stride{};
    std::int64_t padding{};
    std::int64_t groups{};
};

int conv2d_cuda_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !saved_tensors || saved_tensor_count != 3 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 3 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 101;
    }

    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    if (metadata.stride <= 0 || metadata.padding < 0 || metadata.groups <= 0)
        return 102;

    const void* input = saved_tensors[0];
    const void* weight = saved_tensors[1];
    const void* bias = saved_tensors[2];
    void* input_gradient = gradient_inputs[0];
    void* weight_gradient = gradient_inputs[1];
    void* bias_gradient = gradient_inputs[2];

    const void* read_tensors[] = {input, weight, bias, gradient_output};
    for (const void* tensor : read_tensors) {
        if (!tensor || qcore_tensor_dtype(tensor) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_backend(tensor) != QCORE_BACKEND_CUDA ||
            !qcore_tensor_is_contiguous(tensor)) {
            return 103;
        }
    }
    void* write_tensors[] = {
        input_gradient, weight_gradient, bias_gradient
    };
    for (void* tensor : write_tensors) {
        if (!tensor || qcore_tensor_dtype(tensor) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_backend(tensor) != QCORE_BACKEND_CUDA ||
            !qcore_tensor_is_contiguous(tensor)) {
            return 104;
        }
    }

    if (qcore_tensor_rank(input) != 4 || qcore_tensor_rank(weight) != 4 ||
        qcore_tensor_rank(bias) != 1 || qcore_tensor_rank(gradient_output) != 4 ||
        qcore_tensor_rank(input_gradient) != 4 ||
        qcore_tensor_rank(weight_gradient) != 4 ||
        qcore_tensor_rank(bias_gradient) != 1) {
        return 105;
    }

    const auto device = qcore_tensor_device(input);
    if (device < 0 ||
        qcore_tensor_device(weight) != device ||
        qcore_tensor_device(bias) != device ||
        qcore_tensor_device(gradient_output) != device ||
        qcore_tensor_device(input_gradient) != device ||
        qcore_tensor_device(weight_gradient) != device ||
        qcore_tensor_device(bias_gradient) != device ||
        !qcore_device_activate(device)) {
        return 106;
    }

    const auto n = qcore_tensor_extent(input, 0);
    const auto c = qcore_tensor_extent(input, 1);
    const auto h = qcore_tensor_extent(input, 2);
    const auto w = qcore_tensor_extent(input, 3);
    const auto k = qcore_tensor_extent(weight, 0);
    const auto weight_c = qcore_tensor_extent(weight, 1);
    const auto r = qcore_tensor_extent(weight, 2);
    const auto s = qcore_tensor_extent(weight, 3);
    const auto out_n = qcore_tensor_extent(gradient_output, 0);
    const auto out_c = qcore_tensor_extent(gradient_output, 1);
    const auto out_h = qcore_tensor_extent(gradient_output, 2);
    const auto out_w = qcore_tensor_extent(gradient_output, 3);
    if (!positive(n) || !positive(c) || !positive(h) || !positive(w) ||
        !positive(k) || !positive(weight_c) || !positive(r) || !positive(s) ||
        out_n != n || out_c != k || c % metadata.groups != 0 ||
        k % metadata.groups != 0 || weight_c != c / metadata.groups ||
        qcore_tensor_extent(bias, 0) != k ||
        h + metadata.padding * 2 < r ||
        w + metadata.padding * 2 < s ||
        (h + metadata.padding * 2 - r) / metadata.stride + 1 != out_h ||
        (w + metadata.padding * 2 - s) / metadata.stride + 1 != out_w ||
        n > std::numeric_limits<int>::max() ||
        c > std::numeric_limits<int>::max() ||
        h > std::numeric_limits<int>::max() ||
        w > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max() ||
        weight_c > std::numeric_limits<int>::max() ||
        r > std::numeric_limits<int>::max() ||
        s > std::numeric_limits<int>::max() ||
        out_h > std::numeric_limits<int>::max() ||
        out_w > std::numeric_limits<int>::max() ||
        metadata.stride > std::numeric_limits<int>::max() ||
        metadata.padding > std::numeric_limits<int>::max() ||
        metadata.groups > std::numeric_limits<int>::max()) {
        return 107;
    }

    const auto same_shape = [](const void* left, const void* right) {
        const auto rank = qcore_tensor_rank(left);
        if (rank != qcore_tensor_rank(right)) return false;
        for (std::uint64_t axis = 0; axis < rank; ++axis) {
            if (qcore_tensor_extent(left, axis) !=
                qcore_tensor_extent(right, axis)) {
                return false;
            }
        }
        return true;
    };
    if (!same_shape(input, input_gradient) ||
        !same_shape(weight, weight_gradient) ||
        !same_shape(bias, bias_gradient)) {
        return 108;
    }

    const auto* x = cuda_pointer_const(input);
    const auto* filter = cuda_pointer_const(weight);
    const auto* dy = cuda_pointer_const(gradient_output);
    auto* dx = cuda_pointer(input_gradient);
    auto* dfilter = cuda_pointer(weight_gradient);
    auto* dbias = cuda_pointer(bias_gradient);
    if (!x || !filter || !dy || !dx || !dfilter || !dbias) return 109;

    auto& api = cudnn();
    if (!api.backward_ready()) return 110;

    CudnnSession session{&api};
    CudnnTensor x_desc{&api};
    CudnnTensor y_desc{&api};
    CudnnTensor bias_desc{&api};
    CudnnFilter filter_desc{&api};
    CudnnConvolution conv_desc{&api};
    if (api.create(&session.value) != cudnn_success ||
        api.create_tensor(&x_desc.value) != cudnn_success ||
        api.create_tensor(&y_desc.value) != cudnn_success ||
        api.create_tensor(&bias_desc.value) != cudnn_success ||
        api.create_filter(&filter_desc.value) != cudnn_success ||
        api.create_convolution(&conv_desc.value) != cudnn_success) {
        return 111;
    }

    if (api.set_tensor4d(
            x_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            static_cast<int>(n), static_cast<int>(c),
            static_cast<int>(h), static_cast<int>(w)) != cudnn_success ||
        api.set_tensor4d(
            y_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            static_cast<int>(out_n), static_cast<int>(out_c),
            static_cast<int>(out_h), static_cast<int>(out_w)) != cudnn_success ||
        api.set_tensor4d(
            bias_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            1, static_cast<int>(k), 1, 1) != cudnn_success ||
        api.set_filter4d(
            filter_desc.value, cudnn_data_float, cudnn_tensor_nchw,
            static_cast<int>(k), static_cast<int>(weight_c),
            static_cast<int>(r), static_cast<int>(s)) != cudnn_success ||
        api.set_convolution2d(
            conv_desc.value,
            static_cast<int>(metadata.padding),
            static_cast<int>(metadata.padding),
            static_cast<int>(metadata.stride),
            static_cast<int>(metadata.stride),
            1, 1, cudnn_cross_correlation, cudnn_data_float) != cudnn_success ||
        api.set_group_count(
            conv_desc.value, static_cast<int>(metadata.groups)) != cudnn_success) {
        return 112;
    }
    if (api.set_math_type &&
        api.set_math_type(conv_desc.value, cudnn_default_math) != cudnn_success) {
        return 113;
    }

    const int data_algorithm =
        select_fast_cudnn_backward_data_algorithm(
            api, session.value, filter_desc.value, y_desc.value,
            conv_desc.value, x_desc.value);
    const int filter_algorithm =
        select_fast_cudnn_backward_filter_algorithm(
            api, session.value, x_desc.value, y_desc.value,
            conv_desc.value, filter_desc.value);

    std::size_t data_workspace = 0;
    std::size_t filter_workspace = 0;
    if (api.backward_data_workspace_size(
            session.value, filter_desc.value, y_desc.value, conv_desc.value,
            x_desc.value, data_algorithm,
            &data_workspace) != cudnn_success ||
        api.backward_filter_workspace_size(
            session.value, x_desc.value, y_desc.value, conv_desc.value,
            filter_desc.value, filter_algorithm,
            &filter_workspace) != cudnn_success ||
        data_workspace > cudnn_fast_workspace_limit ||
        filter_workspace > cudnn_fast_workspace_limit) {
        return 114;
    }

    const std::size_t workspace_size =
        data_workspace > filter_workspace ? data_workspace : filter_workspace;
    void* workspace_token = nullptr;
    void* workspace = nullptr;
    struct ScratchGuard {
        void*& token;
        ~ScratchGuard() {
            if (token) qcore_device_buffer_release(token);
        }
    } scratch{workspace_token};
    if (workspace_size != 0) {
        workspace_token = qcore_device_buffer_allocate(
            device, static_cast<std::uint64_t>(workspace_size));
        if (!workspace_token) return 114;
        const auto handle = qcore_device_buffer_handle(workspace_token);
        if (handle == 0) return 114;
        workspace = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(handle));
    }

    const float alpha = 1.0F;
    const float zero = 0.0F;
    if (api.convolution_backward_data(
            session.value, &alpha, filter_desc.value, filter,
            y_desc.value, dy, conv_desc.value, data_algorithm,
            workspace, data_workspace, &zero, x_desc.value, dx) != cudnn_success) {
        return 115;
    }
    if (api.convolution_backward_filter(
            session.value, &alpha, x_desc.value, x,
            y_desc.value, dy, conv_desc.value, filter_algorithm,
            workspace, filter_workspace, &zero, filter_desc.value, dfilter) != cudnn_success) {
        return 116;
    }
    if (api.convolution_backward_bias(
            session.value, &alpha, y_desc.value, dy,
            &zero, bias_desc.value, dbias) != cudnn_success) {
        return 117;
    }
    return 0;
}


struct CudaConvGeometry {
    long long device{};
    int batches{};
    int channels_in{};
    int channels_out{};
    int weight_channels{};
    int height{};
    int width{};
    int kernel_height{};
    int kernel_width{};
    int output_height{};
    int output_width{};
    int stride{};
    int padding{};
    int groups{};
};

bool cuda_conv_geometry(
    const void* input,
    const void* weight,
    const void* output,
    const Conv2dAutogradMetadata& metadata,
    CudaConvGeometry& geometry) {
    if (!input || !weight || !output ||
        metadata.stride <= 0 || metadata.padding < 0 || metadata.groups <= 0 ||
        qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(weight) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(input) != QCORE_BACKEND_CUDA ||
        qcore_tensor_backend(weight) != QCORE_BACKEND_CUDA ||
        qcore_tensor_backend(output) != QCORE_BACKEND_CUDA ||
        !qcore_tensor_is_contiguous(input) ||
        !qcore_tensor_is_contiguous(weight) ||
        !qcore_tensor_is_contiguous(output) ||
        qcore_tensor_rank(input) != 4 ||
        qcore_tensor_rank(weight) != 4 ||
        qcore_tensor_rank(output) != 4) {
        return false;
    }

    const auto device = qcore_tensor_device(input);
    if (device < 0 || qcore_tensor_device(weight) != device ||
        qcore_tensor_device(output) != device || !qcore_device_activate(device)) {
        return false;
    }

    const auto batches = qcore_tensor_extent(input, 0);
    const auto channels_in = qcore_tensor_extent(input, 1);
    const auto height = qcore_tensor_extent(input, 2);
    const auto width = qcore_tensor_extent(input, 3);
    const auto channels_out = qcore_tensor_extent(weight, 0);
    const auto weight_channels = qcore_tensor_extent(weight, 1);
    const auto kernel_height = qcore_tensor_extent(weight, 2);
    const auto kernel_width = qcore_tensor_extent(weight, 3);
    const auto output_batches = qcore_tensor_extent(output, 0);
    const auto output_channels = qcore_tensor_extent(output, 1);
    const auto output_height = qcore_tensor_extent(output, 2);
    const auto output_width = qcore_tensor_extent(output, 3);

    if (!positive(batches) || !positive(channels_in) ||
        !positive(height) || !positive(width) ||
        !positive(channels_out) || !positive(weight_channels) ||
        !positive(kernel_height) || !positive(kernel_width) ||
        !positive(output_height) || !positive(output_width) ||
        output_batches != batches || output_channels != channels_out ||
        channels_in % metadata.groups != 0 ||
        channels_out % metadata.groups != 0 ||
        weight_channels != channels_in / metadata.groups ||
        metadata.padding >
            (std::numeric_limits<long long>::max() - height) / 2 ||
        metadata.padding >
            (std::numeric_limits<long long>::max() - width) / 2) {
        return false;
    }

    const auto padded_height = height + metadata.padding * 2;
    const auto padded_width = width + metadata.padding * 2;
    if (padded_height < kernel_height || padded_width < kernel_width ||
        (padded_height - kernel_height) / metadata.stride + 1 != output_height ||
        (padded_width - kernel_width) / metadata.stride + 1 != output_width) {
        return false;
    }

    const auto int_max = static_cast<long long>(std::numeric_limits<int>::max());
    const long long dimensions[] = {
        batches, channels_in, channels_out, weight_channels,
        height, width, kernel_height, kernel_width,
        output_height, output_width,
        metadata.stride, metadata.padding, metadata.groups
    };
    for (const auto dimension : dimensions) {
        if (dimension > int_max) return false;
    }

    geometry = CudaConvGeometry{
        device,
        static_cast<int>(batches),
        static_cast<int>(channels_in),
        static_cast<int>(channels_out),
        static_cast<int>(weight_channels),
        static_cast<int>(height),
        static_cast<int>(width),
        static_cast<int>(kernel_height),
        static_cast<int>(kernel_width),
        static_cast<int>(output_height),
        static_cast<int>(output_width),
        static_cast<int>(metadata.stride),
        static_cast<int>(metadata.padding),
        static_cast<int>(metadata.groups)
    };
    return true;
}

struct CudnnScratch {
    void* token{};
    void* pointer{};

    CudnnScratch() = default;
    CudnnScratch(const CudnnScratch&) = delete;
    CudnnScratch& operator=(const CudnnScratch&) = delete;

    ~CudnnScratch() {
        if (token) qcore_device_buffer_release(token);
    }

    bool allocate(long long device, std::size_t bytes) {
        if (bytes == 0) return true;
        token = qcore_device_buffer_allocate(
            device, static_cast<std::uint64_t>(bytes));
        if (!token) return false;
        const auto handle = qcore_device_buffer_handle(token);
        if (handle == 0) return false;
        pointer = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(handle));
        return true;
    }
};

bool initialize_cudnn_convolution(
    CudnnApi& api,
    const CudaConvGeometry& geometry,
    CudnnSession& session,
    CudnnTensor& input_desc,
    CudnnTensor& output_desc,
    CudnnFilter& filter_desc,
    CudnnConvolution& convolution_desc) {
    if (api.create(&session.value) != cudnn_success ||
        api.create_tensor(&input_desc.value) != cudnn_success ||
        api.create_tensor(&output_desc.value) != cudnn_success ||
        api.create_filter(&filter_desc.value) != cudnn_success ||
        api.create_convolution(&convolution_desc.value) != cudnn_success) {
        return false;
    }

    if (api.set_tensor4d(
            input_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            geometry.batches, geometry.channels_in,
            geometry.height, geometry.width) != cudnn_success ||
        api.set_tensor4d(
            output_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            geometry.batches, geometry.channels_out,
            geometry.output_height, geometry.output_width) != cudnn_success ||
        api.set_filter4d(
            filter_desc.value, cudnn_data_float, cudnn_tensor_nchw,
            geometry.channels_out, geometry.weight_channels,
            geometry.kernel_height, geometry.kernel_width) != cudnn_success ||
        api.set_convolution2d(
            convolution_desc.value,
            geometry.padding, geometry.padding,
            geometry.stride, geometry.stride,
            1, 1, cudnn_cross_correlation, cudnn_data_float) != cudnn_success ||
        api.set_group_count(
            convolution_desc.value, geometry.groups) != cudnn_success) {
        return false;
    }
    return !api.set_math_type ||
        api.set_math_type(
            convolution_desc.value, cudnn_default_math) == cudnn_success;
}

int cuda_conv_forward_no_bias(
    const void* input,
    const void* weight,
    void* output,
    const Conv2dAutogradMetadata& metadata) {
    CudaConvGeometry geometry;
    if (!cuda_conv_geometry(input, weight, output, metadata, geometry))
        return 301;

    const auto* x = cuda_pointer_const(input);
    const auto* filter = cuda_pointer_const(weight);
    auto* y = cuda_pointer(output);
    if (!x || !filter || !y) return 302;

    auto& api = cudnn();
    if (!api.ready()) return 303;
    CudnnSession session{&api};
    CudnnTensor input_desc{&api};
    CudnnTensor output_desc{&api};
    CudnnFilter filter_desc{&api};
    CudnnConvolution convolution_desc{&api};
    if (!initialize_cudnn_convolution(
            api, geometry, session, input_desc, output_desc,
            filter_desc, convolution_desc)) {
        return 304;
    }

    const int algorithm = select_fast_cudnn_algorithm(
        api.forward_algorithms, session.value, input_desc.value,
        filter_desc.value, convolution_desc.value, output_desc.value,
        cudnn_conv_fwd_implicit_gemm);
    std::size_t workspace_size = 0;
    if (api.workspace_size(
            session.value, input_desc.value, filter_desc.value,
            convolution_desc.value, output_desc.value, algorithm,
            &workspace_size) != cudnn_success ||
        workspace_size > cudnn_fast_workspace_limit) {
        return 305;
    }

    CudnnScratch scratch;
    if (!scratch.allocate(geometry.device, workspace_size)) return 306;
    const float alpha = 1.0F;
    const float zero = 0.0F;
    return api.convolution_forward(
               session.value, &alpha, input_desc.value, x,
               filter_desc.value, filter, convolution_desc.value, algorithm,
               scratch.pointer, workspace_size, &zero, output_desc.value, y) ==
           cudnn_success ? 0 : 307;
}

int cuda_conv_backward_data_only(
    const void* weight,
    const void* output_gradient,
    void* input_gradient,
    const Conv2dAutogradMetadata& metadata) {
    CudaConvGeometry geometry;
    if (!cuda_conv_geometry(
            input_gradient, weight, output_gradient, metadata, geometry)) {
        return 311;
    }

    const auto* filter = cuda_pointer_const(weight);
    const auto* dy = cuda_pointer_const(output_gradient);
    auto* dx = cuda_pointer(input_gradient);
    if (!filter || !dy || !dx) return 312;

    auto& api = cudnn();
    if (!api.backward_ready()) return 313;
    CudnnSession session{&api};
    CudnnTensor input_desc{&api};
    CudnnTensor output_desc{&api};
    CudnnFilter filter_desc{&api};
    CudnnConvolution convolution_desc{&api};
    if (!initialize_cudnn_convolution(
            api, geometry, session, input_desc, output_desc,
            filter_desc, convolution_desc)) {
        return 314;
    }

    const int algorithm = select_fast_cudnn_backward_data_algorithm(
        api, session.value, filter_desc.value, output_desc.value,
        convolution_desc.value, input_desc.value);
    std::size_t workspace_size = 0;
    if (api.backward_data_workspace_size(
            session.value, filter_desc.value, output_desc.value,
            convolution_desc.value, input_desc.value, algorithm,
            &workspace_size) != cudnn_success ||
        workspace_size > cudnn_fast_workspace_limit) {
        return 315;
    }

    CudnnScratch scratch;
    if (!scratch.allocate(geometry.device, workspace_size)) return 316;
    const float alpha = 1.0F;
    const float zero = 0.0F;
    return api.convolution_backward_data(
               session.value, &alpha, filter_desc.value, filter,
               output_desc.value, dy, convolution_desc.value, algorithm,
               scratch.pointer, workspace_size, &zero, input_desc.value, dx) ==
           cudnn_success ? 0 : 317;
}

int cuda_conv_backward_filter_only(
    const void* input,
    const void* output_gradient,
    void* weight_gradient,
    const Conv2dAutogradMetadata& metadata) {
    CudaConvGeometry geometry;
    if (!cuda_conv_geometry(
            input, weight_gradient, output_gradient, metadata, geometry)) {
        return 321;
    }

    const auto* x = cuda_pointer_const(input);
    const auto* dy = cuda_pointer_const(output_gradient);
    auto* dw = cuda_pointer(weight_gradient);
    if (!x || !dy || !dw) return 322;

    auto& api = cudnn();
    if (!api.backward_ready()) return 323;
    CudnnSession session{&api};
    CudnnTensor input_desc{&api};
    CudnnTensor output_desc{&api};
    CudnnFilter filter_desc{&api};
    CudnnConvolution convolution_desc{&api};
    if (!initialize_cudnn_convolution(
            api, geometry, session, input_desc, output_desc,
            filter_desc, convolution_desc)) {
        return 324;
    }

    const int algorithm = select_fast_cudnn_backward_filter_algorithm(
        api, session.value, input_desc.value, output_desc.value,
        convolution_desc.value, filter_desc.value);
    std::size_t workspace_size = 0;
    if (api.backward_filter_workspace_size(
            session.value, input_desc.value, output_desc.value,
            convolution_desc.value, filter_desc.value, algorithm,
            &workspace_size) != cudnn_success ||
        workspace_size > cudnn_fast_workspace_limit) {
        return 325;
    }

    CudnnScratch scratch;
    if (!scratch.allocate(geometry.device, workspace_size)) return 326;
    const float alpha = 1.0F;
    const float zero = 0.0F;
    return api.convolution_backward_filter(
               session.value, &alpha, input_desc.value, x,
               output_desc.value, dy, convolution_desc.value, algorithm,
               scratch.pointer, workspace_size, &zero, filter_desc.value, dw) ==
           cudnn_success ? 0 : 327;
}

int cuda_bias_broadcast(
    const void* bias,
    void* output) {
    if (!bias || !output ||
        qcore_tensor_dtype(bias) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(bias) != QCORE_BACKEND_CUDA ||
        qcore_tensor_backend(output) != QCORE_BACKEND_CUDA ||
        !qcore_tensor_is_contiguous(bias) ||
        !qcore_tensor_is_contiguous(output) ||
        qcore_tensor_rank(bias) != 1 ||
        qcore_tensor_rank(output) != 4 ||
        qcore_tensor_device(bias) != qcore_tensor_device(output) ||
        qcore_tensor_extent(output, 1) != qcore_tensor_extent(bias, 0)) {
        return 331;
    }

    const auto n = qcore_tensor_extent(output, 0);
    const auto c = qcore_tensor_extent(output, 1);
    const auto h = qcore_tensor_extent(output, 2);
    const auto w = qcore_tensor_extent(output, 3);
    const auto int_max = static_cast<long long>(std::numeric_limits<int>::max());
    if (!positive(n) || !positive(c) || !positive(h) || !positive(w) ||
        n > int_max || c > int_max || h > int_max || w > int_max ||
        !qcore_device_activate(qcore_tensor_device(output))) {
        return 332;
    }

    const auto* source = cuda_pointer_const(bias);
    auto* destination = cuda_pointer(output);
    if (!source || !destination) return 333;

    auto& api = cudnn();
    if (!api.ready()) return 334;
    CudnnSession session{&api};
    CudnnTensor bias_desc{&api};
    CudnnTensor output_desc{&api};
    if (api.create(&session.value) != cudnn_success ||
        api.create_tensor(&bias_desc.value) != cudnn_success ||
        api.create_tensor(&output_desc.value) != cudnn_success ||
        api.set_tensor4d(
            bias_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            1, static_cast<int>(c), 1, 1) != cudnn_success ||
        api.set_tensor4d(
            output_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            static_cast<int>(n), static_cast<int>(c),
            static_cast<int>(h), static_cast<int>(w)) != cudnn_success) {
        return 335;
    }

    const float alpha = 1.0F;
    const float zero = 0.0F;
    return api.add_tensor(
               session.value, &alpha, bias_desc.value, source,
               &zero, output_desc.value, destination) == cudnn_success
        ? 0 : 336;
}

int conv2d_cuda_dx_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 2 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 340;
    }
    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    const int weight_status = cuda_conv_backward_filter_only(
        gradient_output, saved_tensors[1], gradient_inputs[0], metadata);
    if (weight_status != 0) return 341;
    const int output_status = cuda_conv_forward_no_bias(
        gradient_output, saved_tensors[0], gradient_inputs[1], metadata);
    return output_status == 0 ? 0 : 342;
}

int conv2d_cuda_dw_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 2 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 350;
    }
    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    const int input_status = cuda_conv_backward_data_only(
        gradient_output, saved_tensors[1], gradient_inputs[0], metadata);
    if (input_status != 0) return 351;
    const int output_status = cuda_conv_forward_no_bias(
        saved_tensors[0], gradient_output, gradient_inputs[1], metadata);
    return output_status == 0 ? 0 : 352;
}

int conv2d_cuda_db_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void*,
    std::uint64_t) {
    if (!saved_tensors || saved_tensor_count != 1 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 1) {
        return 360;
    }
    return cuda_bias_broadcast(gradient_output, gradient_inputs[0]);
}

int conv2d_cuda_backward_tracked_f32(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 3)
        return 370;

    const int first_order = conv2d_cuda_backward_f32(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count, metadata_raw, metadata_size);
    if (first_order != 0) return first_order;

    const void* dx_inputs[] = {
        differentiable_inputs[1], gradient_output
    };
    int status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[0], dx_inputs, 2, conv2d_cuda_dx_backward_f32,
        metadata_raw, metadata_size);
    if (status != 0) return 371;

    const void* dw_inputs[] = {
        differentiable_inputs[0], gradient_output
    };
    status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[1], dw_inputs, 2, conv2d_cuda_dw_backward_f32,
        metadata_raw, metadata_size);
    if (status != 0) return 372;

    const void* db_inputs[] = {gradient_output};
    status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[2], db_inputs, 1, conv2d_cuda_db_backward_f32,
        nullptr, 0);
    if (status != 0) return 373;
    return 0;
}

struct CpuConvGeometry {
    std::size_t batches{};
    std::size_t channels_in{};
    std::size_t channels_out{};
    std::size_t weight_channels{};
    std::size_t height{};
    std::size_t width{};
    std::size_t kernel_height{};
    std::size_t kernel_width{};
    std::size_t output_height{};
    std::size_t output_width{};
    std::size_t stride{};
    std::size_t padding{};
    std::size_t groups{};
};

// CPU direct-convolution kernels. They are organized around contiguous
// output (or input) rows so the innermost loops vectorize, while every
// gradient/output element still receives its terms in exactly the order of
// the original per-element loops (and with the same `acc += a * b`
// expressions), so results are bit-identical to them:
//   output[n, oc, oy, ox]: (input channel, ky, kx), then + bias
//   dInput[n, ic, iy, ix]: (output channel, oy, ox)
//   dWeight[oc, ic, ky, kx]: (n, oy, ox)
//   dBias[oc]: (n, oy, ox)

// Output positions o in [first, last) whose tap `tap` reads an in-bounds
// source index o * stride + tap - padding in [0, extent).
void conv_tap_range(
    std::size_t extent, std::size_t outputs, std::size_t stride,
    std::size_t padding, std::size_t tap,
    std::size_t& first, std::size_t& last) {
    first = padding > tap ? (padding - tap + stride - 1) / stride : 0;
    const std::size_t limit = extent - 1 + padding;
    last = limit < tap ? 0 : (limit - tap) / stride + 1;
    if (last > outputs) last = outputs;
    if (first > last) first = last;
}

struct ConvTapRanges {
    std::vector<std::size_t> first_y, last_y, first_x, last_x;

    explicit ConvTapRanges(const CpuConvGeometry& g)
        : first_y(g.kernel_height), last_y(g.kernel_height),
          first_x(g.kernel_width), last_x(g.kernel_width) {
        for (std::size_t ky = 0; ky < g.kernel_height; ++ky)
            conv_tap_range(g.height, g.output_height, g.stride, g.padding, ky,
                           first_y[ky], last_y[ky]);
        for (std::size_t kx = 0; kx < g.kernel_width; ++kx)
            conv_tap_range(g.width, g.output_width, g.stride, g.padding, kx,
                           first_x[kx], last_x[kx]);
    }
};

// Four channels of one group share every pass over a row: the inner loops
// load each value once and update four independent rows (`Stride` 0 reads
// the stride at run time; 1 and 2 let the compiler vectorize the loads).
template <std::size_t Stride>
void accumulate_rows4(
    float* __restrict row0, float* __restrict row1,
    float* __restrict row2, float* __restrict row3,
    const float* __restrict values, std::size_t count, std::size_t stride,
    float tap0, float tap1, float tap2, float tap3) {
    const std::size_t step = Stride != 0 ? Stride : stride;
    for (std::size_t index = 0; index < count; ++index) {
        const float value = values[index * step];
        row0[index] += value * tap0;
        row1[index] += value * tap1;
        row2[index] += value * tap2;
        row3[index] += value * tap3;
    }
}

template <std::size_t Stride>
void scatter_rows4(
    float* __restrict target0, float* __restrict target1,
    float* __restrict target2, float* __restrict target3,
    const float* __restrict gradients, std::size_t count, std::size_t stride,
    float tap0, float tap1, float tap2, float tap3) {
    const std::size_t step = Stride != 0 ? Stride : stride;
    for (std::size_t index = 0; index < count; ++index) {
        const float gradient = gradients[index];
        target0[index * step] += gradient * tap0;
        target1[index * step] += gradient * tap1;
        target2[index * step] += gradient * tap2;
        target3[index * step] += gradient * tap3;
    }
}

// Accumulates the convolution sums (without bias) into `output`.
void cpu_conv_forward_sums(
    const float* input,
    const float* weight,
    float* output,
    const CpuConvGeometry& g) {
    const ConvTapRanges ranges(g);
    const auto outputs_per_group = g.channels_out / g.groups;
    const auto taps = g.kernel_height * g.kernel_width;
    const auto input_plane = g.height * g.width;
    const auto output_plane = g.output_height * g.output_width;
    constexpr std::size_t block = 4;
    for (std::size_t batch = 0; batch < g.batches; ++batch) {
        for (std::size_t group = 0; group < g.groups; ++group) {
            for (std::size_t first_local = 0; first_local < outputs_per_group;
                 first_local += block) {
                const auto channels = std::min(block, outputs_per_group - first_local);
                const auto first_channel = group * outputs_per_group + first_local;
                float* planes[block] = {};
                for (std::size_t j = 0; j < channels; ++j) {
                    planes[j] = output +
                        (batch * g.channels_out + first_channel + j) * output_plane;
                    std::fill(planes[j], planes[j] + output_plane, 0.0F);
                }
                for (std::size_t output_y = 0; output_y < g.output_height;
                     ++output_y) {
                    float* rows[block] = {};
                    for (std::size_t j = 0; j < channels; ++j)
                        rows[j] = planes[j] + output_y * g.output_width;
                    for (std::size_t local_input = 0;
                         local_input < g.weight_channels; ++local_input) {
                        const auto input_channel =
                            group * g.weight_channels + local_input;
                        const float* source = input +
                            (batch * g.channels_in + input_channel) * input_plane;
                        const float* filters[block] = {};
                        for (std::size_t j = 0; j < channels; ++j) {
                            filters[j] = weight +
                                ((first_channel + j) * g.weight_channels + local_input) * taps;
                        }
                        for (std::size_t kernel_y = 0;
                             kernel_y < g.kernel_height; ++kernel_y) {
                            if (output_y < ranges.first_y[kernel_y] ||
                                output_y >= ranges.last_y[kernel_y])
                                continue;
                            const float* source_row = source +
                                (output_y * g.stride + kernel_y - g.padding) * g.width;
                            for (std::size_t kernel_x = 0;
                                 kernel_x < g.kernel_width; ++kernel_x) {
                                const auto tap = kernel_y * g.kernel_width + kernel_x;
                                const auto first = ranges.first_x[kernel_x];
                                const auto count = ranges.last_x[kernel_x] - first;
                                const float* values = source_row +
                                    (first * g.stride + kernel_x - g.padding);
                                if (channels == block) {
                                    float* r0 = rows[0] + first;
                                    float* r1 = rows[1] + first;
                                    float* r2 = rows[2] + first;
                                    float* r3 = rows[3] + first;
                                    const float t0 = filters[0][tap];
                                    const float t1 = filters[1][tap];
                                    const float t2 = filters[2][tap];
                                    const float t3 = filters[3][tap];
                                    if (g.stride == 1)
                                        accumulate_rows4<1>(r0, r1, r2, r3, values, count, 1, t0, t1, t2, t3);
                                    else if (g.stride == 2)
                                        accumulate_rows4<2>(r0, r1, r2, r3, values, count, 2, t0, t1, t2, t3);
                                    else
                                        accumulate_rows4<0>(r0, r1, r2, r3, values, count, g.stride, t0, t1, t2, t3);
                                } else {
                                    for (std::size_t j = 0; j < channels; ++j) {
                                        float* sums = rows[j] + first;
                                        const float tap_value = filters[j][tap];
                                        for (std::size_t index = 0; index < count; ++index)
                                            sums[index] += values[index * g.stride] * tap_value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// dInput = conv_transpose(gradient_output, weight). Four input channels of
// a group share every pass over a gradient row.
void cpu_conv_backward_data_rows(
    const float* weight,
    const float* output_gradient,
    float* input_gradient,
    const CpuConvGeometry& g) {
    const ConvTapRanges ranges(g);
    const auto outputs_per_group = g.channels_out / g.groups;
    const auto taps = g.kernel_height * g.kernel_width;
    const auto input_plane = g.height * g.width;
    const auto output_plane = g.output_height * g.output_width;
    constexpr std::size_t block = 4;
    std::fill(input_gradient,
              input_gradient + g.batches * g.channels_in * input_plane, 0.0F);
    for (std::size_t batch = 0; batch < g.batches; ++batch) {
        for (std::size_t group = 0; group < g.groups; ++group) {
            for (std::size_t first_local = 0; first_local < g.weight_channels;
                 first_local += block) {
                const auto channels = std::min(block, g.weight_channels - first_local);
                float* planes[block] = {};
                for (std::size_t j = 0; j < channels; ++j) {
                    planes[j] = input_gradient +
                        (batch * g.channels_in + group * g.weight_channels +
                         first_local + j) * input_plane;
                }
                for (std::size_t local_output = 0;
                     local_output < outputs_per_group; ++local_output) {
                    const auto output_channel =
                        group * outputs_per_group + local_output;
                    const float* gradient = output_gradient +
                        (batch * g.channels_out + output_channel) * output_plane;
                    const float* filters[block] = {};
                    for (std::size_t j = 0; j < channels; ++j) {
                        filters[j] = weight +
                            (output_channel * g.weight_channels + first_local + j) * taps;
                    }
                    // Descending taps visit each input element's output
                    // positions in ascending (oy, ox) order.
                    for (std::size_t kernel_y = g.kernel_height; kernel_y-- > 0;) {
                        for (std::size_t kernel_x = g.kernel_width; kernel_x-- > 0;) {
                            const auto tap = kernel_y * g.kernel_width + kernel_x;
                            const auto first_x = ranges.first_x[kernel_x];
                            const auto count = ranges.last_x[kernel_x] - first_x;
                            const auto offset = kernel_x + first_x * g.stride - g.padding;
                            for (std::size_t output_y = ranges.first_y[kernel_y];
                                 output_y < ranges.last_y[kernel_y]; ++output_y) {
                                const float* gradient_row = gradient +
                                    output_y * g.output_width + first_x;
                                const auto row =
                                    (output_y * g.stride + kernel_y - g.padding) * g.width + offset;
                                if (channels == block) {
                                    const float t0 = filters[0][tap];
                                    const float t1 = filters[1][tap];
                                    const float t2 = filters[2][tap];
                                    const float t3 = filters[3][tap];
                                    if (g.stride == 1)
                                        scatter_rows4<1>(planes[0] + row, planes[1] + row, planes[2] + row, planes[3] + row, gradient_row, count, 1, t0, t1, t2, t3);
                                    else if (g.stride == 2)
                                        scatter_rows4<2>(planes[0] + row, planes[1] + row, planes[2] + row, planes[3] + row, gradient_row, count, 2, t0, t1, t2, t3);
                                    else
                                        scatter_rows4<0>(planes[0] + row, planes[1] + row, planes[2] + row, planes[3] + row, gradient_row, count, g.stride, t0, t1, t2, t3);
                                } else {
                                    for (std::size_t j = 0; j < channels; ++j) {
                                        float* target = planes[j] + row;
                                        const float tap_value = filters[j][tap];
                                        for (std::size_t index = 0; index < count; ++index)
                                            target[index * g.stride] += gradient_row[index] * tap_value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Twelve dWeight chains at once: four output channels times the three taps
// of kernel row `kernel_y` (kernel width 3). Each chain still runs over
// (n, oy, ox) in ascending order: per row, the positions only some taps
// reach come first or last for that tap. Returns false (nothing written)
// when the taps share no position range.
bool filter_row3x4(
    const float* input,
    const float* output_gradient,
    float* weight_gradient,
    const CpuConvGeometry& g,
    const ConvTapRanges& ranges,
    std::size_t first_channel,
    std::size_t input_channel,
    std::size_t local_input,
    std::size_t kernel_y) {
    constexpr std::size_t block = 4;
    constexpr std::size_t width = 3;
    std::size_t common_first = 0;
    std::size_t common_last = g.output_width;
    for (std::size_t kernel_x = 0; kernel_x < width; ++kernel_x) {
        common_first = std::max(common_first, ranges.first_x[kernel_x]);
        common_last = std::min(common_last, ranges.last_x[kernel_x]);
    }
    if (common_first >= common_last) return false;
    const auto taps = g.kernel_height * width;
    const auto input_plane = g.height * g.width;
    const auto output_plane = g.output_height * g.output_width;
    float sums[block][width] = {};
    for (std::size_t batch = 0; batch < g.batches; ++batch) {
        const float* source = input +
            (batch * g.channels_in + input_channel) * input_plane;
        for (std::size_t output_y = ranges.first_y[kernel_y];
             output_y < ranges.last_y[kernel_y]; ++output_y) {
            const float* gradients[block];
            for (std::size_t j = 0; j < block; ++j) {
                gradients[j] = output_gradient +
                    (batch * g.channels_out + first_channel + j) * output_plane +
                    output_y * g.output_width;
            }
            const float* row = source +
                (output_y * g.stride + kernel_y - g.padding) * g.width;
            for (std::size_t kernel_x = 0; kernel_x < width; ++kernel_x) {
#pragma clang loop vectorize(disable) interleave(disable)
                for (std::size_t output_x = ranges.first_x[kernel_x];
                     output_x < common_first; ++output_x) {
                    const float value =
                        row[output_x * g.stride + kernel_x - g.padding];
                    for (std::size_t j = 0; j < block; ++j)
                        sums[j][kernel_x] += gradients[j][output_x] * value;
                }
            }
            float s00 = sums[0][0], s01 = sums[0][1], s02 = sums[0][2];
            float s10 = sums[1][0], s11 = sums[1][1], s12 = sums[1][2];
            float s20 = sums[2][0], s21 = sums[2][1], s22 = sums[2][2];
            float s30 = sums[3][0], s31 = sums[3][1], s32 = sums[3][2];
            const float* g0 = gradients[0];
            const float* g1 = gradients[1];
            const float* g2 = gradients[2];
            const float* g3 = gradients[3];
#pragma clang loop vectorize(disable) interleave(disable)
            for (std::size_t output_x = common_first; output_x < common_last;
                 ++output_x) {
                const float* window = row + output_x * g.stride - g.padding;
                const float v0 = window[0];
                const float v1 = window[1];
                const float v2 = window[2];
                const float a = g0[output_x];
                const float b = g1[output_x];
                const float c = g2[output_x];
                const float d = g3[output_x];
                s00 += a * v0; s01 += a * v1; s02 += a * v2;
                s10 += b * v0; s11 += b * v1; s12 += b * v2;
                s20 += c * v0; s21 += c * v1; s22 += c * v2;
                s30 += d * v0; s31 += d * v1; s32 += d * v2;
            }
            sums[0][0] = s00; sums[0][1] = s01; sums[0][2] = s02;
            sums[1][0] = s10; sums[1][1] = s11; sums[1][2] = s12;
            sums[2][0] = s20; sums[2][1] = s21; sums[2][2] = s22;
            sums[3][0] = s30; sums[3][1] = s31; sums[3][2] = s32;
            for (std::size_t kernel_x = 0; kernel_x < width; ++kernel_x) {
#pragma clang loop vectorize(disable) interleave(disable)
                for (std::size_t output_x = common_last;
                     output_x < ranges.last_x[kernel_x]; ++output_x) {
                    const float value =
                        row[output_x * g.stride + kernel_x - g.padding];
                    for (std::size_t j = 0; j < block; ++j)
                        sums[j][kernel_x] += gradients[j][output_x] * value;
                }
            }
        }
    }
    for (std::size_t j = 0; j < block; ++j) {
        for (std::size_t kernel_x = 0; kernel_x < width; ++kernel_x) {
            weight_gradient[((first_channel + j) * g.weight_channels +
                             local_input) * taps +
                            kernel_y * width + kernel_x] = sums[j][kernel_x];
        }
    }
    return true;
}

// dWeight = correlation of input with gradient_output. Each weight element
// is one multiply-add chain over (n, oy, ox); four output channels of a
// group advance their chains together so the latency-bound chains overlap
// while each one keeps its own order. The chains are not vectorized: a
// vectorized reduction would round the products separately.
void cpu_conv_backward_filter_rows(
    const float* input,
    const float* output_gradient,
    float* weight_gradient,
    const CpuConvGeometry& g) {
    const ConvTapRanges ranges(g);
    const auto outputs_per_group = g.channels_out / g.groups;
    const auto taps = g.kernel_height * g.kernel_width;
    const auto input_plane = g.height * g.width;
    const auto output_plane = g.output_height * g.output_width;
    constexpr std::size_t block = 4;
    for (std::size_t group = 0; group < g.groups; ++group) {
        for (std::size_t first_local = 0; first_local < outputs_per_group;
             first_local += block) {
            const auto count_channels =
                std::min(block, outputs_per_group - first_local);
            const auto first_channel = group * outputs_per_group + first_local;
            for (std::size_t local_input = 0;
                 local_input < g.weight_channels; ++local_input) {
                const auto input_channel =
                    group * g.weight_channels + local_input;
                for (std::size_t kernel_y = 0; kernel_y < g.kernel_height;
                     ++kernel_y) {
                    if (count_channels == block && g.kernel_width == 3 &&
                        filter_row3x4(input, output_gradient, weight_gradient,
                                      g, ranges, first_channel, input_channel,
                                      local_input, kernel_y)) {
                        continue;
                    }
                    for (std::size_t kernel_x = 0;
                         kernel_x < g.kernel_width; ++kernel_x) {
                        const auto tap = kernel_y * g.kernel_width + kernel_x;
                        const auto first_x = ranges.first_x[kernel_x];
                        const auto count = ranges.last_x[kernel_x] - first_x;
                        float sums[block] = {0.0F, 0.0F, 0.0F, 0.0F};
                        for (std::size_t batch = 0; batch < g.batches; ++batch) {
                            const float* source = input +
                                (batch * g.channels_in + input_channel) * input_plane;
                            const float* gradients[block] = {};
                            for (std::size_t j = 0; j < count_channels; ++j) {
                                gradients[j] = output_gradient +
                                    (batch * g.channels_out + first_channel + j) *
                                        output_plane;
                            }
                            for (std::size_t output_y = ranges.first_y[kernel_y];
                                 output_y < ranges.last_y[kernel_y]; ++output_y) {
                                const auto row = output_y * g.output_width + first_x;
                                const float* values = source +
                                    (output_y * g.stride + kernel_y - g.padding) * g.width +
                                    (first_x * g.stride + kernel_x - g.padding);
                                if (count_channels == block) {
                                    const float* g0 = gradients[0] + row;
                                    const float* g1 = gradients[1] + row;
                                    const float* g2 = gradients[2] + row;
                                    const float* g3 = gradients[3] + row;
                                    float s0 = sums[0];
                                    float s1 = sums[1];
                                    float s2 = sums[2];
                                    float s3 = sums[3];
#pragma clang loop vectorize(disable) interleave(disable)
                                    for (std::size_t index = 0; index < count; ++index) {
                                        const float value = values[index * g.stride];
                                        s0 += g0[index] * value;
                                        s1 += g1[index] * value;
                                        s2 += g2[index] * value;
                                        s3 += g3[index] * value;
                                    }
                                    sums[0] = s0;
                                    sums[1] = s1;
                                    sums[2] = s2;
                                    sums[3] = s3;
                                } else {
                                    for (std::size_t j = 0; j < count_channels; ++j) {
                                        const float* gradient_row = gradients[j] + row;
                                        float sum = sums[j];
#pragma clang loop vectorize(disable) interleave(disable)
                                        for (std::size_t index = 0; index < count; ++index)
                                            sum += gradient_row[index] * values[index * g.stride];
                                        sums[j] = sum;
                                    }
                                }
                            }
                        }
                        for (std::size_t j = 0; j < count_channels; ++j) {
                            weight_gradient[((first_channel + j) * g.weight_channels +
                                             local_input) * taps + tap] = sums[j];
                        }
                    }
                }
            }
        }
    }
}

void cpu_conv_backward_bias_rows(
    const float* output_gradient,
    float* bias_gradient,
    const CpuConvGeometry& g) {
    const auto output_plane = g.output_height * g.output_width;
    for (std::size_t channel = 0; channel < g.channels_out; ++channel) {
        float sum = 0.0F;
        for (std::size_t batch = 0; batch < g.batches; ++batch) {
            const float* gradient = output_gradient +
                (batch * g.channels_out + channel) * output_plane;
            for (std::size_t index = 0; index < output_plane; ++index)
                sum += gradient[index];
        }
        bias_gradient[channel] = sum;
    }
}

int conv2d_cpu_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !saved_tensors || saved_tensor_count != 3 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 3 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 201;
    }

    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    if (metadata.stride <= 0 || metadata.padding < 0 || metadata.groups <= 0)
        return 202;

    const void* input = saved_tensors[0];
    const void* weight = saved_tensors[1];
    const void* bias = saved_tensors[2];
    void* input_gradient = gradient_inputs[0];
    void* weight_gradient = gradient_inputs[1];
    void* bias_gradient = gradient_inputs[2];

    const void* read_tensors[] = {input, weight, bias, gradient_output};
    for (const void* tensor : read_tensors) {
        if (!tensor || qcore_tensor_dtype(tensor) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_backend(tensor) != QCORE_BACKEND_CPU ||
            !qcore_tensor_is_contiguous(tensor)) {
            return 203;
        }
    }
    void* write_tensors[] = {input_gradient, weight_gradient, bias_gradient};
    for (void* tensor : write_tensors) {
        if (!tensor || qcore_tensor_dtype(tensor) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_backend(tensor) != QCORE_BACKEND_CPU ||
            !qcore_tensor_is_contiguous(tensor)) {
            return 204;
        }
    }

    if (qcore_tensor_rank(input) != 4 || qcore_tensor_rank(weight) != 4 ||
        qcore_tensor_rank(bias) != 1 || qcore_tensor_rank(gradient_output) != 4 ||
        qcore_tensor_rank(input_gradient) != 4 ||
        qcore_tensor_rank(weight_gradient) != 4 ||
        qcore_tensor_rank(bias_gradient) != 1) {
        return 205;
    }

    const auto same_shape = [](const void* left, const void* right) {
        const auto rank = qcore_tensor_rank(left);
        if (rank != qcore_tensor_rank(right)) return false;
        for (std::uint64_t axis = 0; axis < rank; ++axis) {
            if (qcore_tensor_extent(left, axis) != qcore_tensor_extent(right, axis))
                return false;
        }
        return true;
    };
    if (!same_shape(input, input_gradient) ||
        !same_shape(weight, weight_gradient) ||
        !same_shape(bias, bias_gradient)) {
        return 206;
    }

    const auto batches_raw = qcore_tensor_extent(input, 0);
    const auto channels_in_raw = qcore_tensor_extent(input, 1);
    const auto height_raw = qcore_tensor_extent(input, 2);
    const auto width_raw = qcore_tensor_extent(input, 3);
    const auto channels_out_raw = qcore_tensor_extent(weight, 0);
    const auto weight_channels_raw = qcore_tensor_extent(weight, 1);
    const auto kernel_height_raw = qcore_tensor_extent(weight, 2);
    const auto kernel_width_raw = qcore_tensor_extent(weight, 3);
    const auto output_height_raw = qcore_tensor_extent(gradient_output, 2);
    const auto output_width_raw = qcore_tensor_extent(gradient_output, 3);
    if (!positive(batches_raw) || !positive(channels_in_raw) ||
        !positive(height_raw) || !positive(width_raw) ||
        !positive(channels_out_raw) || !positive(weight_channels_raw) ||
        !positive(kernel_height_raw) || !positive(kernel_width_raw) ||
        !positive(output_height_raw) || !positive(output_width_raw) ||
        channels_in_raw % metadata.groups != 0 ||
        channels_out_raw % metadata.groups != 0 ||
        weight_channels_raw != channels_in_raw / metadata.groups ||
        qcore_tensor_extent(bias, 0) != channels_out_raw ||
        qcore_tensor_extent(gradient_output, 0) != batches_raw ||
        qcore_tensor_extent(gradient_output, 1) != channels_out_raw) {
        return 207;
    }
    if (metadata.padding >
            (std::numeric_limits<long long>::max() - height_raw) / 2 ||
        metadata.padding >
            (std::numeric_limits<long long>::max() - width_raw) / 2) {
        return 208;
    }
    const auto padded_height_raw = height_raw + metadata.padding * 2;
    const auto padded_width_raw = width_raw + metadata.padding * 2;
    if (padded_height_raw < kernel_height_raw ||
        padded_width_raw < kernel_width_raw ||
        (padded_height_raw - kernel_height_raw) / metadata.stride + 1 != output_height_raw ||
        (padded_width_raw - kernel_width_raw) / metadata.stride + 1 != output_width_raw) {
        return 208;
    }

    const auto* source =
        static_cast<const float*>(qcore_tensor_cpu_data_const(input));
    const auto* weights =
        static_cast<const float*>(qcore_tensor_cpu_data_const(weight));
    const auto* output_gradient =
        static_cast<const float*>(qcore_tensor_cpu_data_const(gradient_output));
    auto* input_grad =
        static_cast<float*>(qcore_tensor_cpu_data(input_gradient));
    auto* weight_grad =
        static_cast<float*>(qcore_tensor_cpu_data(weight_gradient));
    auto* bias_grad =
        static_cast<float*>(qcore_tensor_cpu_data(bias_gradient));
    if (!source || !weights || !output_gradient ||
        !input_grad || !weight_grad || !bias_grad) {
        return 209;
    }

    const CpuConvGeometry geometry{
        static_cast<std::size_t>(batches_raw),
        static_cast<std::size_t>(channels_in_raw),
        static_cast<std::size_t>(channels_out_raw),
        static_cast<std::size_t>(weight_channels_raw),
        static_cast<std::size_t>(height_raw),
        static_cast<std::size_t>(width_raw),
        static_cast<std::size_t>(kernel_height_raw),
        static_cast<std::size_t>(kernel_width_raw),
        static_cast<std::size_t>(output_height_raw),
        static_cast<std::size_t>(output_width_raw),
        static_cast<std::size_t>(metadata.stride),
        static_cast<std::size_t>(metadata.padding),
        static_cast<std::size_t>(metadata.groups)
    };
    cpu_conv_backward_bias_rows(output_gradient, bias_grad, geometry);
    cpu_conv_backward_filter_rows(source, output_gradient, weight_grad, geometry);
    cpu_conv_backward_data_rows(weights, output_gradient, input_grad, geometry);
    return 0;
}


bool cpu_conv_geometry(
    const void* input,
    const void* weight,
    const void* output,
    const Conv2dAutogradMetadata& metadata,
    CpuConvGeometry& geometry) {
    if (!input || !weight || !output ||
        metadata.stride <= 0 || metadata.padding < 0 || metadata.groups <= 0 ||
        qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(weight) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(input) != QCORE_BACKEND_CPU ||
        qcore_tensor_backend(weight) != QCORE_BACKEND_CPU ||
        qcore_tensor_backend(output) != QCORE_BACKEND_CPU ||
        !qcore_tensor_is_contiguous(input) ||
        !qcore_tensor_is_contiguous(weight) ||
        !qcore_tensor_is_contiguous(output) ||
        qcore_tensor_rank(input) != 4 ||
        qcore_tensor_rank(weight) != 4 ||
        qcore_tensor_rank(output) != 4) {
        return false;
    }

    const auto batches = qcore_tensor_extent(input, 0);
    const auto channels_in = qcore_tensor_extent(input, 1);
    const auto height = qcore_tensor_extent(input, 2);
    const auto width = qcore_tensor_extent(input, 3);
    const auto channels_out = qcore_tensor_extent(weight, 0);
    const auto weight_channels = qcore_tensor_extent(weight, 1);
    const auto kernel_height = qcore_tensor_extent(weight, 2);
    const auto kernel_width = qcore_tensor_extent(weight, 3);
    const auto output_height = qcore_tensor_extent(output, 2);
    const auto output_width = qcore_tensor_extent(output, 3);
    if (!positive(batches) || !positive(channels_in) ||
        !positive(height) || !positive(width) ||
        !positive(channels_out) || !positive(weight_channels) ||
        !positive(kernel_height) || !positive(kernel_width) ||
        !positive(output_height) || !positive(output_width) ||
        channels_in % metadata.groups != 0 ||
        channels_out % metadata.groups != 0 ||
        weight_channels != channels_in / metadata.groups ||
        qcore_tensor_extent(output, 0) != batches ||
        qcore_tensor_extent(output, 1) != channels_out) {
        return false;
    }
    if (metadata.padding >
            (std::numeric_limits<long long>::max() - height) / 2 ||
        metadata.padding >
            (std::numeric_limits<long long>::max() - width) / 2) {
        return false;
    }
    const auto padded_height = height + metadata.padding * 2;
    const auto padded_width = width + metadata.padding * 2;
    if (padded_height < kernel_height || padded_width < kernel_width ||
        (padded_height - kernel_height) / metadata.stride + 1 != output_height ||
        (padded_width - kernel_width) / metadata.stride + 1 != output_width) {
        return false;
    }

    geometry = CpuConvGeometry{
        static_cast<std::size_t>(batches),
        static_cast<std::size_t>(channels_in),
        static_cast<std::size_t>(channels_out),
        static_cast<std::size_t>(weight_channels),
        static_cast<std::size_t>(height),
        static_cast<std::size_t>(width),
        static_cast<std::size_t>(kernel_height),
        static_cast<std::size_t>(kernel_width),
        static_cast<std::size_t>(output_height),
        static_cast<std::size_t>(output_width),
        static_cast<std::size_t>(metadata.stride),
        static_cast<std::size_t>(metadata.padding),
        static_cast<std::size_t>(metadata.groups)
    };
    return true;
}

void cpu_conv_forward_no_bias(
    const float* input,
    const float* weight,
    float* output,
    const CpuConvGeometry& g) {
    cpu_conv_forward_sums(input, weight, output, g);
}

void cpu_conv_backward_data(
    const float* weight,
    const float* output_gradient,
    float* input_gradient,
    const CpuConvGeometry& g) {
    cpu_conv_backward_data_rows(weight, output_gradient, input_gradient, g);
}

void cpu_conv_backward_filter(
    const float* input,
    const float* output_gradient,
    float* weight_gradient,
    const CpuConvGeometry& g) {
    cpu_conv_backward_filter_rows(input, output_gradient, weight_gradient, g);
}

int conv2d_cpu_dx_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 2 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 220;
    }
    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    const void* weight = saved_tensors[0];
    const void* first_gradient = saved_tensors[1];
    CpuConvGeometry geometry;
    if (!cpu_conv_geometry(
            gradient_output, weight, first_gradient, metadata, geometry)) {
        return 221;
    }
    if (!gradient_inputs[0] || !gradient_inputs[1] ||
        qcore_tensor_dtype(gradient_inputs[0]) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(gradient_inputs[1]) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(gradient_inputs[0]) != QCORE_BACKEND_CPU ||
        qcore_tensor_backend(gradient_inputs[1]) != QCORE_BACKEND_CPU ||
        !qcore_tensor_is_contiguous(gradient_inputs[0]) ||
        !qcore_tensor_is_contiguous(gradient_inputs[1])) {
        return 222;
    }
    const auto* h =
        static_cast<const float*>(qcore_tensor_cpu_data_const(gradient_output));
    const auto* w =
        static_cast<const float*>(qcore_tensor_cpu_data_const(weight));
    const auto* g =
        static_cast<const float*>(qcore_tensor_cpu_data_const(first_gradient));
    auto* dw = static_cast<float*>(qcore_tensor_cpu_data(gradient_inputs[0]));
    auto* dg = static_cast<float*>(qcore_tensor_cpu_data(gradient_inputs[1]));
    if (!h || !w || !g || !dw || !dg) return 223;
    cpu_conv_backward_filter(h, g, dw, geometry);
    cpu_conv_forward_no_bias(h, w, dg, geometry);
    return 0;
}

int conv2d_cpu_dw_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 2 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 230;
    }
    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    const void* input = saved_tensors[0];
    const void* first_gradient = saved_tensors[1];
    CpuConvGeometry geometry;
    if (!cpu_conv_geometry(
            input, gradient_output, first_gradient, metadata, geometry)) {
        return 231;
    }
    if (!gradient_inputs[0] || !gradient_inputs[1] ||
        qcore_tensor_dtype(gradient_inputs[0]) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(gradient_inputs[1]) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(gradient_inputs[0]) != QCORE_BACKEND_CPU ||
        qcore_tensor_backend(gradient_inputs[1]) != QCORE_BACKEND_CPU ||
        !qcore_tensor_is_contiguous(gradient_inputs[0]) ||
        !qcore_tensor_is_contiguous(gradient_inputs[1])) {
        return 232;
    }
    const auto* x =
        static_cast<const float*>(qcore_tensor_cpu_data_const(input));
    const auto* h =
        static_cast<const float*>(qcore_tensor_cpu_data_const(gradient_output));
    const auto* g =
        static_cast<const float*>(qcore_tensor_cpu_data_const(first_gradient));
    auto* dx = static_cast<float*>(qcore_tensor_cpu_data(gradient_inputs[0]));
    auto* dg = static_cast<float*>(qcore_tensor_cpu_data(gradient_inputs[1]));
    if (!x || !h || !g || !dx || !dg) return 233;
    cpu_conv_backward_data(h, g, dx, geometry);
    cpu_conv_forward_no_bias(x, h, dg, geometry);
    return 0;
}

int conv2d_cpu_db_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void*,
    std::uint64_t) {
    if (!saved_tensors || saved_tensor_count != 1 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 1) {
        return 240;
    }
    const void* first_gradient = saved_tensors[0];
    void* output = gradient_inputs[0];
    if (qcore_tensor_dtype(first_gradient) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(gradient_output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(first_gradient) != QCORE_BACKEND_CPU ||
        qcore_tensor_backend(gradient_output) != QCORE_BACKEND_CPU ||
        qcore_tensor_backend(output) != QCORE_BACKEND_CPU ||
        !qcore_tensor_is_contiguous(first_gradient) ||
        !qcore_tensor_is_contiguous(gradient_output) ||
        !qcore_tensor_is_contiguous(output) ||
        qcore_tensor_rank(first_gradient) != 4 ||
        qcore_tensor_rank(gradient_output) != 1 ||
        qcore_tensor_rank(output) != 4 ||
        qcore_tensor_extent(output, 0) != qcore_tensor_extent(first_gradient, 0) ||
        qcore_tensor_extent(output, 1) != qcore_tensor_extent(first_gradient, 1) ||
        qcore_tensor_extent(output, 2) != qcore_tensor_extent(first_gradient, 2) ||
        qcore_tensor_extent(output, 3) != qcore_tensor_extent(first_gradient, 3) ||
        qcore_tensor_extent(gradient_output, 0) !=
            qcore_tensor_extent(first_gradient, 1)) {
        return 241;
    }
    const auto* bias_gradient =
        static_cast<const float*>(qcore_tensor_cpu_data_const(gradient_output));
    auto* dg = static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!bias_gradient || !dg) return 242;
    const auto batches =
        static_cast<std::size_t>(qcore_tensor_extent(first_gradient, 0));
    const auto channels =
        static_cast<std::size_t>(qcore_tensor_extent(first_gradient, 1));
    const auto height =
        static_cast<std::size_t>(qcore_tensor_extent(first_gradient, 2));
    const auto width =
        static_cast<std::size_t>(qcore_tensor_extent(first_gradient, 3));
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
            for (std::size_t y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < width; ++x) {
                    const auto index =
                        ((batch * channels + channel) * height + y) * width + x;
                    dg[index] = bias_gradient[channel];
                }
            }
        }
    }
    return 0;
}

int conv2d_cpu_backward_tracked_f32(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 3)
        return 250;
    const int first_order = conv2d_cpu_backward_f32(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count,
        metadata_raw, metadata_size);
    if (first_order != 0) return first_order;

    const void* dx_inputs[] = {
        differentiable_inputs[1], gradient_output
    };
    int status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[0], dx_inputs, 2, conv2d_cpu_dx_backward_f32,
        metadata_raw, metadata_size);
    if (status != 0) return 251;

    const void* dw_inputs[] = {
        differentiable_inputs[0], gradient_output
    };
    status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[1], dw_inputs, 2, conv2d_cpu_dw_backward_f32,
        metadata_raw, metadata_size);
    if (status != 0) return 252;

    const void* db_inputs[] = {gradient_output};
    status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[2], db_inputs, 1, conv2d_cpu_db_backward_f32,
        nullptr, 0);
    if (status != 0) return 253;
    return 0;
}

#ifdef __APPLE__
// Metal Conv2D autograd mirrors the CPU callbacks above: one first-order node
// for input/weight/bias plus tracked second-order nodes whose kernels are the
// same NN-owned Metal convolutions.
int conv2d_metal_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !saved_tensors || saved_tensor_count != 3 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 3 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 401;
    }
    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    const int status = nn_metal_conv2d_backward(
        saved_tensors[0], saved_tensors[1], gradient_output,
        gradient_inputs[0], gradient_inputs[1], gradient_inputs[2],
        metadata.stride, metadata.padding, metadata.groups);
    return status == 0 ? 0 : 400 + status;
}

int conv2d_metal_dx_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 2 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 420;
    }
    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    const void* weight = saved_tensors[0];
    const void* first_gradient = saved_tensors[1];
    // dInput = conv_transpose(G, W) is bilinear: its adjoint with respect to W
    // correlates the upstream H with G, and with respect to G convolves H.
    int status = nn_metal_conv2d_backward_filter(
        gradient_output, first_gradient, gradient_inputs[0],
        metadata.stride, metadata.padding, metadata.groups);
    if (status != 0) return 421;
    status = nn_metal_conv2d_forward(
        gradient_output, weight, nullptr, gradient_inputs[1],
        metadata.stride, metadata.padding, metadata.groups);
    return status == 0 ? 0 : 422;
}

int conv2d_metal_dw_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 2 ||
        !metadata_raw || metadata_size != sizeof(Conv2dAutogradMetadata)) {
        return 430;
    }
    const auto& metadata =
        *static_cast<const Conv2dAutogradMetadata*>(metadata_raw);
    const void* input = saved_tensors[0];
    const void* first_gradient = saved_tensors[1];
    int status = nn_metal_conv2d_backward_data(
        gradient_output, first_gradient, gradient_inputs[0],
        metadata.stride, metadata.padding, metadata.groups);
    if (status != 0) return 431;
    status = nn_metal_conv2d_forward(
        input, gradient_output, nullptr, gradient_inputs[1],
        metadata.stride, metadata.padding, metadata.groups);
    return status == 0 ? 0 : 432;
}

int conv2d_metal_db_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void*,
    std::uint64_t) {
    if (!saved_tensors || saved_tensor_count != 1 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 1) {
        return 440;
    }
    return nn_metal_conv2d_bias_broadcast(
               gradient_output, gradient_inputs[0]) == 0 ? 0 : 441;
}

int conv2d_metal_backward_tracked_f32(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata_raw,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 3)
        return 450;
    const int first_order = conv2d_metal_backward_f32(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count,
        metadata_raw, metadata_size);
    if (first_order != 0) return first_order;

    const void* dx_inputs[] = {
        differentiable_inputs[1], gradient_output
    };
    int status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[0], dx_inputs, 2, conv2d_metal_dx_backward_f32,
        metadata_raw, metadata_size);
    if (status != 0) return 451;

    const void* dw_inputs[] = {
        differentiable_inputs[0], gradient_output
    };
    status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[1], dw_inputs, 2, conv2d_metal_dw_backward_f32,
        metadata_raw, metadata_size);
    if (status != 0) return 452;

    const void* db_inputs[] = {gradient_output};
    status = qcore_tensor_attach_custom_autograd(
        gradient_inputs[2], db_inputs, 1, conv2d_metal_db_backward_f32,
        nullptr, 0);
    if (status != 0) return 453;
    return 0;
}

// One custom autograd node for a Metal Conv2D output (no-op when no input is
// tracked).
int attach_conv2d_metal_autograd(
    const void* input, const void* weight, const void* bias, void* output,
    long long stride, long long padding, long long groups) {
    const Conv2dAutogradMetadata metadata{
        static_cast<std::int64_t>(stride),
        static_cast<std::int64_t>(padding),
        static_cast<std::int64_t>(groups)
    };
    const void* autograd_inputs[] = {input, weight, bias};
    return qcore_tensor_attach_custom_autograd_ex(
        output, autograd_inputs, 3, conv2d_metal_backward_f32,
        conv2d_metal_backward_tracked_f32, &metadata, sizeof(metadata));
}
#endif

bool tensor_cpu_dense_dtype(const void* tensor, int dtype) {
    return tensor &&
           qcore_tensor_dtype(tensor) == dtype &&
           qcore_tensor_backend(tensor) == QCORE_BACKEND_CPU &&
           qcore_tensor_is_contiguous(tensor) != 0;
}

bool same_tensor_shape(const void* left, const void* right) {
    if (!left || !right) return false;
    const auto rank = qcore_tensor_rank(left);
    if (qcore_tensor_rank(right) != rank) return false;
    for (std::uint64_t axis = 0; axis < rank; ++axis) {
        if (qcore_tensor_extent(left, axis) !=
            qcore_tensor_extent(right, axis)) {
            return false;
        }
    }
    return true;
}

template <typename T>
std::int32_t adam_step_cpu_impl(
    const void* parameter,
    const void* gradient,
    const void* first,
    const void* second,
    void* next_parameter,
    void* next_first,
    void* next_second,
    double beta1_raw,
    double beta2_raw,
    double one_minus_beta1_raw,
    double one_minus_beta2_raw,
    double adjusted_epsilon_raw,
    double step_scale_raw) {
    const int dtype = std::is_same_v<T, float>
        ? QCORE_DTYPE_FLOAT32
        : QCORE_DTYPE_FLOAT64;
    const void* reads[] = {parameter, gradient, first, second};
    for (const void* tensor : reads) {
        if (!tensor_cpu_dense_dtype(tensor, dtype) ||
            !same_tensor_shape(parameter, tensor)) {
            return 2;
        }
    }
    const void* writes[] = {next_parameter, next_first, next_second};
    for (const void* tensor : writes) {
        if (!tensor_cpu_dense_dtype(tensor, dtype) ||
            !same_tensor_shape(parameter, tensor)) {
            return 3;
        }
    }
    if (!std::isfinite(beta1_raw) || !std::isfinite(beta2_raw) ||
        !std::isfinite(one_minus_beta1_raw) ||
        !std::isfinite(one_minus_beta2_raw) ||
        !std::isfinite(adjusted_epsilon_raw) ||
        !std::isfinite(step_scale_raw) ||
        beta1_raw < 0.0 || beta1_raw >= 1.0 ||
        beta2_raw < 0.0 || beta2_raw >= 1.0 ||
        one_minus_beta1_raw < 0.0 || one_minus_beta2_raw < 0.0 ||
        adjusted_epsilon_raw <= 0.0) {
        return 4;
    }

    const auto count = qcore_tensor_element_count(parameter);
    const auto* parameter_data =
        static_cast<const T*>(qcore_tensor_cpu_data_const(parameter));
    const auto* gradient_data =
        static_cast<const T*>(qcore_tensor_cpu_data_const(gradient));
    const auto* first_data =
        static_cast<const T*>(qcore_tensor_cpu_data_const(first));
    const auto* second_data =
        static_cast<const T*>(qcore_tensor_cpu_data_const(second));
    auto* next_parameter_data =
        static_cast<T*>(qcore_tensor_cpu_data(next_parameter));
    auto* next_first_data =
        static_cast<T*>(qcore_tensor_cpu_data(next_first));
    auto* next_second_data =
        static_cast<T*>(qcore_tensor_cpu_data(next_second));
    if (!parameter_data || !gradient_data || !first_data || !second_data ||
        !next_parameter_data || !next_first_data || !next_second_data) {
        return 5;
    }

    const T beta1 = static_cast<T>(beta1_raw);
    const T beta2 = static_cast<T>(beta2_raw);
    const T one_minus_beta1 = static_cast<T>(one_minus_beta1_raw);
    const T one_minus_beta2 = static_cast<T>(one_minus_beta2_raw);
    const T adjusted_epsilon = static_cast<T>(adjusted_epsilon_raw);
    const T step_scale = static_cast<T>(step_scale_raw);

    for (std::uint64_t index = 0; index < count; ++index) {
        const T g = gradient_data[index];
        const T m =
            first_data[index] * beta1 + g * one_minus_beta1;
        const T v =
            second_data[index] * beta2 +
            g * g * one_minus_beta2;
        const T denominator =
            static_cast<T>(std::sqrt(v)) + adjusted_epsilon;
        next_first_data[index] = m;
        next_second_data[index] = v;
        next_parameter_data[index] =
            parameter_data[index] - (m / denominator) * step_scale;
    }
    return 0;
}

} // namespace


extern "C" std::int32_t nn_native_nccl_begin(long long count) {
    auto& state = nccl_state;
    if (state.group_open) return -1;
    state.reset_pending();
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        count <= 1 ||
        count > static_cast<long long>(std::numeric_limits<int>::max()) ||
        (qcore_execution_policy_get() == QCORE_EXECUTION_DETERMINISTIC) ||
        !nccl().ready()) {
        return 0;
    }
    state.expected = static_cast<int>(count);
    state.pending_devices.assign(static_cast<std::size_t>(count), -1);
    state.pending_backend_devices.assign(static_cast<std::size_t>(count), -1);
    state.registered.assign(static_cast<std::size_t>(count), 0);
    return 1;
}

extern "C" std::int32_t nn_native_nccl_register(
    const void* tensor, long long destination, long long rank_raw) {
    auto& state = nccl_state;
    if (!tensor || state.expected <= 1 || rank_raw < 0 ||
        rank_raw >= state.expected ||
        qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        qcore_tensor_backend(tensor) != QCORE_BACKEND_CUDA ||
        qcore_tensor_device(tensor) != destination ||
        !qcore_tensor_is_contiguous(tensor) ||
        qcore_tensor_device_handle_const(tensor) == 0) {
        return 0;
    }

    const auto dtype = qcore_tensor_dtype(tensor);
    if (dtype != QCORE_DTYPE_FLOAT32 && dtype != QCORE_DTYPE_FLOAT64)
        return 0;
    const auto backend_device = qcore_tensor_backend_device_index(tensor);
    if (backend_device < 0) return 0;

    const auto rank = static_cast<std::size_t>(rank_raw);
    for (std::size_t index = 0; index < state.registered.size(); ++index) {
        if (index != rank && state.registered[index] != 0 &&
            state.pending_backend_devices[index] == backend_device) {
            return 0;
        }
    }

    const auto count = qcore_tensor_element_count(tensor);
    if (state.dtype == 0) {
        state.dtype = dtype;
        state.count = count;
    } else if (state.dtype != dtype || state.count != count) {
        return 0;
    }

    state.pending_devices[rank] = destination;
    state.pending_backend_devices[rank] = backend_device;
    state.registered[rank] = 1;
    return 1;
}

extern "C" std::int32_t nn_native_nccl_commit() {
    return nccl_select_pending_communicators() ? 1 : 0;
}

extern "C" std::int32_t nn_native_nccl_cancel() {
    auto& state = nccl_state;
    if (state.group_open) {
        (void)nccl().group_end();
        state.group_open = false;
    }
    state.reset_pending();
    return 0;
}

extern "C" std::int32_t nn_native_nccl_group_start() {
    auto& state = nccl_state;
    if (state.group_open || state.selected == nccl_no_selection ||
        state.selected >= state.cache.size()) {
        return 1;
    }
    if (nccl().group_start() != nccl_success) return 2;
    state.group_open = true;
    return 0;
}

extern "C" std::int32_t nn_native_nccl_all_reduce(
    void* tensor, long long rank_raw) {
    auto& state = nccl_state;
    if (!state.group_open || !tensor || rank_raw < 0 ||
        rank_raw >= state.expected ||
        state.selected == nccl_no_selection ||
        state.selected >= state.cache.size()) {
        return 1;
    }

    const auto rank = static_cast<std::size_t>(rank_raw);
    auto& set = state.cache[state.selected];
    if (rank >= set.communicators.size() ||
        qcore_tensor_backend(tensor) != QCORE_BACKEND_CUDA ||
        qcore_tensor_device(tensor) != set.devices[rank] ||
        qcore_tensor_backend_device_index(tensor) != set.backend_devices[rank] ||
        qcore_tensor_dtype(tensor) != state.dtype ||
        qcore_tensor_element_count(tensor) != state.count ||
        !qcore_tensor_is_contiguous(tensor) ||
        !qcore_device_activate(set.devices[rank])) {
        return 2;
    }

    auto* pointer = cuda_pointer(tensor);
    if (!pointer) return 3;
    const int datatype =
        state.dtype == QCORE_DTYPE_FLOAT32 ? nccl_float32 : nccl_float64;
    return nccl().all_reduce(
               pointer, pointer, static_cast<std::size_t>(state.count),
               datatype, nccl_sum, set.communicators[rank], nullptr) ==
               nccl_success
        ? 0
        : 4;
}

extern "C" std::int32_t nn_native_nccl_group_end() {
    auto& state = nccl_state;
    if (!state.group_open) return 1;
    const auto status = nccl().group_end();
    state.group_open = false;
    state.reset_pending();
    return status == nccl_success ? 0 : 2;
}

extern "C" std::int32_t nn_native_cuda_device_f32(
    const void* input) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input ||
        qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(input) != QCORE_BACKEND_CUDA ||
        !qcore_tensor_is_contiguous(input)) {
        return -1;
    }
    const auto device = qcore_tensor_device(input);
    if (device < 0 ||
        device > static_cast<long long>(
            std::numeric_limits<std::int32_t>::max())) {
        return -1;
    }
    return static_cast<std::int32_t>(device);
}

std::int32_t conv2d_cuda_f32_impl(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride,
    long long padding,
    long long groups) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !weight || !bias || !output ||
        stride <= 0 || padding < 0 || groups <= 0) {
        return 1;
    }
    if (qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(weight) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(bias) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(input) != QCORE_BACKEND_CUDA ||
        qcore_tensor_backend(weight) != QCORE_BACKEND_CUDA ||
        qcore_tensor_backend(bias) != QCORE_BACKEND_CUDA ||
        qcore_tensor_backend(output) != QCORE_BACKEND_CUDA ||
        !qcore_tensor_is_contiguous(input) ||
        !qcore_tensor_is_contiguous(weight) ||
        !qcore_tensor_is_contiguous(bias) ||
        !qcore_tensor_is_contiguous(output) ||
        qcore_tensor_rank(input) != 4 ||
        qcore_tensor_rank(weight) != 4 ||
        qcore_tensor_rank(bias) != 1 ||
        qcore_tensor_rank(output) != 4) {
        return 2;
    }

    const auto device = qcore_tensor_device(input);
    if (device < 0 ||
        qcore_tensor_device(weight) != device ||
        qcore_tensor_device(bias) != device ||
        qcore_tensor_device(output) != device ||
        !qcore_device_activate(device)) {
        return 3;
    }

    const auto n = qcore_tensor_extent(input, 0);
    const auto c = qcore_tensor_extent(input, 1);
    const auto h = qcore_tensor_extent(input, 2);
    const auto w = qcore_tensor_extent(input, 3);
    const auto k = qcore_tensor_extent(weight, 0);
    const auto weight_c = qcore_tensor_extent(weight, 1);
    const auto r = qcore_tensor_extent(weight, 2);
    const auto s = qcore_tensor_extent(weight, 3);
    const auto out_n = qcore_tensor_extent(output, 0);
    const auto out_c = qcore_tensor_extent(output, 1);
    const auto out_h = qcore_tensor_extent(output, 2);
    const auto out_w = qcore_tensor_extent(output, 3);
    if (!positive(n) || !positive(c) || !positive(h) || !positive(w) ||
        !positive(k) || !positive(weight_c) || !positive(r) || !positive(s) ||
        out_n != n || out_c != k ||
        c % groups != 0 || k % groups != 0 ||
        weight_c != c / groups ||
        qcore_tensor_extent(bias, 0) != k ||
        n > std::numeric_limits<int>::max() ||
        c > std::numeric_limits<int>::max() ||
        h > std::numeric_limits<int>::max() ||
        w > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max() ||
        weight_c > std::numeric_limits<int>::max() ||
        r > std::numeric_limits<int>::max() ||
        s > std::numeric_limits<int>::max() ||
        out_h > std::numeric_limits<int>::max() ||
        out_w > std::numeric_limits<int>::max() ||
        stride > std::numeric_limits<int>::max() ||
        padding > std::numeric_limits<int>::max() ||
        groups > std::numeric_limits<int>::max()) {
        return 4;
    }

    const auto expected_h =
        (h + padding * 2 - r) / stride + 1;
    const auto expected_w =
        (w + padding * 2 - s) / stride + 1;
    if (h + padding * 2 < r || w + padding * 2 < s ||
        expected_h != out_h || expected_w != out_w) {
        return 5;
    }

    const auto* x = cuda_pointer_const(input);
    const auto* filter = cuda_pointer_const(weight);
    const auto* b = cuda_pointer_const(bias);
    auto* y = cuda_pointer(output);
    if (!x || !filter || !b || !y) return 6;

    auto& api = cudnn();
    if (!api.ready()) return 7;

    CudnnSession session{&api};
    CudnnTensor x_desc{&api};
    CudnnTensor y_desc{&api};
    CudnnTensor bias_desc{&api};
    CudnnFilter filter_desc{&api};
    CudnnConvolution conv_desc{&api};

    if (api.create(&session.value) != cudnn_success ||
        api.create_tensor(&x_desc.value) != cudnn_success ||
        api.create_tensor(&y_desc.value) != cudnn_success ||
        api.create_tensor(&bias_desc.value) != cudnn_success ||
        api.create_filter(&filter_desc.value) != cudnn_success ||
        api.create_convolution(&conv_desc.value) != cudnn_success) {
        return 8;
    }

    if (api.set_tensor4d(
            x_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            static_cast<int>(n), static_cast<int>(c),
            static_cast<int>(h), static_cast<int>(w)) != cudnn_success ||
        api.set_tensor4d(
            y_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            static_cast<int>(out_n), static_cast<int>(out_c),
            static_cast<int>(out_h), static_cast<int>(out_w)) != cudnn_success ||
        api.set_tensor4d(
            bias_desc.value, cudnn_tensor_nchw, cudnn_data_float,
            1, static_cast<int>(k), 1, 1) != cudnn_success ||
        api.set_filter4d(
            filter_desc.value, cudnn_data_float, cudnn_tensor_nchw,
            static_cast<int>(k), static_cast<int>(weight_c),
            static_cast<int>(r), static_cast<int>(s)) != cudnn_success ||
        api.set_convolution2d(
            conv_desc.value,
            static_cast<int>(padding), static_cast<int>(padding),
            static_cast<int>(stride), static_cast<int>(stride),
            1, 1, cudnn_cross_correlation, cudnn_data_float) != cudnn_success ||
        api.set_group_count(
            conv_desc.value, static_cast<int>(groups)) != cudnn_success) {
        return 9;
    }

    // NN owns backend policy. Deterministic mode deliberately stays on the
    // fixed deterministic IMPLICIT_GEMM path. Fast mode asks cuDNN's heuristic
    // for the fastest default-math algorithm within NN's bounded workspace
    // budget; optional/deprecated heuristic symbols are never required.
    if (api.set_math_type &&
        api.set_math_type(conv_desc.value, cudnn_default_math) != cudnn_success) {
        return 10;
    }

    const int forward_algorithm = select_fast_cudnn_algorithm(
        api.forward_algorithms, session.value, x_desc.value,
        filter_desc.value, conv_desc.value, y_desc.value,
        cudnn_conv_fwd_implicit_gemm);

    std::size_t workspace_size = 0;
    if (api.workspace_size(
            session.value, x_desc.value, filter_desc.value, conv_desc.value,
            y_desc.value, forward_algorithm,
            &workspace_size) != cudnn_success ||
        workspace_size > cudnn_fast_workspace_limit) {
        return 11;
    }

    void* workspace_token = nullptr;
    void* workspace = nullptr;
    struct ForwardScratchGuard {
        void*& token;
        ~ForwardScratchGuard() {
            if (token) qcore_device_buffer_release(token);
        }
    } scratch{workspace_token};
    if (workspace_size != 0) {
        workspace_token = qcore_device_buffer_allocate(
            device, static_cast<std::uint64_t>(workspace_size));
        if (!workspace_token) return 11;
        const auto handle = qcore_device_buffer_handle(workspace_token);
        if (handle == 0) return 11;
        workspace = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(handle));
    }

    const float alpha = 1.0F;
    const float zero = 0.0F;
    const float one = 1.0F;

    if (api.convolution_forward(
            session.value, &alpha, x_desc.value, x,
            filter_desc.value, filter, conv_desc.value,
            forward_algorithm, workspace, workspace_size,
            &zero, y_desc.value, y) != cudnn_success) {
        return 12;
    }
    if (api.add_tensor(
            session.value, &alpha, bias_desc.value, b,
            &one, y_desc.value, y) != cudnn_success) {
        return 13;
    }


    const Conv2dAutogradMetadata metadata{
        static_cast<std::int64_t>(stride),
        static_cast<std::int64_t>(padding),
        static_cast<std::int64_t>(groups)
    };
    const void* autograd_inputs[] = {input, weight, bias};
    const int autograd_status = qcore_tensor_attach_custom_autograd_ex(
        output, autograd_inputs, 3, conv2d_cuda_backward_f32,
        conv2d_cuda_backward_tracked_f32, &metadata, sizeof(metadata));
    if (autograd_status != 0) return 14;
    return 0;
}

extern "C" std::int32_t nn_native_conv2d_cuda_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride,
    long long padding,
    long long groups) {
    return conv2d_cuda_f32_impl(
        input, weight, bias, output,
        stride, padding, groups);
}

extern "C" std::int32_t nn_native_is_cpu_floating(
    const void* input) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input) {
        return 0;
    }
    const auto dtype = qcore_tensor_dtype(input);
    return (dtype == QCORE_DTYPE_FLOAT32 || dtype == QCORE_DTYPE_FLOAT64) &&
           qcore_tensor_backend(input) == QCORE_BACKEND_CPU &&
           qcore_tensor_is_contiguous(input) != 0 ? 1 : 0;
}

// Metal Adam batch: begin returns a batch token owned by the caller, then one
// encode per float32 Parameter into that batch, then one commit that runs
// every update in a single GPU round trip. The coefficient contract matches
// the CPU kernel; encode refuses anything else. A token <= 0 means no batch.
extern "C" std::int32_t nn_native_adam_metal_begin() {
#ifdef __APPLE__
    return nn_metal_adam_begin();
#else
    return 0;
#endif
}

extern "C" std::int32_t nn_native_adam_metal_encode(
    long long batch,
    const void* parameter,
    const void* gradient,
    const void* first,
    const void* second,
    void* next_parameter,
    void* next_first,
    void* next_second,
    double beta1,
    double beta2,
    double one_minus_beta1,
    double one_minus_beta2,
    double adjusted_epsilon,
    double step_scale) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        batch <= 0 || batch > std::numeric_limits<std::int32_t>::max() ||
        !parameter || qcore_tensor_backend(parameter) != QCORE_BACKEND_METAL ||
        qcore_tensor_dtype(parameter) != QCORE_DTYPE_FLOAT32) {
        return 1;
    }
    if (!std::isfinite(beta1) || !std::isfinite(beta2) ||
        !std::isfinite(one_minus_beta1) || !std::isfinite(one_minus_beta2) ||
        !std::isfinite(adjusted_epsilon) || !std::isfinite(step_scale) ||
        beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0 ||
        one_minus_beta1 < 0.0 || one_minus_beta2 < 0.0 ||
        adjusted_epsilon <= 0.0) {
        return 4;
    }
#ifdef __APPLE__
    return nn_metal_adam_encode(
               static_cast<std::int32_t>(batch),
               parameter, gradient, first, second,
               next_parameter, next_first, next_second,
               static_cast<float>(beta1), static_cast<float>(beta2),
               static_cast<float>(one_minus_beta1),
               static_cast<float>(one_minus_beta2),
               static_cast<float>(adjusted_epsilon),
               static_cast<float>(step_scale)) == 0 ? 0 : 6;
#else
    (void)gradient;
    (void)first;
    (void)second;
    (void)next_parameter;
    (void)next_first;
    (void)next_second;
    return 6;
#endif
}

// Commits batch `batch`, which must hold exactly `expected` encoded updates
// (the ones the caller will replace state from), and releases it. Anything
// else (unknown batch, a lost or extra update, a failed command) is an error
// and nothing may be replaced. `expected == 0` only releases the batch.
extern "C" std::int32_t nn_native_adam_metal_commit(
    long long batch,
    long long expected) {
    if (batch <= 0 || batch > std::numeric_limits<std::int32_t>::max() ||
        expected < 0) {
        return 1;
    }
#ifdef __APPLE__
    return nn_metal_adam_commit(
               static_cast<std::int32_t>(batch),
               static_cast<std::uint64_t>(expected)) == 0 ? 0 : 1;
#else
    return expected == 0 ? 0 : 1;
#endif
}

extern "C" std::int32_t nn_native_adam_step(
    const void* parameter,
    const void* gradient,
    const void* first,
    const void* second,
    void* next_parameter,
    void* next_first,
    void* next_second,
    double beta1,
    double beta2,
    double one_minus_beta1,
    double one_minus_beta2,
    double adjusted_epsilon,
    double step_scale) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !parameter) {
        return 1;
    }
    switch (qcore_tensor_dtype(parameter)) {
        case QCORE_DTYPE_FLOAT32:
            return adam_step_cpu_impl<float>(
                parameter, gradient, first, second,
                next_parameter, next_first, next_second,
                beta1, beta2, one_minus_beta1, one_minus_beta2,
                adjusted_epsilon, step_scale);
        case QCORE_DTYPE_FLOAT64:
            return adam_step_cpu_impl<double>(
                parameter, gradient, first, second,
                next_parameter, next_first, next_second,
                beta1, beta2, one_minus_beta1, one_minus_beta2,
                adjusted_epsilon, step_scale);
        default:
            return 2;
    }
}

static bool relu_training_tensor_f32(
    const void* tensor, std::uint64_t count) {
    return tensor &&
           qcore_tensor_dtype(tensor) == QCORE_DTYPE_FLOAT32 &&
           qcore_tensor_device(tensor) == -1 &&
           qcore_tensor_is_contiguous(tensor) != 0 &&
           qcore_tensor_element_count(tensor) == count;
}

static float relu_training_abs_derivative(float value) {
    return value < 0.0F ? -1.0F : (value > 0.0F ? 1.0F : 0.0F);
}

static int relu_training_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void*,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 1 ||
        !saved_tensors[0] || !gradient_output ||
        !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0] || metadata_size != 0) {
        return 1;
    }
    const auto count = qcore_tensor_element_count(saved_tensors[0]);
    if (!relu_training_tensor_f32(saved_tensors[0], count) ||
        !relu_training_tensor_f32(gradient_output, count) ||
        !relu_training_tensor_f32(gradient_inputs[0], count)) {
        return 2;
    }
    const auto* input = static_cast<const float*>(
        qcore_tensor_cpu_data_const(saved_tensors[0]));
    const auto* upstream = static_cast<const float*>(
        qcore_tensor_cpu_data_const(gradient_output));
    auto* gradient = static_cast<float*>(
        qcore_tensor_cpu_data(gradient_inputs[0]));
    if (!input || !upstream || !gradient) return 3;

    // Match the compositional nn.relu graph exactly:
    // s = upstream / 2; dx = s + s * d(abs)/dx.
    for (std::uint64_t index = 0; index < count; ++index) {
        const float half = upstream[index] * 0.5F;
        const float sign = relu_training_abs_derivative(input[index]);
        gradient[index] = half + half * sign;
    }
    return 0;
}

static int relu_training_second_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void*,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !saved_tensors[0] || !saved_tensors[1] ||
        !gradient_output || !gradient_inputs ||
        gradient_input_count != 2 ||
        !gradient_inputs[0] || !gradient_inputs[1] ||
        metadata_size != 0) {
        return 1;
    }
    const auto count = qcore_tensor_element_count(saved_tensors[0]);
    if (!relu_training_tensor_f32(saved_tensors[0], count) ||
        !relu_training_tensor_f32(saved_tensors[1], count) ||
        !relu_training_tensor_f32(gradient_output, count) ||
        !relu_training_tensor_f32(gradient_inputs[0], count) ||
        !relu_training_tensor_f32(gradient_inputs[1], count)) {
        return 2;
    }
    const auto* input = static_cast<const float*>(
        qcore_tensor_cpu_data_const(saved_tensors[0]));
    const auto* first_upstream = static_cast<const float*>(
        qcore_tensor_cpu_data_const(saved_tensors[1]));
    const auto* upstream = static_cast<const float*>(
        qcore_tensor_cpu_data_const(gradient_output));
    auto* gradient_input = static_cast<float*>(
        qcore_tensor_cpu_data(gradient_inputs[0]));
    auto* gradient_upstream = static_cast<float*>(
        qcore_tensor_cpu_data(gradient_inputs[1]));
    if (!input || !first_upstream || !upstream ||
        !gradient_input || !gradient_upstream) {
        return 3;
    }

    for (std::uint64_t index = 0; index < count; ++index) {
        const float sign = relu_training_abs_derivative(input[index]);
        const float first_half = first_upstream[index] * 0.5F;
        // Math abs defines its second derivative as zero. Preserve the
        // original multiplication order so inf/NaN behavior stays identical.
        gradient_input[index] =
            (upstream[index] * first_half) * 0.0F;
        const float direct = upstream[index] * 0.5F;
        const float via_abs = (upstream[index] * sign) * 0.5F;
        gradient_upstream[index] = direct + via_abs;
    }
    return 0;
}

static int relu_training_backward_tracked_f32(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 1 ||
        !differentiable_inputs[0] || !gradient_output) {
        return 1;
    }
    const int status = relu_training_backward_f32(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count, metadata, metadata_size);
    if (status != 0) return status;

    const void* parents[] = {differentiable_inputs[0], gradient_output};
    return qcore_tensor_attach_custom_autograd_ex(
        gradient_inputs[0], parents, 2,
        relu_training_second_backward_f32, nullptr,
        nullptr, 0);
}

extern "C" std::int32_t nn_native_relu_training_f32(
    const void* input, void* output) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !output) {
        return 1;
    }
    const auto count = qcore_tensor_element_count(input);
    if (!relu_training_tensor_f32(input, count) ||
        !relu_training_tensor_f32(output, count)) {
        return 2;
    }
    const auto* source = static_cast<const float*>(
        qcore_tensor_cpu_data_const(input));
    auto* destination = static_cast<float*>(
        qcore_tensor_cpu_data(output));
    if (!source || !destination) return 3;

    for (std::uint64_t index = 0; index < count; ++index) {
        const float value = source[index];
        destination[index] = (value + std::fabs(value)) * 0.5F;
    }

    const void* inputs[] = {input};
    const int attach_status = qcore_tensor_attach_custom_autograd_ex(
        output, inputs, 1,
        relu_training_backward_f32,
        relu_training_backward_tracked_f32,
        nullptr, 0);
    return attach_status == 0 ? 0 : 4;
}

// Fused NN activations for tracked and untracked tensors. Activation ids are
// NN policy shared with the CUDA epilogue: 1 = ReLU expression
// (x + |x|) / 2, 2 = GELU expression x / 2 * (1 + nn.tanh(sqrt(2/pi) *
// (x + 0.044715 x^3))) where nn.tanh clamps its argument to [-20, 20] and
// evaluates (e^(2z) - 1) / (e^(2z) + 1). The forward kernels repeat every
// rounding step of that source expression (no contraction), so CPU results
// are bit-identical to the compositional graph. Backward kernels apply the
// analytic first and second derivatives of the same expression.
namespace {

enum class NnActivation : std::uint8_t {
    Relu = 1,
    Gelu = 2,
};

bool activation_from_id(long long raw, NnActivation& activation) {
    if (raw != 1 && raw != 2) return false;
    activation = static_cast<NnActivation>(raw);
    return true;
}

[[gnu::always_inline]] inline float activation_abs_derivative(float value) {
    return value < 0.0F ? -1.0F : (value > 0.0F ? 1.0F : 0.0F);
}

float relu_expression(float value) {
#pragma clang fp contract(off)
    return (value + std::fabs(value)) * 0.5F;
}

float gelu_expression(float value) {
#pragma clang fp contract(off)
    const float square = value * value;
    const float cubic = square * value;
    const float inner = (value + cubic * 0.044715F) * 0.7978845608028654F;
    const float lifted = ((inner - 20.0F) + std::fabs(inner + 20.0F)) * 0.5F;
    const float limited = ((lifted + 20.0F) - std::fabs(lifted - 20.0F)) * 0.5F;
    const float exponent = std::exp(limited * 2.0F);
    const float hyperbolic = (exponent - 1.0F) / (exponent + 1.0F);
    return (value * 0.5F) * (1.0F + hyperbolic);
}

// exp for the GELU derivative kernels, whose argument is 2 * clamp(z) in
// [-40, 40] or NaN. A Cody-Waite reduction with the Cephes expf polynomial
// is accurate to about one ulp and, unlike a libm call, can be inlined into
// the element loops. Only derivatives use it; the forward keeps std::exp,
// the function behind math.exp, so forward values stay bit-identical.
[[gnu::always_inline]] inline float derivative_exp(float value) {
    const float bounded = std::fmin(std::fmax(value, -80.0F), 80.0F);
    const float count = std::floor(bounded * 1.44269504088896341F + 0.5F);
    const float reduced =
        (bounded - count * 0.693359375F) - count * -2.12194440e-4F;
    float polynomial = 1.9875691500e-4F;
    polynomial = polynomial * reduced + 1.3981999507e-3F;
    polynomial = polynomial * reduced + 8.3334519073e-3F;
    polynomial = polynomial * reduced + 4.1665795894e-2F;
    polynomial = polynomial * reduced + 1.6666665459e-1F;
    polynomial = polynomial * reduced + 5.0000001201e-1F;
    polynomial = polynomial * (reduced * reduced) + reduced + 1.0F;
    const std::int32_t bits = (static_cast<std::int32_t>(count) + 127) << 23;
    float scale = 0.0F;
    std::memcpy(&scale, &bits, sizeof(scale));
    const float result = polynomial * scale;
    return value == value ? result : value;
}

// Shared terms of the GELU derivatives at one input.
struct GeluTerms {
    float square;
    float hyperbolic;
    float slope;
    float clamp;
    float inner_slope;
    float inner_derivative;
};

// The clamp contributes the compositional graph's piecewise factor: 1
// inside, 0 outside and 1/2 on an exact boundary, because math.abs has
// derivative 0 at 0 and no curvature.
[[gnu::always_inline]] inline GeluTerms gelu_terms(float value) {
#pragma clang fp contract(off)
    GeluTerms terms{};
    terms.square = value * value;
    const float cubic = terms.square * value;
    const float inner = (value + cubic * 0.044715F) * 0.7978845608028654F;
    const float lower = inner + 20.0F;
    const float lifted = ((inner - 20.0F) + std::fabs(lower)) * 0.5F;
    const float upper = lifted - 20.0F;
    const float limited = ((lifted + 20.0F) - std::fabs(upper)) * 0.5F;
    const float exponent = derivative_exp(limited * 2.0F);
    const float denominator = exponent + 1.0F;
    terms.hyperbolic = (exponent - 1.0F) / denominator;
    terms.clamp =
        ((1.0F + activation_abs_derivative(lower)) * 0.5F) *
        ((1.0F - activation_abs_derivative(upper)) * 0.5F);
    // d tanh / d z = 4 e / (e + 1)^2, i.e. 1 - tanh^2 without cancellation.
    terms.slope = (4.0F * exponent) / (denominator * denominator);
    terms.inner_slope = terms.slope * terms.clamp;
    terms.inner_derivative =
        0.7978845608028654F * (1.0F + (3.0F * 0.044715F) * terms.square);
    return terms;
}

[[gnu::always_inline]] inline float gelu_first_from(
    float value, const GeluTerms& terms) {
#pragma clang fp contract(off)
    return 0.5F * (1.0F + terms.hyperbolic) +
           ((0.5F * value) * terms.inner_slope) * terms.inner_derivative;
}

// First derivative of gelu_expression, for the first-order backward. It
// returns by value and inlines, so the backward loop needs no call per
// element and no second derivative it would discard.
[[gnu::always_inline]] inline float gelu_first_derivative(float value) {
    return gelu_first_from(value, gelu_terms(value));
}

// First and second derivative of gelu_expression (second backward).
[[gnu::always_inline]] inline void gelu_derivatives(
    float value, float& first, float& second) {
#pragma clang fp contract(off)
    const GeluTerms terms = gelu_terms(value);
    const float hyperbolic = terms.hyperbolic;
    const float slope = terms.slope;
    const float clamp = terms.clamp;
    const float inner_slope = terms.inner_slope;
    const float inner_derivative = terms.inner_derivative;
    first = gelu_first_from(value, terms);
    // tanh'' times the square of d limited / dx = clamp * inner_derivative.
    // Squaring that product keeps the term exactly 0 outside the clamp
    // window, where inner_derivative^2 alone overflows float32 for
    // |x| >= 1.3128e10 (the forward stays finite up to |x| = 6.9815e12) and
    // clamp^2 * inf would be NaN. Inside the window clamp is 1 (1/2 on an
    // exact boundary: a power of two), so every rounding equals that of
    // (curvature * clamp^2) * (inner_derivative * inner_derivative).
    const float curvature = (-2.0F * hyperbolic) * slope;
    const float limited_derivative = clamp * inner_derivative;
    const float inner_curvature =
        0.7978845608028654F * ((6.0F * 0.044715F) * value);
    second = inner_slope * inner_derivative +
             (0.5F * value) *
                 (curvature * (limited_derivative * limited_derivative) +
                  inner_slope * inner_curvature);
}

bool activation_cpu_tensor(const void* tensor, std::uint64_t count) {
    return tensor &&
           qcore_tensor_dtype(tensor) == QCORE_DTYPE_FLOAT32 &&
           qcore_tensor_backend(tensor) == QCORE_BACKEND_CPU &&
           qcore_tensor_is_contiguous(tensor) != 0 &&
           qcore_tensor_element_count(tensor) == count;
}

int activation_backward_impl(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size,
    bool internal) {
    NnActivation activation{};
    if (!saved_tensors || saved_tensor_count != 1 || !saved_tensors[0] ||
        !gradient_output || !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0] || !metadata || metadata_size != 1 ||
        !activation_from_id(*static_cast<const std::uint8_t*>(metadata),
                            activation)) {
        return 501;
    }
    const void* input_tensor = saved_tensors[0];
    if (qcore_tensor_backend(input_tensor) == QCORE_BACKEND_METAL) {
#ifdef __APPLE__
        const auto launch = internal ? nn_metal_activation_backward_internal
                                     : nn_metal_activation_backward;
        return launch(input_tensor, gradient_output, gradient_inputs[0],
                      static_cast<long long>(activation)) == 0 ? 0 : 502;
#else
        return 502;
#endif
    }
    const auto count = qcore_tensor_element_count(input_tensor);
    if (!activation_cpu_tensor(input_tensor, count) ||
        !activation_cpu_tensor(gradient_output, count) ||
        !activation_cpu_tensor(gradient_inputs[0], count)) {
        return 503;
    }
    // Empty tensors have no host storage (NULL data) and nothing to write.
    if (count == 0) return 0;
    const auto* input = static_cast<const float*>(
        qcore_tensor_cpu_data_const(input_tensor));
    const auto* upstream = static_cast<const float*>(
        qcore_tensor_cpu_data_const(gradient_output));
    auto* gradient = static_cast<float*>(
        qcore_tensor_cpu_data(gradient_inputs[0]));
    if (!input || !upstream || !gradient) return 504;
    if (activation == NnActivation::Relu) {
        for (std::uint64_t index = 0; index < count; ++index) {
            // The compositional ReLU graph: s = g / 2; dx = s + s * abs'(x).
            const float half = upstream[index] * 0.5F;
            gradient[index] =
                half + half * activation_abs_derivative(input[index]);
        }
    } else {
        for (std::uint64_t index = 0; index < count; ++index)
            gradient[index] =
                upstream[index] * gelu_first_derivative(input[index]);
    }
    return 0;
}

int activation_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    return activation_backward_impl(
        saved_tensors, saved_tensor_count, gradient_output, gradient_inputs,
        gradient_input_count, metadata, metadata_size, false);
}

// For the activation of a Conv2D fusion target: its input is the fused
// call's pre-activation tensor, which only NN's Conv2D node reads, so the
// Metal kernel need not wait (nn_metal_activation_backward_internal).
int activation_backward_internal_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    return activation_backward_impl(
        saved_tensors, saved_tensor_count, gradient_output, gradient_inputs,
        gradient_input_count, metadata, metadata_size, true);
}

// Adjoint of dx = g * f'(x): gradient_inputs are (d/dx, d/dg).
int activation_second_backward_f32(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    NnActivation activation{};
    if (!saved_tensors || saved_tensor_count != 2 ||
        !saved_tensors[0] || !saved_tensors[1] || !gradient_output ||
        !gradient_inputs || gradient_input_count != 2 ||
        !gradient_inputs[0] || !gradient_inputs[1] ||
        !metadata || metadata_size != 1 ||
        !activation_from_id(*static_cast<const std::uint8_t*>(metadata),
                            activation)) {
        return 511;
    }
    const void* input_tensor = saved_tensors[0];
    if (qcore_tensor_backend(input_tensor) == QCORE_BACKEND_METAL) {
#ifdef __APPLE__
        return nn_metal_activation_second_backward(
                   input_tensor, saved_tensors[1], gradient_output,
                   gradient_inputs[0], gradient_inputs[1],
                   static_cast<long long>(activation)) == 0 ? 0 : 512;
#else
        return 512;
#endif
    }
    const auto count = qcore_tensor_element_count(input_tensor);
    if (!activation_cpu_tensor(input_tensor, count) ||
        !activation_cpu_tensor(saved_tensors[1], count) ||
        !activation_cpu_tensor(gradient_output, count) ||
        !activation_cpu_tensor(gradient_inputs[0], count) ||
        !activation_cpu_tensor(gradient_inputs[1], count)) {
        return 513;
    }
    if (count == 0) return 0;
    const auto* input = static_cast<const float*>(
        qcore_tensor_cpu_data_const(input_tensor));
    const auto* first_upstream = static_cast<const float*>(
        qcore_tensor_cpu_data_const(saved_tensors[1]));
    const auto* upstream = static_cast<const float*>(
        qcore_tensor_cpu_data_const(gradient_output));
    auto* gradient_input = static_cast<float*>(
        qcore_tensor_cpu_data(gradient_inputs[0]));
    auto* gradient_first = static_cast<float*>(
        qcore_tensor_cpu_data(gradient_inputs[1]));
    if (!input || !first_upstream || !upstream ||
        !gradient_input || !gradient_first) {
        return 514;
    }
    if (activation == NnActivation::Relu) {
        for (std::uint64_t index = 0; index < count; ++index) {
            // Same rules as relu_training_second_backward_f32: math.abs has
            // zero curvature, preserving inf/NaN propagation.
            const float sign = activation_abs_derivative(input[index]);
            const float first_half = first_upstream[index] * 0.5F;
            gradient_input[index] = (upstream[index] * first_half) * 0.0F;
            gradient_first[index] =
                upstream[index] * 0.5F + (upstream[index] * sign) * 0.5F;
        }
    } else {
        for (std::uint64_t index = 0; index < count; ++index) {
            float first = 0.0F;
            float second = 0.0F;
            gelu_derivatives(input[index], first, second);
            gradient_input[index] =
                (upstream[index] * first_upstream[index]) * second;
            gradient_first[index] = upstream[index] * first;
        }
    }
    return 0;
}

int activation_backward_tracked_f32(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 1 ||
        !differentiable_inputs[0] || !gradient_output) {
        return 521;
    }
    const int status = activation_backward_f32(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count, metadata, metadata_size);
    if (status != 0) return status;
    const void* parents[] = {differentiable_inputs[0], gradient_output};
    return qcore_tensor_attach_custom_autograd_ex(
               gradient_inputs[0], parents, 2,
               activation_second_backward_f32, nullptr,
               metadata, metadata_size) == 0 ? 0 : 522;
}

// One custom autograd node for an activation output (no-op when the input
// is untracked).
int attach_activation_autograd(
    const void* input, void* output, long long activation,
    bool internal_input = false) {
    const auto metadata = static_cast<std::uint8_t>(activation);
    const void* inputs[] = {input};
    return qcore_tensor_attach_custom_autograd_ex(
        output, inputs, 1,
        internal_input ? activation_backward_internal_f32
                       : activation_backward_f32,
        activation_backward_tracked_f32,
        &metadata, sizeof(metadata));
}

} // namespace

// Global average pooling y[n, c] = sum(x[n, c, :, :]) / (H * W) and its
// adjoint u[n, c, y, x] = g[n, c] / (H * W). Both are linear, so each one's
// backward is the other and every order of differentiation stays available
// (the compositional gather/sum_last graph also supported any order). The
// forward sums each plane left to right like math.sum_last and divides by
// the float32 plane size, bit-identical to the former composition on CPU.
namespace {

bool pooling_cpu_shapes(
    const void* planes, const void* rows,
    std::uint64_t& row_count, std::uint64_t& plane) {
    if (!planes || !rows ||
        qcore_tensor_dtype(planes) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(rows) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_backend(planes) != QCORE_BACKEND_CPU ||
        qcore_tensor_backend(rows) != QCORE_BACKEND_CPU ||
        !qcore_tensor_is_contiguous(planes) ||
        !qcore_tensor_is_contiguous(rows) ||
        qcore_tensor_rank(planes) != 4 || qcore_tensor_rank(rows) != 2 ||
        qcore_tensor_extent(planes, 0) != qcore_tensor_extent(rows, 0) ||
        qcore_tensor_extent(planes, 1) != qcore_tensor_extent(rows, 1)) {
        return false;
    }
    row_count = qcore_tensor_element_count(rows);
    const auto total = qcore_tensor_element_count(planes);
    if (row_count == 0 || total % row_count != 0) return false;
    plane = total / row_count;
    return plane != 0;
}

int pool_forward(const void* input, void* output) {
    if (qcore_tensor_backend(input) == QCORE_BACKEND_METAL) {
#ifdef __APPLE__
        return nn_metal_global_average_pool(input, output) == 0 ? 0 : 1;
#else
        return 1;
#endif
    }
    std::uint64_t rows = 0;
    std::uint64_t plane = 0;
    if (!pooling_cpu_shapes(input, output, rows, plane)) return 2;
    const auto* source =
        static_cast<const float*>(qcore_tensor_cpu_data_const(input));
    auto* destination = static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!source || !destination) return 3;
    const auto divisor = static_cast<float>(plane);
    for (std::uint64_t row = 0; row < rows; ++row) {
        const float* values = source + row * plane;
        float sum = values[0];
        for (std::uint64_t index = 1; index < plane; ++index)
            sum = sum + values[index];
        destination[row] = sum / divisor;
    }
    return 0;
}

int pool_adjoint(const void* gradient, void* output) {
    if (qcore_tensor_backend(gradient) == QCORE_BACKEND_METAL) {
#ifdef __APPLE__
        return nn_metal_global_average_unpool(gradient, output) == 0 ? 0 : 1;
#else
        return 1;
#endif
    }
    std::uint64_t rows = 0;
    std::uint64_t plane = 0;
    if (!pooling_cpu_shapes(output, gradient, rows, plane)) return 2;
    const auto* source =
        static_cast<const float*>(qcore_tensor_cpu_data_const(gradient));
    auto* destination = static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!source || !destination) return 3;
    const auto divisor = static_cast<float>(plane);
    for (std::uint64_t row = 0; row < rows; ++row) {
        const float value = source[row] / divisor;
        float* values = destination + row * plane;
        for (std::uint64_t index = 0; index < plane; ++index)
            values[index] = value;
    }
    return 0;
}

int pool_backward_f32(
    const void* const*, std::uint64_t saved_tensor_count,
    const void* gradient_output, void* const* gradient_inputs,
    std::uint64_t gradient_input_count, const void*, std::uint64_t) {
    if (saved_tensor_count != 0 || !gradient_output || !gradient_inputs ||
        gradient_input_count != 1 || !gradient_inputs[0])
        return 601;
    return pool_adjoint(gradient_output, gradient_inputs[0]) == 0 ? 0 : 602;
}

int unpool_backward_f32(
    const void* const*, std::uint64_t saved_tensor_count,
    const void* gradient_output, void* const* gradient_inputs,
    std::uint64_t gradient_input_count, const void*, std::uint64_t) {
    if (saved_tensor_count != 0 || !gradient_output || !gradient_inputs ||
        gradient_input_count != 1 || !gradient_inputs[0])
        return 611;
    return pool_forward(gradient_output, gradient_inputs[0]) == 0 ? 0 : 612;
}

int unpool_backward_tracked_f32(
    const void* const*, std::uint64_t,
    const void* const*, std::uint64_t,
    const void*, void* const*, std::uint64_t,
    const void*, std::uint64_t);

// Backward of the pool with provenance: the gradient is the adjoint of g,
// itself a differentiable pooling-adjoint node of g.
int pool_backward_tracked_f32(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 1)
        return 621;
    const int status = pool_backward_f32(
        saved_tensors, saved_tensor_count, gradient_output, gradient_inputs,
        gradient_input_count, metadata, metadata_size);
    if (status != 0) return status;
    const void* parents[] = {gradient_output};
    return qcore_tensor_attach_custom_autograd_with_saved_ex(
               gradient_inputs[0], parents, 1, nullptr, 0,
               unpool_backward_f32, unpool_backward_tracked_f32,
               nullptr, 0) == 0 ? 0 : 622;
}

int unpool_backward_tracked_f32(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 1)
        return 631;
    const int status = unpool_backward_f32(
        saved_tensors, saved_tensor_count, gradient_output, gradient_inputs,
        gradient_input_count, metadata, metadata_size);
    if (status != 0) return status;
    const void* parents[] = {gradient_output};
    return qcore_tensor_attach_custom_autograd_with_saved_ex(
               gradient_inputs[0], parents, 1, nullptr, 0,
               pool_backward_f32, pool_backward_tracked_f32,
               nullptr, 0) == 0 ? 0 : 632;
}

} // namespace

// Global average pooling of a dense float32 [N, C, H, W] CPU or Metal tensor
// into [N, C] with one custom autograd node. It saves nothing: the gradient
// does not depend on the input values.
extern "C" std::int32_t nn_native_global_average_pool2d_f32(
    const void* input,
    void* output) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !output ||
        qcore_tensor_backend(input) != qcore_tensor_backend(output) ||
        qcore_tensor_device(input) != qcore_tensor_device(output)) {
        return 1;
    }
    const auto backend = qcore_tensor_backend(input);
    if (backend != QCORE_BACKEND_CPU && backend != QCORE_BACKEND_METAL)
        return 2;
    if (pool_forward(input, output) != 0) return 3;
    const void* inputs[] = {input};
    return qcore_tensor_attach_custom_autograd_with_saved_ex(
               output, inputs, 1, nullptr, 0,
               pool_backward_f32, pool_backward_tracked_f32,
               nullptr, 0) == 0 ? 0 : 4;
}

// -1 for a float32 CPU tensor, the device index for a float32 Metal tensor,
// and -2 when NN has no fused elementwise/pooling kernel for the backend.
extern "C" std::int32_t nn_native_fused_device_f32(
    const void* input) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION || !input ||
        qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32) {
        return -2;
    }
    if (qcore_tensor_backend(input) == QCORE_BACKEND_CPU) return -1;
#ifdef __APPLE__
    const auto device = nn_metal_device_f32(input);
    if (device >= 0 &&
        device <= static_cast<long long>(
            std::numeric_limits<std::int32_t>::max())) {
        return static_cast<std::int32_t>(device);
    }
#endif
    return -2;
}

// output = activation(input) plus one custom autograd node (first order and
// a tracked callback for backward(track = true)). Inputs must be dense; the
// source-level caller never materializes a tracked tensor.
extern "C" std::int32_t nn_native_activation_f32(
    const void* input,
    void* output,
    long long activation_raw) {
    NnActivation activation{};
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !output || !activation_from_id(activation_raw, activation)) {
        return 1;
    }
    const auto backend = qcore_tensor_backend(input);
    if (qcore_tensor_backend(output) != backend ||
        qcore_tensor_device(output) != qcore_tensor_device(input) ||
        !same_tensor_shape(input, output)) {
        return 2;
    }
    if (backend == QCORE_BACKEND_METAL) {
#ifdef __APPLE__
        if (nn_metal_activation_forward(input, output, activation_raw) != 0)
            return 3;
#else
        return 3;
#endif
    } else if (backend == QCORE_BACKEND_CPU) {
        const auto count = qcore_tensor_element_count(input);
        if (!activation_cpu_tensor(input, count) ||
            !activation_cpu_tensor(output, count)) {
            return 4;
        }
        const auto* source =
            static_cast<const float*>(qcore_tensor_cpu_data_const(input));
        auto* destination = static_cast<float*>(qcore_tensor_cpu_data(output));
        if (count != 0 && (!source || !destination)) return 5;
        if (activation == NnActivation::Relu) {
            for (std::uint64_t index = 0; index < count; ++index)
                destination[index] = relu_expression(source[index]);
        } else {
            for (std::uint64_t index = 0; index < count; ++index)
                destination[index] = gelu_expression(source[index]);
        }
    } else {
        return 6;
    }

    return attach_activation_autograd(input, output, activation_raw) == 0
               ? 0 : 7;
}

// Metal Conv2D + activation in one GPU round trip. Produces the tensors and
// autograd nodes of nn_native_conv2d_f32 followed by nn_native_activation_f32:
// conv_output carries the Conv2D node and activation_output the activation
// node whose input is conv_output. conv_output must stay internal to the
// caller (the activation backward does not wait for its Metal kernel; the
// Conv2D backward that reads the result does). Other backends answer an
// error so the caller runs the two operations.
extern "C" std::int32_t nn_native_conv2d_activation_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* conv_output,
    void* activation_output,
    long long stride_raw,
    long long padding_raw,
    long long groups_raw,
    long long activation_raw) {
    NnActivation activation{};
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !weight || !bias || !conv_output || !activation_output ||
        !activation_from_id(activation_raw, activation) ||
        qcore_tensor_backend(input) != QCORE_BACKEND_METAL) {
        return 1;
    }
#ifdef __APPLE__
    const int status = nn_metal_conv2d_activation_forward(
        input, weight, bias, conv_output, activation_output,
        stride_raw, padding_raw, groups_raw, activation_raw);
    if (status != 0) return 100 + status;
    if (attach_conv2d_metal_autograd(
            input, weight, bias, conv_output,
            stride_raw, padding_raw, groups_raw) != 0)
        return 11;
    return attach_activation_autograd(
               conv_output, activation_output, activation_raw, true) == 0
               ? 0 : 12;
#else
    (void)stride_raw;
    (void)padding_raw;
    (void)groups_raw;
    return 2;
#endif
}

extern "C" std::int32_t nn_native_relu_reuse_f32(void* value) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION || !value)
        return 1;
    if (qcore_tensor_dtype(value) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_device(value) != -1 ||
        qcore_tensor_is_contiguous(value) == 0) {
        return 2;
    }
    // The compiler selects this target only for an owned last-use temporary.
    // Observe storage before Core's mutable COW boundary, then require writable
    // access to keep the same address. Success therefore proves that this path
    // actually reused the existing tensor buffer instead of allocating a copy.
    const auto* original_data =
        static_cast<const float*>(qcore_tensor_cpu_data_const(value));
    if (!original_data) return 3;
    auto* data = static_cast<float*>(qcore_tensor_cpu_data(value));
    if (!data) return 4;
    if (data != original_data) return 5;
    const auto count = qcore_tensor_element_count(value);
    for (std::uint64_t index = 0; index < count; ++index) {
        // Match nn.relu exactly, including infinities, signed zero, and
        // finite overflow behavior. Compiler replacements must preserve the
        // package expression rather than substitute an idealized max(0, x).
        const float input = data[index];
        data[index] = (input + std::fabs(input)) * 0.5F;
    }
    return 0;
}

// Test probe, not part of the Quidra-visible package API: how many NN Metal
// commands of one kernel family (DispatchKind in native/nn_metal.mm)
// completed in this process; 0 where NN has no Metal kernels. NN's native
// Metal kernels and its portable fallback produce the same values, so tests
// compare these counts around an operation to prove the native path ran.
extern "C" std::int64_t nn_native_metal_dispatch_count(std::int32_t kind) {
#ifdef __APPLE__
    const auto count = nn_metal_dispatch_count(static_cast<int>(kind));
    return count > static_cast<unsigned long long>(
                       std::numeric_limits<std::int64_t>::max())
               ? std::numeric_limits<std::int64_t>::max()
               : static_cast<std::int64_t>(count);
#else
    (void)kind;
    return 0;
#endif
}

extern "C" std::int32_t nn_native_metal_device_f32(
    const void* input) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION || !input)
        return -1;
#ifdef __APPLE__
    const auto device = nn_metal_device_f32(input);
    if (device >= 0 &&
        device <= static_cast<long long>(
            std::numeric_limits<std::int32_t>::max())) {
        return static_cast<std::int32_t>(device);
    }
#endif
    return -1;
}

extern "C" std::int32_t nn_native_is_cpu_f32(
    const void* input) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input) {
        return 0;
    }
    return qcore_tensor_dtype(input) == QCORE_DTYPE_FLOAT32 &&
           qcore_tensor_device(input) == -1 ? 1 : 0;
}

extern "C" std::int32_t nn_native_fc_relu_inference_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !weight || !bias || !output) {
        return 1;
    }
    const void* tensors[] = {input, weight, bias, output};
    for (const void* tensor : tensors) {
        if (qcore_tensor_dtype(tensor) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_device(tensor) != -1 ||
            qcore_tensor_is_contiguous(tensor) == 0) {
            return 2;
        }
    }

    const auto input_rank = qcore_tensor_rank(input);
    if (input_rank < 1 ||
        qcore_tensor_rank(weight) != 2 ||
        qcore_tensor_rank(bias) != 1 ||
        qcore_tensor_rank(output) != input_rank) {
        return 3;
    }

    const auto features_in_raw =
        qcore_tensor_extent(input, input_rank - 1);
    const auto features_out_raw = qcore_tensor_extent(weight, 0);
    if (features_in_raw <= 0 || features_out_raw <= 0 ||
        qcore_tensor_extent(weight, 1) != features_in_raw ||
        qcore_tensor_extent(bias, 0) != features_out_raw ||
        qcore_tensor_extent(output, input_rank - 1) != features_out_raw) {
        return 4;
    }
    for (std::uint64_t axis = 0; axis + 1 < input_rank; ++axis) {
        if (qcore_tensor_extent(output, axis) !=
            qcore_tensor_extent(input, axis)) {
            return 4;
        }
    }

    const auto features_in =
        static_cast<std::uint64_t>(features_in_raw);
    const auto features_out =
        static_cast<std::uint64_t>(features_out_raw);
    const auto input_count = qcore_tensor_element_count(input);
    if (input_count % features_in != 0) return 5;
    const auto rows = input_count / features_in;
    if (rows > std::numeric_limits<std::uint64_t>::max() / features_out ||
        qcore_tensor_element_count(weight) != features_in * features_out ||
        qcore_tensor_element_count(bias) != features_out ||
        qcore_tensor_element_count(output) != rows * features_out) {
        return 5;
    }

    const auto* x =
        static_cast<const float*>(qcore_tensor_cpu_data_const(input));
    const auto* w =
        static_cast<const float*>(qcore_tensor_cpu_data_const(weight));
    const auto* b =
        static_cast<const float*>(qcore_tensor_cpu_data_const(bias));
    auto* y = static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!x || !w || !b || !y) return 6;

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t out = 0; out < features_out; ++out) {
            float value = b[out];
            for (std::uint64_t inner = 0; inner < features_in; ++inner) {
                value += x[row * features_in + inner] *
                         w[out * features_in + inner];
            }
            y[row * features_out + out] =
                (value + std::fabs(value)) * 0.5F;
        }
    }
    return 0;
}

extern "C" std::int32_t nn_native_fc_gelu_inference_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !weight || !bias || !output) {
        return 1;
    }
    const void* tensors[] = {input, weight, bias, output};
    for (const void* tensor : tensors) {
        if (qcore_tensor_dtype(tensor) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_device(tensor) != -1 ||
            qcore_tensor_is_contiguous(tensor) == 0) {
            return 2;
        }
    }

    const auto input_rank = qcore_tensor_rank(input);
    if (input_rank < 1 ||
        qcore_tensor_rank(weight) != 2 ||
        qcore_tensor_rank(bias) != 1 ||
        qcore_tensor_rank(output) != input_rank) {
        return 3;
    }
    const auto features_in_raw =
        qcore_tensor_extent(input, input_rank - 1);
    const auto features_out_raw = qcore_tensor_extent(weight, 0);
    if (features_in_raw <= 0 || features_out_raw <= 0 ||
        qcore_tensor_extent(weight, 1) != features_in_raw ||
        qcore_tensor_extent(bias, 0) != features_out_raw ||
        qcore_tensor_extent(output, input_rank - 1) != features_out_raw) {
        return 4;
    }
    for (std::uint64_t axis = 0; axis + 1 < input_rank; ++axis) {
        if (qcore_tensor_extent(output, axis) !=
            qcore_tensor_extent(input, axis)) {
            return 4;
        }
    }

    const auto features_in =
        static_cast<std::uint64_t>(features_in_raw);
    const auto features_out =
        static_cast<std::uint64_t>(features_out_raw);
    const auto input_count = qcore_tensor_element_count(input);
    if (input_count % features_in != 0) return 5;
    const auto rows = input_count / features_in;
    if (rows > std::numeric_limits<std::uint64_t>::max() / features_out ||
        qcore_tensor_element_count(weight) != features_in * features_out ||
        qcore_tensor_element_count(bias) != features_out ||
        qcore_tensor_element_count(output) != rows * features_out) {
        return 5;
    }

    const auto* x =
        static_cast<const float*>(qcore_tensor_cpu_data_const(input));
    const auto* w =
        static_cast<const float*>(qcore_tensor_cpu_data_const(weight));
    const auto* b =
        static_cast<const float*>(qcore_tensor_cpu_data_const(bias));
    auto* y = static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!x || !w || !b || !y) return 6;

    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t out = 0; out < features_out; ++out) {
            float value = b[out];
            for (std::uint64_t inner_index = 0;
                 inner_index < features_in; ++inner_index) {
                value += x[row * features_in + inner_index] *
                         w[out * features_in + inner_index];
            }
            const float cubic = value * value * value;
            const float inner =
                (value + cubic * 0.044715F) * 0.7978845608028654F;
            const float lifted =
                (inner - 20.0F + std::fabs(inner + 20.0F)) * 0.5F;
            const float limited =
                (lifted + 20.0F - std::fabs(lifted - 20.0F)) * 0.5F;
            const float exponent = std::exp(limited * 2.0F);
            const float hyperbolic =
                (exponent - 1.0F) / (exponent + 1.0F);
            y[row * features_out + out] =
                value * 0.5F * (1.0F + hyperbolic);
        }
    }
    return 0;
}

std::int32_t conv2d_cpu_f32_impl(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride_raw,
    long long padding_raw,
    long long groups_raw,
    bool apply_relu,
    bool apply_gelu,
    bool attach_autograd) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !weight || !bias || !output) {
        return 1;
    }
    if (qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(weight) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(bias) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_device(input) != -1 ||
        qcore_tensor_device(weight) != -1 ||
        qcore_tensor_device(bias) != -1 ||
        qcore_tensor_device(output) != -1 ||
        qcore_tensor_rank(input) != 4 ||
        qcore_tensor_rank(weight) != 4 ||
        qcore_tensor_rank(bias) != 1 ||
        qcore_tensor_rank(output) != 4) {
        return 2;
    }
    if (!positive(stride_raw) || padding_raw < 0 || !positive(groups_raw)) {
        return 3;
    }

    const auto batches_raw = qcore_tensor_extent(input, 0);
    const auto channels_in_raw = qcore_tensor_extent(input, 1);
    const auto height_raw = qcore_tensor_extent(input, 2);
    const auto width_raw = qcore_tensor_extent(input, 3);
    const auto channels_out_raw = qcore_tensor_extent(weight, 0);
    const auto weight_channels_raw = qcore_tensor_extent(weight, 1);
    const auto kernel_height_raw = qcore_tensor_extent(weight, 2);
    const auto kernel_width_raw = qcore_tensor_extent(weight, 3);
    if (!positive(batches_raw) || !positive(channels_in_raw) ||
        !positive(height_raw) || !positive(width_raw) ||
        !positive(channels_out_raw) || !positive(weight_channels_raw) ||
        !positive(kernel_height_raw) || !positive(kernel_width_raw)) {
        return 4;
    }
    if (channels_in_raw % groups_raw != 0 ||
        channels_out_raw % groups_raw != 0 ||
        weight_channels_raw != channels_in_raw / groups_raw ||
        qcore_tensor_extent(bias, 0) != channels_out_raw) {
        return 5;
    }
    if (padding_raw >
        (std::numeric_limits<long long>::max() - height_raw) / 2 ||
        padding_raw >
        (std::numeric_limits<long long>::max() - width_raw) / 2) {
        return 6;
    }
    const auto padded_height_raw = height_raw + padding_raw * 2;
    const auto padded_width_raw = width_raw + padding_raw * 2;
    if (padded_height_raw < kernel_height_raw ||
        padded_width_raw < kernel_width_raw) {
        return 6;
    }
    const auto output_height_raw =
        (padded_height_raw - kernel_height_raw) / stride_raw + 1;
    const auto output_width_raw =
        (padded_width_raw - kernel_width_raw) / stride_raw + 1;
    if (!positive(output_height_raw) || !positive(output_width_raw) ||
        qcore_tensor_extent(output, 0) != batches_raw ||
        qcore_tensor_extent(output, 1) != channels_out_raw ||
        qcore_tensor_extent(output, 2) != output_height_raw ||
        qcore_tensor_extent(output, 3) != output_width_raw) {
        return 7;
    }

    const auto batches = static_cast<std::size_t>(batches_raw);
    const auto channels_in = static_cast<std::size_t>(channels_in_raw);
    const auto channels_out = static_cast<std::size_t>(channels_out_raw);
    const auto weight_channels = static_cast<std::size_t>(weight_channels_raw);
    const auto height = static_cast<std::size_t>(height_raw);
    const auto width = static_cast<std::size_t>(width_raw);
    const auto kernel_height = static_cast<std::size_t>(kernel_height_raw);
    const auto kernel_width = static_cast<std::size_t>(kernel_width_raw);
    const auto output_height = static_cast<std::size_t>(output_height_raw);
    const auto output_width = static_cast<std::size_t>(output_width_raw);
    const auto stride = static_cast<std::size_t>(stride_raw);
    const auto padding = static_cast<std::size_t>(padding_raw);
    const auto groups = static_cast<std::size_t>(groups_raw);
    const auto outputs_per_group = channels_out / groups;

    if (!multiply_ok(channels_in, height) ||
        !multiply_ok(channels_in * height, width) ||
        !multiply_ok(channels_out, weight_channels) ||
        !multiply_ok(channels_out * weight_channels, kernel_height) ||
        !multiply_ok(channels_out * weight_channels * kernel_height, kernel_width) ||
        !multiply_ok(channels_out, output_height) ||
        !multiply_ok(channels_out * output_height, output_width)) {
        return 8;
    }

    const auto* source =
        static_cast<const float*>(qcore_tensor_cpu_data_const(input));
    const auto* weights =
        static_cast<const float*>(qcore_tensor_cpu_data_const(weight));
    const auto* biases =
        static_cast<const float*>(qcore_tensor_cpu_data_const(bias));
    auto* destination =
        static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!source || !weights || !biases || !destination) return 9;

    const CpuConvGeometry geometry{
        batches, channels_in, channels_out, weight_channels,
        height, width, kernel_height, kernel_width,
        output_height, output_width, stride, padding, groups
    };
    cpu_conv_forward_sums(source, weights, destination, geometry);
    const auto output_plane = output_height * output_width;
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t output_channel = 0; output_channel < channels_out;
             ++output_channel) {
            float* values = destination +
                (batch * channels_out + output_channel) * output_plane;
            for (std::size_t index = 0; index < output_plane; ++index) {
                const float biased =
                    values[index] + biases[output_channel];
                if (apply_relu) {
                    values[index] =
                        (biased + std::fabs(biased)) * 0.5F;
                } else if (apply_gelu) {
                    const float cubic = biased * biased * biased;
                    const float inner =
                        (biased + cubic * 0.044715F) *
                        0.7978845608028654F;
                    const float lifted =
                        (inner - 20.0F + std::fabs(inner + 20.0F)) *
                        0.5F;
                    const float limited =
                        (lifted + 20.0F -
                         std::fabs(lifted - 20.0F)) * 0.5F;
                    const float exponent =
                        std::exp(limited * 2.0F);
                    const float hyperbolic =
                        (exponent - 1.0F) / (exponent + 1.0F);
                    values[index] =
                        biased * 0.5F * (1.0F + hyperbolic);
                } else {
                    values[index] = biased;
                }
            }
        }
    }

    if (!attach_autograd) return 0;

    const Conv2dAutogradMetadata metadata{
        static_cast<std::int64_t>(stride_raw),
        static_cast<std::int64_t>(padding_raw),
        static_cast<std::int64_t>(groups_raw)
    };
    const void* autograd_inputs[] = {input, weight, bias};
    const int autograd_status = qcore_tensor_attach_custom_autograd_ex(
        output, autograd_inputs, 3, conv2d_cpu_backward_f32,
        conv2d_cpu_backward_tracked_f32, &metadata, sizeof(metadata));
    if (autograd_status != 0) return 11;
    return 0;
}

extern "C" std::int32_t nn_native_conv2d_metal_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride_raw,
    long long padding_raw,
    long long groups_raw);

// Backend dispatcher for differentiable Conv2D: Metal tensors use the Metal
// kernels, everything else the CPU kernel (which rejects non-CPU tensors).
extern "C" std::int32_t nn_native_conv2d_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride_raw,
    long long padding_raw,
    long long groups_raw) {
    if (input && qcore_tensor_backend(input) == QCORE_BACKEND_METAL) {
        return nn_native_conv2d_metal_f32(
            input, weight, bias, output, stride_raw, padding_raw, groups_raw);
    }
    return conv2d_cpu_f32_impl(
        input, weight, bias, output,
        stride_raw, padding_raw, groups_raw,
        false, false, true);
}

// Metal Conv2D: NN's direct convolution plus one custom autograd node that
// saves input/weight/bias and the stride/padding/groups metadata. Tracked
// inputs must already be dense; the bridge never materializes them.
extern "C" std::int32_t nn_native_conv2d_metal_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride_raw,
    long long padding_raw,
    long long groups_raw) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !weight || !bias || !output) {
        return 1;
    }
#ifdef __APPLE__
    const int status = nn_metal_conv2d_forward(
        input, weight, bias, output, stride_raw, padding_raw, groups_raw);
    if (status != 0) return 100 + status;
    return attach_conv2d_metal_autograd(
               input, weight, bias, output,
               stride_raw, padding_raw, groups_raw) == 0 ? 0 : 11;
#else
    (void)stride_raw;
    (void)padding_raw;
    (void)groups_raw;
    return 2;
#endif
}

extern "C" std::int32_t nn_native_conv2d_relu_inference_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride_raw,
    long long padding_raw,
    long long groups_raw) {
    return conv2d_cpu_f32_impl(
        input, weight, bias, output,
        stride_raw, padding_raw, groups_raw,
        true, false, false);
}

extern "C" std::int32_t nn_native_conv2d_gelu_inference_f32(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    long long stride_raw,
    long long padding_raw,
    long long groups_raw) {
    return conv2d_cpu_f32_impl(
        input, weight, bias, output,
        stride_raw, padding_raw, groups_raw,
        false, true, false);
}

extern "C" std::int32_t nn_native_im2col_f32(
    const void* input,
    void* output,
    long long channels_raw,
    long long height_raw,
    long long width_raw,
    long long kernel_height_raw,
    long long kernel_width_raw,
    long long stride_raw,
    long long output_height_raw,
    long long output_width_raw) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !input || !output) {
        return 1;
    }
    if (qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
        qcore_tensor_device(input) != -1 || qcore_tensor_device(output) != -1 ||
        qcore_tensor_rank(input) != 4 || qcore_tensor_rank(output) != 2) {
        return 2;
    }
    if (!positive(channels_raw) || !positive(height_raw) || !positive(width_raw) ||
        !positive(kernel_height_raw) || !positive(kernel_width_raw) ||
        !positive(stride_raw) || !positive(output_height_raw) ||
        !positive(output_width_raw)) {
        return 3;
    }

    const auto batches_raw = qcore_tensor_extent(input, 0);
    if (!positive(batches_raw) ||
        qcore_tensor_extent(input, 1) != channels_raw ||
        qcore_tensor_extent(input, 2) != height_raw ||
        qcore_tensor_extent(input, 3) != width_raw) {
        return 4;
    }

    const auto batches = static_cast<std::size_t>(batches_raw);
    const auto channels = static_cast<std::size_t>(channels_raw);
    const auto height = static_cast<std::size_t>(height_raw);
    const auto width = static_cast<std::size_t>(width_raw);
    const auto kernel_height = static_cast<std::size_t>(kernel_height_raw);
    const auto kernel_width = static_cast<std::size_t>(kernel_width_raw);
    const auto stride = static_cast<std::size_t>(stride_raw);
    const auto output_height = static_cast<std::size_t>(output_height_raw);
    const auto output_width = static_cast<std::size_t>(output_width_raw);

    if (!multiply_ok(channels, kernel_height) ||
        !multiply_ok(channels * kernel_height, kernel_width) ||
        !multiply_ok(batches, output_height) ||
        !multiply_ok(batches * output_height, output_width)) {
        return 5;
    }
    const auto patch_width = channels * kernel_height * kernel_width;
    const auto patch_rows = batches * output_height * output_width;
    if (!multiply_ok(patch_rows, patch_width)) return 5;

    if (qcore_tensor_extent(output, 0) != static_cast<long long>(patch_rows) ||
        qcore_tensor_extent(output, 1) != static_cast<long long>(patch_width) ||
        qcore_tensor_element_count(output) != patch_rows * patch_width) {
        return 6;
    }

    const auto* source =
        static_cast<const float*>(qcore_tensor_cpu_data_const(input));
    auto* destination = static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!source || !destination) return 7;

    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t output_y = 0; output_y < output_height; ++output_y) {
            for (std::size_t output_x = 0; output_x < output_width; ++output_x) {
                const auto row =
                    (batch * output_height + output_y) * output_width + output_x;
                for (std::size_t channel = 0; channel < channels; ++channel) {
                    for (std::size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                        for (std::size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                            const auto source_y = output_y * stride + kernel_y;
                            const auto source_x = output_x * stride + kernel_x;
                            if (source_y >= height || source_x >= width) return 8;
                            const auto source_index =
                                ((batch * channels + channel) * height + source_y) *
                                    width + source_x;
                            const auto column =
                                (channel * kernel_height + kernel_y) *
                                    kernel_width + kernel_x;
                            destination[row * patch_width + column] =
                                source[source_index];
                        }
                    }
                }
            }
        }
    }
    return 0;
}
