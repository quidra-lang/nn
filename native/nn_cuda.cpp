#include <quidra/native_extension.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

class DynamicLibrary {
public:
    DynamicLibrary() = default;
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    ~DynamicLibrary() {
#ifdef _WIN32
        if (handle_) FreeLibrary(static_cast<HMODULE>(handle_));
#else
        if (handle_) dlclose(handle_);
#endif
    }

    bool open(const char* name) {
#ifdef _WIN32
        handle_ = static_cast<void*>(LoadLibraryA(name));
#else
        handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        return handle_ != nullptr;
    }

    void* symbol(const char* name) const {
        if (!handle_) return nullptr;
#ifdef _WIN32
        return reinterpret_cast<void*>(
            GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
        return dlsym(handle_, name);
#endif
    }

private:
    void* handle_{};
};

template <typename T>
T load_symbol(const DynamicLibrary& library, const char* name) {
    return reinterpret_cast<T>(library.symbol(name));
}

struct CudaApi {
    using Module = void*;
    using Function = void*;
    using Stream = void*;
    using Result = int;

    DynamicLibrary library;
    Result (*init)(unsigned){};
    Result (*module_load_data)(Module*, const void*){};
    Result (*module_get_function)(Function*, Module, const char*){};
    Result (*launch_kernel)(
        Function, unsigned, unsigned, unsigned,
        unsigned, unsigned, unsigned, unsigned,
        Stream, void**, void**){};
    bool ready{};

    CudaApi() {
#ifdef _WIN32
        if (!library.open("nvcuda.dll")) return;
#else
        if (!library.open("libcuda.so.1") && !library.open("libcuda.so"))
            return;
#endif
        init = load_symbol<decltype(init)>(library, "cuInit");
        module_load_data =
            load_symbol<decltype(module_load_data)>(library, "cuModuleLoadData");
        module_get_function = load_symbol<decltype(module_get_function)>(
            library, "cuModuleGetFunction");
        launch_kernel =
            load_symbol<decltype(launch_kernel)>(library, "cuLaunchKernel");
        ready = init && module_load_data && module_get_function &&
                launch_kernel && init(0) == 0;
    }
};

CudaApi& cuda() {
    static CudaApi api;
    return api;
}

bool cuda_f32(const void* tensor) {
    return tensor && qcore_tensor_backend(tensor) == QCORE_BACKEND_CUDA &&
           qcore_tensor_dtype(tensor) == QCORE_DTYPE_FLOAT32 &&
           qcore_tensor_is_contiguous(tensor) != 0;
}

std::uint64_t pointer_mut(void* tensor) {
    const auto base = qcore_tensor_device_handle(tensor);
    const auto offset = qcore_tensor_device_offset_bytes(tensor);
    if (!base || offset > std::numeric_limits<std::uint64_t>::max() - base)
        return 0;
    return base + offset;
}

std::string activation_ptx(std::int32_t activation) {
    std::string body;
    if (activation == 1) {
        // Exactly mirrors nn.relu: (x + abs(x)) * 0.5.
        body =
            "  abs.f32 %f2,%f1;\n"
            "  add.rn.f32 %f3,%f1,%f2;\n"
            "  mov.f32 %f4,0f3f000000;\n"
            "  mul.rn.f32 %f5,%f3,%f4;\n"
            "  st.global.f32 [%rd9],%f5;\n";
    } else if (activation == 2) {
        // Exactly mirrors NN's tanh-approximation GELU expression. Math's
        // CUDA exp path uses the same ex2.approx/log2(e) implementation.
        body =
            "  mul.rn.f32 %f2,%f1,%f1;\n"
            "  mul.rn.f32 %f3,%f2,%f1;\n"
            "  mov.f32 %f4,0f3d372713;\n"
            "  mul.rn.f32 %f5,%f3,%f4;\n"
            "  add.rn.f32 %f6,%f1,%f5;\n"
            "  mov.f32 %f7,0f3f4c422a;\n"
            "  mul.rn.f32 %f8,%f6,%f7;\n"
            "  mov.f32 %f9,0f41a00000;\n"
            "  add.rn.f32 %f10,%f8,%f9;\n"
            "  abs.f32 %f11,%f10;\n"
            "  sub.rn.f32 %f12,%f8,%f9;\n"
            "  add.rn.f32 %f13,%f12,%f11;\n"
            "  mov.f32 %f14,0f3f000000;\n"
            "  mul.rn.f32 %f15,%f13,%f14;\n"
            "  add.rn.f32 %f16,%f15,%f9;\n"
            "  sub.rn.f32 %f17,%f15,%f9;\n"
            "  abs.f32 %f18,%f17;\n"
            "  sub.rn.f32 %f19,%f16,%f18;\n"
            "  mul.rn.f32 %f20,%f19,%f14;\n"
            "  mov.f32 %f21,0f40000000;\n"
            "  mul.rn.f32 %f22,%f20,%f21;\n"
            "  mov.f32 %f23,0f3fb8aa3b;\n"
            "  mul.rn.f32 %f24,%f22,%f23;\n"
            "  ex2.approx.f32 %f25,%f24;\n"
            "  mov.f32 %f26,0f3f800000;\n"
            "  sub.rn.f32 %f27,%f25,%f26;\n"
            "  add.rn.f32 %f28,%f25,%f26;\n"
            "  div.rn.f32 %f29,%f27,%f28;\n"
            "  add.rn.f32 %f30,%f26,%f29;\n"
            "  mul.rn.f32 %f31,%f1,%f14;\n"
            "  mul.rn.f32 %f32,%f31,%f30;\n"
            "  st.global.f32 [%rd9],%f32;\n";
    } else {
        return {};
    }

    return
        ".version 6.0\n"
        ".target sm_30\n"
        ".address_size 64\n\n"
        ".visible .entry nn_activation(\n"
        "  .param .u64 p_value,\n"
        "  .param .u64 p_count\n"
        ")\n"
        "{\n"
        "  .reg .pred %p<3>;\n"
        "  .reg .b32 %r<8>;\n"
        "  .reg .b64 %rd<16>;\n"
        "  .reg .f32 %f<40>;\n"
        "  mov.u32 %r1,%ctaid.x;\n"
        "  mov.u32 %r2,%ntid.x;\n"
        "  mov.u32 %r3,%tid.x;\n"
        "  mad.lo.s32 %r4,%r1,%r2,%r3;\n"
        "  cvt.u64.u32 %rd1,%r4;\n"
        "  ld.param.u64 %rd2,[p_value];\n"
        "  ld.param.u64 %rd3,[p_count];\n"
        "  setp.ge.u64 %p1,%rd1,%rd3;\n"
        "  @%p1 bra DONE;\n"
        "  mul.lo.u64 %rd8,%rd1,4;\n"
        "  add.u64 %rd9,%rd2,%rd8;\n"
        "  ld.global.f32 %f1,[%rd9];\n" +
        body +
        "DONE:\n"
        "  ret;\n"
        "}\n";
}

struct CachedKernel {
    CudaApi::Module module{};
    CudaApi::Function function{};
};

std::mutex cache_mutex;
std::map<std::string, CachedKernel> cache;

CudaApi::Function kernel(
    long long device, std::int32_t activation, const std::string& ptx) {
    auto& api = cuda();
    if (!api.ready || ptx.empty()) return nullptr;
    const auto key =
        std::to_string(device) + ":" + std::to_string(activation);
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (const auto found = cache.find(key); found != cache.end())
        return found->second.function;

    CudaApi::Module module = nullptr;
    CudaApi::Function function = nullptr;
    if (api.module_load_data(&module, ptx.c_str()) != 0 || !module ||
        api.module_get_function(
            &function, module, "nn_activation") != 0 ||
        !function) {
        return nullptr;
    }
    cache.emplace(key, CachedKernel{module, function});
    return function;
}

int launch_activation(
    long long device, std::int32_t activation,
    std::uint64_t pointer, std::uint64_t count) {
    if (count == 0) return 0;
    auto& api = cuda();
    if (!api.ready || !qcore_device_activate(device)) return 5;
    constexpr unsigned threads = 256;
    const auto blocks64 = (count + threads - 1) / threads;
    if (blocks64 > std::numeric_limits<unsigned>::max()) return 2;
    const auto ptx = activation_ptx(activation);
    const auto function = kernel(device, activation, ptx);
    if (!function) return 5;
    auto stream = reinterpret_cast<CudaApi::Stream>(
        static_cast<std::uintptr_t>(qcore_device_queue_handle(device)));
    void* arguments[] = {&pointer, &count};
    return api.launch_kernel(
               function, static_cast<unsigned>(blocks64), 1, 1,
               threads, 1, 1, 0, stream, arguments, nullptr) == 0
        ? 0
        : 5;
}

} // namespace

extern "C" std::int32_t nn_cuda_activation_inplace_f32(
    void* value, std::int32_t activation) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION || !value)
        return 1;
    if ((activation != 1 && activation != 2) || !cuda_f32(value)) return 2;
    const auto device = qcore_tensor_device(value);
    if (device < 0) return 2;

    // Memory-reuse callers rely on this being a true in-place epilogue, not a
    // copy-on-write allocation hidden behind mutable device access. Observe the
    // device address first, then require Core's writable borrow to preserve it.
    const auto original_base = qcore_tensor_device_handle_const(value);
    const auto original_offset = qcore_tensor_device_offset_bytes(value);
    if (!original_base ||
        original_offset >
            std::numeric_limits<std::uint64_t>::max() - original_base) {
        return 3;
    }
    const auto original_pointer = original_base + original_offset;
    const auto pointer = pointer_mut(value);
    if (!pointer) return 3;
    if (pointer != original_pointer) return 4;

    return launch_activation(
        device, activation, pointer, qcore_tensor_element_count(value));
}
