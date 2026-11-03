// Exercise NN's real launch paths while substituting the external CUDA driver.
#include "../native/nn_cuda.cpp"

#include <cstdio>
#include <cstdlib>

namespace {
struct Tensor {
    unsigned long long count{8};
    int dtype{QCORE_DTYPE_FLOAT32};
};
std::uint64_t dispatches{};
int launch_result{};
int defer_result{};
bool activate{true};

int load_module(void** module, const void*) {
    *module = reinterpret_cast<void*>(1);
    return 0;
}
int load_function(void** function, void*, const char*) {
    *function = reinterpret_cast<void*>(2);
    return 0;
}
int enqueue(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
            unsigned, void*, void**, void**) {
    return launch_result;
}
void expect(const char* name, int observed_status, int wanted_status,
            std::uint64_t wanted_dispatches) {
    if (observed_status != wanted_status || dispatches != wanted_dispatches) {
        std::fprintf(stderr, "%s: status %d (want %d), dispatches %llu (want %llu)\n",
                     name, observed_status, wanted_status,
                     static_cast<unsigned long long>(dispatches),
                     static_cast<unsigned long long>(wanted_dispatches));
        std::exit(1);
    }
}
} // namespace

extern "C" unsigned int qcore_native_abi_version() { return QUIDRA_NATIVE_ABI_VERSION; }
extern "C" int qcore_tensor_backend(const void*) { return QCORE_BACKEND_CUDA; }
extern "C" int qcore_tensor_dtype(const void* tensor) { return static_cast<const Tensor*>(tensor)->dtype; }
extern "C" int qcore_tensor_is_contiguous(const void*) { return 1; }
extern "C" long long qcore_tensor_device(const void*) { return 0; }
extern "C" unsigned long long qcore_tensor_element_count(const void* tensor) { return static_cast<const Tensor*>(tensor)->count; }
extern "C" std::uint64_t qcore_tensor_device_handle_const(const void*) { return 4096; }
extern "C" std::uint64_t qcore_tensor_device_handle(void*) { return 4096; }
extern "C" std::uint64_t qcore_tensor_device_offset_bytes(const void*) { return 0; }
extern "C" int qcore_device_activate(long long) { return activate ? 1 : 0; }
extern "C" std::uint64_t qcore_device_queue_handle(long long) { return 0; }
extern "C" int qcore_device_status_slot(long long, std::uint64_t* buffer,
                                       std::uint64_t* offset, std::uint64_t* slot) {
    *buffer = 8192;
    *offset = 0;
    *slot = 1;
    return 0;
}
extern "C" void qcore_device_status_release(long long, std::uint64_t) {}
extern "C" int qcore_device_defer_status(long long, std::uint64_t, const char*) { return defer_result; }
extern "C" int qcore_device_status_wait(long long, std::uint64_t, std::uint32_t* status) { *status = 0; return 0; }
extern "C" int qcore_tensor_report_status(const void*, const char*, std::uint32_t) { return 0; }
extern "C" int qcore_op_register_flags(const char*, std::uint32_t) { return 0; }
extern "C" void qcore_counter_note_dispatches(std::uint64_t count) { dispatches += count; }

int main() {
    auto& driver = cuda();
    driver.ready = true;
    driver.module_load_data = load_module;
    driver.module_get_function = load_function;
    driver.launch_kernel = enqueue;
    Tensor tensor;

    expect("activation and validation enqueue", nn_cuda_activation_inplace_f32(&tensor, 1), 0, 2);
    dispatches = 0;
    launch_result = 1;
    expect("failed activation enqueue", nn_cuda_activation_inplace_f32(&tensor, 1), 5, 0);
    expect("failed validation enqueue", nn_cuda_validate_output(&tensor, "test", "test"), 6, 0);
    launch_result = 0;
    tensor.count = 0;
    expect("empty activation and validation", nn_cuda_activation_inplace_f32(&tensor, 1), 0, 0);
    tensor.count = 8;
    activate = false;
    expect("activation refusal", nn_cuda_activation_inplace_f32(&tensor, 1), 5, 0);
    activate = true;
    expect("unsupported activation", nn_cuda_activation_inplace_f32(&tensor, 3), 2, 0);
    defer_result = 1;
    expect("enqueued validation survives later refusal", nn_cuda_validate_output(&tensor, "test", "test"), 6, 1);
    std::puts("NN CUDA dispatch counters: ok");
}
