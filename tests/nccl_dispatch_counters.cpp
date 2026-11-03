// Exercise NN's group ownership and enqueue paths without a GPU or NCCL.
#include "../native/nn_native.cpp"

namespace {
struct Tensor {
    long long device{};
    unsigned long long count{8};
};
struct Tensors { Tensor* values[2]; };
std::uint64_t dispatches{};
bool counters_enabled{true};
int start_result{}, enqueue_result{}, end_result{};

int start_group() { return start_result; }
int end_group() { return end_result; }
int unique_id(NcclUniqueId*) { return 0; }
int initialize_rank(void** communicator, int, NcclUniqueId, int rank) {
    *communicator = reinterpret_cast<void*>(static_cast<std::uintptr_t>(rank + 1));
    return 0;
}
int finish_communicator(void*) { return 0; }
int enqueue_rank(const void*, void*, std::size_t, int, int, void*, void*) {
    return enqueue_result;
}
void prepare_group(unsigned long long elements = 8) {
    nccl_state.reset_pending();
    nccl_state.cache.clear();
    nccl_state.cache.push_back({{0, 1}, {0, 1},
                              {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2)}});
    nccl_state.expected = 2;
    nccl_state.dtype = QCORE_DTYPE_FLOAT32;
    nccl_state.count = elements;
    nccl_state.selected = 0;
    dispatches = 0;
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

extern "C" int qcore_tensor_backend(const void*) { return QCORE_BACKEND_CUDA; }
extern "C" unsigned int qcore_native_abi_version() { return QUIDRA_NATIVE_ABI_VERSION; }
extern "C" int qcore_execution_policy_get() { return QCORE_EXECUTION_FAST; }
extern "C" int qcore_tensor_dtype(const void*) { return QCORE_DTYPE_FLOAT32; }
extern "C" long long qcore_tensor_device(const void* tensor) { return static_cast<const Tensor*>(tensor)->device; }
extern "C" long long qcore_tensor_backend_device_index(const void* tensor) { return static_cast<const Tensor*>(tensor)->device; }
extern "C" unsigned long long qcore_tensor_element_count(const void* tensor) { return static_cast<const Tensor*>(tensor)->count; }
extern "C" int qcore_tensor_is_contiguous(const void*) { return 1; }
extern "C" unsigned long long qcore_tensor_rank(const void*) { return 1; }
extern "C" long long qcore_tensor_extent(const void* tensor, unsigned long long) { return static_cast<long long>(static_cast<const Tensor*>(tensor)->count); }
extern "C" std::uint64_t qcore_tensor_array_count(const void*) { return 2; }
extern "C" void* qcore_tensor_array_element(void* tensors, std::uint64_t index) { return static_cast<Tensors*>(tensors)->values[index]; }
extern "C" std::uint64_t qcore_tensor_device_handle_const(const void*) { return 4096; }
extern "C" std::uint64_t qcore_tensor_device_handle(void*) { return 4096; }
extern "C" std::uint64_t qcore_tensor_device_offset_bytes(const void*) { return 0; }
extern "C" int qcore_device_activate(long long) { return 1; }
extern "C" int qcore_op_register_flags(const char*, std::uint32_t) { return 0; }
extern "C" int qcore_counters_enabled() { return counters_enabled ? 1 : 0; }
extern "C" [[noreturn]] void qcore_runtime_immediate_failure_exit() { std::_Exit(101); }
extern "C" int nn_cuda_validate_output(const void*, const char*, const char*) { return 0; }
extern "C" void qcore_counter_note_dispatches(std::uint64_t count) {
    if (counters_enabled) dispatches += count;
}

int main() {
    nccl().group_start = start_group;
    nccl().group_end = end_group;
    nccl().all_reduce = enqueue_rank;
    nccl().get_unique_id = unique_id;
    nccl().comm_init_rank = initialize_rank;
    nccl().comm_destroy = finish_communicator;
    nccl().comm_abort = finish_communicator;
    Tensor first{0}, second{1};

    prepare_group();
    expect("group start", nn_native_nccl_group_start(), 0, 0);
    expect("first accepted rank remains pending", nn_native_nccl_all_reduce(&first, 0), 0, 0);
    expect("second accepted rank remains pending", nn_native_nccl_all_reduce(&second, 1), 0, 0);
    expect("successful group publishes both ranks", nn_native_nccl_group_end(), 0, 2);

    prepare_group();
    expect("prepared group start", nn_native_nccl_group_start(), 0, 0);
    const PreparedNcclRank prepared{reinterpret_cast<void*>(4096), 0, reinterpret_cast<void*>(1)};
    expect("prepared rank remains pending", enqueue_prepared_nccl_rank(prepared, 8, nccl_float32), 0, 0);
    expect("prepared group publishes rank", nn_native_nccl_group_end(), 0, 1);

    prepare_group();
    expect("failing group start", nn_native_nccl_group_start(), 0, 0);
    expect("accepted rank before failure", nn_native_nccl_all_reduce(&first, 0), 0, 0);
    end_result = 1;
    expect("failed group publishes nothing", nn_native_nccl_group_end(), 2, 0);
    end_result = 0;
    prepare_group();
    expect("empty group start", nn_native_nccl_group_start(), 0, 0);
    expect("empty group end", nn_native_nccl_group_end(), 0, 0);

    prepare_group(0);
    first.count = 0;
    expect("zero element group start", nn_native_nccl_group_start(), 0, 0);
    expect("zero element rank", nn_native_nccl_all_reduce(&first, 0), 0, 0);
    expect("zero element group end", nn_native_nccl_group_end(), 0, 0);
    first.count = 8;

    prepare_group();
    expect("enqueue failure group start", nn_native_nccl_group_start(), 0, 0);
    enqueue_result = 1;
    expect("failed rank is not accepted", nn_native_nccl_all_reduce(&first, 0), 4, 0);
    expect("group with no accepted ranks", nn_native_nccl_group_end(), 0, 0);
    enqueue_result = 0;

    prepare_group();
    expect("cancelled group start", nn_native_nccl_group_start(), 0, 0);
    expect("cancelled group accepted rank", nn_native_nccl_all_reduce(&first, 0), 0, 0);
    expect("cancel closes and submits the accepted rank", nn_native_nccl_cancel(), 0, 1);
    prepare_group();
    expect("next group start", nn_native_nccl_group_start(), 0, 0);
    expect("cancelled rank does not leak", nn_native_nccl_group_end(), 0, 0);

    prepare_group();
    expect("failed cancel group start", nn_native_nccl_group_start(), 0, 0);
    expect("failed cancel accepts rank", nn_native_nccl_all_reduce(&first, 0), 0, 0);
    end_result = 1;
    expect("failed cancel submits no confirmed work", nn_native_nccl_cancel(), 0, 0);
    end_result = 0;

    counters_enabled = false;
    prepare_group();
    expect("disabled group start", nn_native_nccl_group_start(), 0, 0);
    expect("disabled accepted rank", nn_native_nccl_all_reduce(&first, 0), 0, 0);
    expect("disabled group end", nn_native_nccl_group_end(), 0, 0);

    counters_enabled = true;
    Tensors values{{&first, &second}};
    prepare_group();
    expect("complete collective publishes ranks", nn_native_nccl_collective(&values), 0, 2);
    prepare_group();
    end_result = 1;
    expect("failed complete collective publishes nothing", nn_native_nccl_collective(&values), 3, 0);
    end_result = 0;
    prepare_group(0);
    first.count = second.count = 0;
    expect("zero element complete collective", nn_native_nccl_collective(&values), 0, 0);
    std::puts("NN NCCL dispatch counters: ok");
}
