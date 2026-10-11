#include "../native/nn_half.hpp"
#include <quidra/abi/half_format.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

int main(int argc, char** argv) {
    const std::uint64_t begin = argc > 1 ? std::strtoull(argv[1], nullptr, 0) : 0;
    const std::uint64_t end = argc > 2 ? std::strtoull(argv[2], nullptr, 0) : (UINT64_C(1) << 32);
    if (argc > 3 || begin > end || end > (UINT64_C(1) << 32)) return 2;
    for (bool bfloat : {false, true}) {
        const auto& format = bfloat ? quidra::abi::half::bfloat16 : quidra::abi::half::binary16;
        for (std::uint32_t raw = 0; raw < 65536; ++raw) {
            const float actual = nn_half::value(nn_half::widen(static_cast<std::uint16_t>(raw), bfloat));
            const float expected = static_cast<float>(quidra::abi::half::value_of(static_cast<std::uint16_t>(raw), format));
            // Neither Quidra format exposes a NaN payload as a real value.
            if (!(std::isnan(actual) && std::isnan(expected)) &&
                std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(expected)) {
                std::fprintf(stderr, "widen mismatch format=%d bits=%u\n", bfloat, raw); return 1;
            }
        }
        for (auto raw = begin; raw < end; ++raw) {
            const float source = std::bit_cast<float>(static_cast<std::uint32_t>(raw));
            const auto expected = quidra::abi::half::round_to(static_cast<double>(source), format);
            std::uint16_t actual = 0;
            const bool valid = nn_half::narrow(static_cast<std::uint32_t>(raw), bfloat, actual);
            const bool expected_valid = std::isfinite(source) && quidra::abi::half::is_finite(expected, format);
            if (valid != expected_valid || (valid && actual != expected)) {
                std::fprintf(stderr, "narrow mismatch format=%d bits=%llu\n", bfloat, static_cast<unsigned long long>(raw)); return 1;
            }
        }
    }
    return 0;
}
