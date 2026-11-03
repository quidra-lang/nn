#pragma once

#include <cstdint>
#include <cstring>

namespace nn_half {
inline std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
inline float value(std::uint32_t bits) {
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
inline std::uint64_t nearest(std::uint64_t mantissa, int shift) {
    if (shift <= 0) return mantissa << -shift;
    if (shift > 64) return 0;
    if (shift == 64) return mantissa > UINT64_C(0x8000000000000000);
    const auto whole = mantissa >> shift;
    const auto remainder = mantissa & ((UINT64_C(1) << shift) - 1);
    const auto midpoint = UINT64_C(1) << (shift - 1);
    return whole + (remainder > midpoint || (remainder == midpoint && (whole & 1)));
}
inline std::uint32_t widen(std::uint16_t raw, bool bfloat) {
    if (bfloat) return static_cast<std::uint32_t>(raw) << 16;
    const auto sign = static_cast<std::uint32_t>(raw & 0x8000u) << 16;
    const auto exponent = (raw >> 10) & 31u;
    auto fraction = static_cast<std::uint32_t>(raw & 1023u);
    if (exponent == 31) return sign | 0x7f800000u | (fraction << 13);
    if (exponent != 0) return sign | ((exponent + 112u) << 23) | (fraction << 13);
    if (fraction == 0) return sign;
    int exponent32 = -14;
    while (fraction < 1024u) { fraction <<= 1; --exponent32; }
    return sign | (static_cast<std::uint32_t>(exponent32 + 127) << 23) | ((fraction & 1023u) << 13);
}
inline bool narrow(std::uint32_t raw, bool bfloat, std::uint16_t& result) {
    const auto magnitude = raw & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return false; // NaN, including signaling payloads.
    if (magnitude == 0x7f800000u) {
        // Preserve an existing infinity; finite narrowing overflow remains checked below.
        result = static_cast<std::uint16_t>(((raw >> 16) & 0x8000u) |
                                            (bfloat ? 0x7f80u : 0x7c00u));
        return true;
    }
    if (bfloat) {
        const auto rounded = raw + 0x7fffu + ((raw >> 16) & 1u);
        result = static_cast<std::uint16_t>(rounded >> 16);
        return (result & 0x7f80u) != 0x7f80u;
    }
    const auto sign = static_cast<std::uint16_t>((raw >> 16) & 0x8000u);
    const int exponent = static_cast<int>((raw >> 23) & 255u) - 127;
    const auto mantissa = (raw & 0x7fffffu) | 0x800000u;
    if (exponent < -25) { result = sign; return true; }
    if (exponent < -14) {
        result = sign | static_cast<std::uint16_t>(nearest(mantissa, -exponent - 1));
        return true;
    }
    auto rounded = nearest(mantissa, 13);
    int half_exponent = exponent + 15;
    if (rounded >= 2048) { rounded >>= 1; ++half_exponent; }
    if (half_exponent >= 31) return false;
    result = sign | static_cast<std::uint16_t>((half_exponent << 10) | (rounded & 1023u));
    return true;
}
} // namespace nn_half
