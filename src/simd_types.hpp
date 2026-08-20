#pragma once

#include <cstdint>
#include <vector>
#include <cstdlib>
#include <new>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Cache-line and SIMD-aligned vector allocator
template <typename T, size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#else
        ptr = std::malloc(n * sizeof(T));
        if (!ptr) throw std::bad_alloc();
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
#if defined(_MSC_VER) || defined(__MINGW32__)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
};

namespace simd {

// Fast vector popcount intersection: count(A & B)
static inline int popcount_and(const uint64_t* __restrict__ a, 
                               const uint64_t* __restrict__ b, 
                               int num_words) {
    int total = 0;
    int w = 0;
#if defined(__AVX2__)
    for (; w + 4 <= num_words; w += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + w));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + w));
        __m256i vand = _mm256_and_si256(va, vb);

        uint64_t r0 = _mm256_extract_epi64(vand, 0);
        uint64_t r1 = _mm256_extract_epi64(vand, 1);
        uint64_t r2 = _mm256_extract_epi64(vand, 2);
        uint64_t r3 = _mm256_extract_epi64(vand, 3);

        total += __builtin_popcountll(r0) + __builtin_popcountll(r1) + 
                 __builtin_popcountll(r2) + __builtin_popcountll(r3);
    }
#endif
    for (; w < num_words; ++w) {
        total += __builtin_popcountll(a[w] & b[w]);
    }
    return total;
}

// Fast vector and-not popcount: count(A & ~B)
static inline int popcount_andnot(const uint64_t* __restrict__ a, 
                                  const uint64_t* __restrict__ b, 
                                  int num_words) {
    int total = 0;
    int w = 0;
#if defined(__AVX2__)
    for (; w + 4 <= num_words; w += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + w));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + w));
        __m256i vandnot = _mm256_andnot_si256(vb, va); // (~vb) & va

        uint64_t r0 = _mm256_extract_epi64(vandnot, 0);
        uint64_t r1 = _mm256_extract_epi64(vandnot, 1);
        uint64_t r2 = _mm256_extract_epi64(vandnot, 2);
        uint64_t r3 = _mm256_extract_epi64(vandnot, 3);

        total += __builtin_popcountll(r0) + __builtin_popcountll(r1) + 
                 __builtin_popcountll(r2) + __builtin_popcountll(r3);
    }
#endif
    for (; w < num_words; ++w) {
        total += __builtin_popcountll(a[w] & ~b[w]);
    }
    return total;
}

} // namespace simd
