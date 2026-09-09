#include "runtime_policy.hpp"

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace hmc {

bool configure_allocator(std::size_t mmap_threshold_bytes,
                         std::size_t trim_threshold_bytes) {
#ifdef __GLIBC__
    mallopt(M_MMAP_THRESHOLD, static_cast<int>(mmap_threshold_bytes));
    mallopt(M_TRIM_THRESHOLD, static_cast<int>(trim_threshold_bytes));
    return true;
#else
    (void)mmap_threshold_bytes;
    (void)trim_threshold_bytes;
    return false;
#endif
}

} // namespace hmc
