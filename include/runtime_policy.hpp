#pragma once

#include <cstddef>

namespace hmc {

// Process-wide allocator policy (spec A19).
//
// Large iterative solves allocate and free multi-megabyte buffers; on glibc,
// leaving them in the arena instead of returning them to the OS made peak RSS
// climb over the iterations. The nested solver and the H-matrix constructor
// used to call mallopt() themselves, which silently changed the policy of the
// WHOLE host process — including code that has nothing to do with ASPHER.
//
// That is now an explicit application-level choice. Call this once from an
// application or benchmark if you measure that it helps; ASPHER never calls it
// on its own. Returns false when the platform has no such control (non-glibc),
// in which case nothing was changed.
//
// Defaults match the thresholds the library used to force.
bool configure_allocator(std::size_t mmap_threshold_bytes = 128 * 1024,
                         std::size_t trim_threshold_bytes = 128 * 1024);

} // namespace hmc
