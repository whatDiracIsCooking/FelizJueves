// device_buffer.h -- an owned span of device memory, in bytes (#649).
//
// Move-only RAII: the destructor frees, a moved-from buffer owns nothing.
// Untyped on purpose -- the runner hands the kernel a pointer whose element
// type it checked against the kernel's own signature, so a typed buffer would
// add a cast and nothing else. A zero-byte buffer still allocates one byte,
// so every array argument is a real device address and never a null the
// kernel might be compiled to test against.
//
// The destructor cannot throw; a failed free there is reported on stderr and
// otherwise ignored (the process is about to report its own result).

#pragma once

#include <cstddef>
#include <cstdio>
#include <utility>

#include "fj_gpu.h"

namespace fj {

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        gpu::check(gpu::malloc(&ptr_, bytes ? bytes : 1), "malloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = std::exchange(other.ptr_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    void* get() const { return ptr_; }
    std::size_t bytes() const { return bytes_; }

    void zero() { gpu::check(gpu::memset(ptr_, 0, bytes_), "memset"); }

    // Copy exactly bytes() from `host` to the device, or back.
    void upload(const void* host) {
        if (bytes_) gpu::check(gpu::to_device(ptr_, host, bytes_), "memcpy H2D");
    }
    void download(void* host) const {
        if (bytes_) gpu::check(gpu::to_host(host, ptr_, bytes_), "memcpy D2H");
    }

private:
    void release() noexcept {
        if (ptr_ != nullptr) {
            const gpu::error_t e = gpu::free(ptr_);
            if (!gpu::ok(e)) {
                std::fprintf(stderr, "fj-runner: free: %s\n", gpu::error_string(e));
            }
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace fj
