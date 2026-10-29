// kernel_table.h -- a generated unit's kernels, as the runner can launch them (#649).
//
// The runner launches through cudaLaunchKernel / hipLaunchKernel, which take
// the kernel as an untyped address and its arguments as `void**` -- so one
// runner launches any signature. What that loses is the signature itself,
// and a launch spec that disagrees with it (a float where the kernel takes a
// double, a scalar where it takes a pointer, one argument too few) would read
// garbage instead of failing. `FJ_KERNEL(name)` keeps it: the entry records
// the kernel's arity and, per parameter, pointer or scalar, const-ness of the
// pointee, kind (f / i / u / b) and size, deduced from the kernel's own type
// by the compiler that compiled it. runner.h checks the spec against that
// before it allocates anything.
//
// A generated runner source spells its table as
//
//     static const fj::KernelEntry fj_kernels[] = {FJ_KERNEL(axpy)};
//
// after the unit's #include, so `axpy` is declared.

#pragma once

#include <cstddef>
#include <type_traits>

namespace fj {

struct ArgType {
    bool pointer = false;   // a global array, as opposed to a scalar
    bool is_const = false;  // a pointer to const: the kernel only reads it
    char kind = '?';        // 'f' floating, 'i' signed, 'u' unsigned, 'b' bool
    std::size_t size = 0;   // of the scalar, or of the pointee
};

template <class T>
constexpr char kind_of() {
    return std::is_same<T, bool>::value             ? 'b'
           : std::is_floating_point<T>::value       ? 'f'
           : std::is_integral<T>::value             ? (std::is_signed<T>::value ? 'i' : 'u')
                                                    : '?';
}

template <class T>
constexpr ArgType arg_type() {
    using U = typename std::remove_cv<T>::type;
    using E = typename std::remove_pointer<U>::type;
    return std::is_pointer<U>::value
               ? ArgType{true, std::is_const<E>::value,
                         kind_of<typename std::remove_cv<E>::type>(), sizeof(E)}
               : ArgType{false, false, kind_of<U>(), sizeof(U)};
}

struct KernelEntry {
    const char* name;
    const void* fn;
    std::size_t arity;
    const ArgType* args;  // `arity` of them
};

template <class... A>
KernelEntry kernel_entry(const char* name, void (*fn)(A...)) {
    static const ArgType types[sizeof...(A) + 1] = {arg_type<A>()..., ArgType{}};
    return KernelEntry{name, reinterpret_cast<const void*>(fn), sizeof...(A), types};
}

}  // namespace fj

#define FJ_KERNEL(name) ::fj::kernel_entry(#name, &name)
