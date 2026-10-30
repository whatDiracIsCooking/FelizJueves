// runner.h -- the generic runner: one launch of one generated kernel (#649).
//
//     <runner> spec.json
//
// A runner binary is a generated unit plus this file: the source
// test/harness_driver.py writes #includes the harness, then the unit, then
// lists the unit's kernels (kernel_table.h) and calls `fj::runner_main`. The
// launch is described by the spec, so the binary is compiled once per unit
// and run once per case:
//
//     {"kernel": "axpy",
//      "grid": [2, 1, 1], "block": [64, 1, 1], "shared_bytes": 0,
//      "args": [
//        {"name": "n",   "type": "i32", "value": 128},
//        {"name": "x",   "type": "f64", "array": true, "in": "x.npy"},
//        {"name": "out", "type": "f64", "array": true, "out": "out.npy",
//         "shape": [128]}]}
//
// `args` is the kernel's parameter list, in order, every parameter once
// (harness_driver.py builds it from the IR `Kernel.params`, by name). A
// scalar carries `value`; an array carries `in` (an .npy uploaded before the
// launch), `out` (downloaded after it and written as an .npy), or both (an
// array the kernel reads and updates). An `out` with no `in` is zero-filled
// on the device and needs its `shape`. Relative paths resolve against the
// spec's own directory. `type` is f64 f32 i32 i64 u32 u64; an input's dtype
// must be exactly that type's little-endian spelling.
//
// Optional `runs` (default 1) and `warmup` (default 0), #654: `warmup + runs`
// launches, each from the same state (every `out` restored to its `in`, or
// zero), each one's outputs required byte-identical to the first's -- the
// run-to-run witness -- and each of the `runs` timed by device events, one
// `fj-runner run <r> events_ns <ns>` line apiece.
//
// Before anything is allocated the spec is checked against the kernel's
// compiled signature (kernel_table.h): the arity, and per argument scalar or
// pointer, kind and size, and that no `out` names a pointer-to-const. A
// mismatch is an error, not a launch.
//
// Exit 0 with every output written; 1 on any error (stderr says which); 2 on
// bad usage. On success stdout carries the `run` lines, then one
// `fj-runner launch ...` line, then one `fj-runner out <name> <path>` line
// per output.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "device_buffer.h"
#include "fj_gpu.h"
#include "json.h"
#include "kernel_table.h"
#include "npy.h"

namespace fj {
namespace runner {

struct TypeInfo {
    const char* name;
    char kind;
    std::size_t size;
    const char* descr;
};

inline const TypeInfo& type_info(const std::string& name) {
    static const TypeInfo types[] = {
        {"f64", 'f', 8, "<f8"}, {"f32", 'f', 4, "<f4"}, {"i32", 'i', 4, "<i4"},
        {"i64", 'i', 8, "<i8"}, {"u32", 'u', 4, "<u4"}, {"u64", 'u', 8, "<u8"},
    };
    for (const TypeInfo& t : types) {
        if (name == t.name) return t;
    }
    throw std::runtime_error("unknown type '" + name + "' (f64 f32 i32 i64 u32 u64)");
}

inline std::string resolve(const std::string& base, const std::string& path) {
    if (path.empty() || path[0] == '/' || base.empty()) return path;
    return base + "/" + path;
}

inline dim3 dims(const json::Value& v, const char* what) {
    if (!v.is(json::Value::Kind::Array) || v.items.empty() || v.items.size() > 3) {
        throw std::runtime_error(std::string(what) + " must be 1 to 3 integers");
    }
    unsigned d[3] = {1, 1, 1};
    for (std::size_t k = 0; k < v.items.size(); ++k) {
        const long long x = v.items[k].integer();
        if (x < 1 || x > 0x7fffffffLL) {
            throw std::runtime_error(std::string(what) + " extent out of range");
        }
        d[k] = static_cast<unsigned>(x);
    }
    return dim3(d[0], d[1], d[2]);
}

// A scalar's bytes, at its own type, in an 8-byte slot.
inline std::uint64_t scalar_bits(const TypeInfo& t, const json::Value& v,
                                 const std::string& name) {
    std::uint64_t slot = 0;
    if (t.kind == 'f' && t.size == 8) {
        const double x = v.number();
        std::memcpy(&slot, &x, sizeof x);
    } else if (t.kind == 'f') {
        const float x = static_cast<float>(v.number());
        std::memcpy(&slot, &x, sizeof x);
    } else {
        const long long x = v.integer();
        const bool fits = t.size == 8 ? (t.kind == 'i' || x >= 0)
                          : t.kind == 'i' ? (x >= INT32_MIN && x <= INT32_MAX)
                                          : (x >= 0 && x <= static_cast<long long>(UINT32_MAX));
        if (!fits) throw std::runtime_error(name + ": value out of range for its type");
        if (t.size == 8) {
            std::memcpy(&slot, &x, sizeof x);
        } else if (t.kind == 'i') {
            const std::int32_t y = static_cast<std::int32_t>(x);
            std::memcpy(&slot, &y, sizeof y);
        } else {
            const std::uint32_t y = static_cast<std::uint32_t>(x);
            std::memcpy(&slot, &y, sizeof y);
        }
    }
    return slot;
}

struct Output {
    std::string name;
    std::string path;
    npy::Array host;                     // the first launch's result
    std::vector<unsigned char> initial;  // its `in`, or zeros: restored per launch
    std::size_t arg = 0;                 // index into the device buffers
};

inline int count(const json::Value& spec, const char* key, int fallback, int least) {
    const json::Value* v = spec.find(key);
    const long long x = v ? v->integer() : fallback;
    if (x < least || x > 1000000) {
        throw std::runtime_error(std::string(key) + " must be an integer from " +
                                 std::to_string(least) + " to 1000000");
    }
    return static_cast<int>(x);
}

inline int run(const std::string& spec_path, const KernelEntry* table, std::size_t n_table) {
    const json::Value spec = json::read(spec_path);
    const std::size_t slash = spec_path.rfind('/');
    const std::string base = slash == std::string::npos ? "" : spec_path.substr(0, slash);

    const std::string& kname = spec.at("kernel").str();
    const KernelEntry* entry = nullptr;
    for (std::size_t k = 0; k < n_table; ++k) {
        if (kname == table[k].name) entry = &table[k];
    }
    if (entry == nullptr) throw std::runtime_error("no kernel '" + kname + "' in this runner");

    const json::Value& args = spec.at("args");
    if (!args.is(json::Value::Kind::Array) || args.items.size() != entry->arity) {
        throw std::runtime_error(kname + " takes " + std::to_string(entry->arity) +
                                 " arguments; the spec lists " +
                                 std::to_string(args.items.size()));
    }

    // Check every argument against the signature before touching the card.
    std::vector<const TypeInfo*> types(entry->arity);
    for (std::size_t k = 0; k < entry->arity; ++k) {
        const json::Value& a = args.items[k];
        const std::string& name = a.at("name").str();
        const TypeInfo& t = type_info(a.at("type").str());
        const ArgType& want = entry->args[k];
        const json::Value* array = a.find("array");
        const bool is_array = array != nullptr && array->boolean;
        const std::string where = kname + " argument " + std::to_string(k) + " (" + name + ")";
        if (is_array != want.pointer) {
            throw std::runtime_error(where + ": the kernel takes a " +
                                     (want.pointer ? "pointer" : "scalar"));
        }
        if (t.kind != want.kind || t.size != want.size) {
            throw std::runtime_error(where + ": spec type " + t.name +
                                     " does not match the kernel's (kind '" +
                                     std::string(1, want.kind) + "', " +
                                     std::to_string(want.size) + " bytes)");
        }
        if (is_array) {
            if (a.find("in") == nullptr && a.find("out") == nullptr) {
                throw std::runtime_error(where + ": an array needs `in`, `out` or both");
            }
            if (a.find("out") != nullptr && want.is_const) {
                throw std::runtime_error(where + ": the kernel takes it const; it cannot "
                                                 "be an output");
            }
        } else if (a.find("value") == nullptr) {
            throw std::runtime_error(where + ": a scalar needs `value`");
        }
        types[k] = &t;
    }
    const int warmup = count(spec, "warmup", 0, 0);
    const int runs = count(spec, "runs", 1, 1);

    int devices = 0;
    gpu::check(gpu::device_count(&devices), "device count");
    if (devices < 1) throw std::runtime_error("no device");
    const json::Value* dev = spec.find("device");
    gpu::check(gpu::set_device(dev ? static_cast<int>(dev->integer()) : 0), "set device");

    std::vector<std::uint64_t> scalars(entry->arity, 0);
    std::vector<DeviceBuffer> buffers(entry->arity);
    std::vector<void*> pointers(entry->arity, nullptr);
    std::vector<void*> launch_args(entry->arity, nullptr);
    std::vector<Output> outputs;

    for (std::size_t k = 0; k < entry->arity; ++k) {
        const json::Value& a = args.items[k];
        const std::string& name = a.at("name").str();
        const TypeInfo& t = *types[k];
        if (!entry->args[k].pointer) {
            scalars[k] = scalar_bits(t, a.at("value"), name);
            launch_args[k] = &scalars[k];
            continue;
        }
        npy::Array host;
        const json::Value* in = a.find("in");
        if (in != nullptr) {
            const std::string path = resolve(base, in->str());
            host = npy::read(path);
            if (host.descr != t.descr) {
                throw std::runtime_error(path + ": dtype " + host.descr + ", but " + name +
                                         " is " + t.name + " (" + t.descr + ")");
            }
            buffers[k] = DeviceBuffer(host.data.size());
            buffers[k].upload(host.data.data());
        } else {
            host.descr = t.descr;
            for (const json::Value& d : a.at("shape").items) {
                const long long x = d.integer();
                if (x < 0) throw std::runtime_error(name + ": negative extent");
                host.shape.push_back(static_cast<std::size_t>(x));
            }
            host.data.assign(host.count() * t.size, 0);
            buffers[k] = DeviceBuffer(host.data.size());
            buffers[k].zero();
        }
        pointers[k] = buffers[k].get();
        launch_args[k] = &pointers[k];
        const json::Value* out = a.find("out");
        if (out != nullptr) {
            std::vector<unsigned char> initial = host.data;
            outputs.push_back(
                Output{name, resolve(base, out->str()), std::move(host), std::move(initial), k});
        }
    }

    const dim3 grid = dims(spec.at("grid"), "grid");
    const dim3 block = dims(spec.at("block"), "block");
    const json::Value* shared = spec.find("shared_bytes");
    const long long shm = shared ? shared->integer() : 0;
    if (shm < 0) throw std::runtime_error("shared_bytes is negative");

    gpu::event_t ev[2];  // destroyed at the end; an error exits the process anyway
    for (gpu::event_t& e : ev) gpu::check(gpu::event_create(&e), "event create");
    std::vector<unsigned char> again;
    for (int r = 0; r < warmup + runs; ++r) {
        if (r > 0) {
            for (const Output& o : outputs) buffers[o.arg].upload(o.initial.data());
        }
        gpu::check(gpu::event_record(ev[0]), "event record");
        gpu::check(gpu::launch(entry->fn, grid, block, launch_args.data(),
                               static_cast<std::size_t>(shm), nullptr),
                   "launch");
        gpu::check(gpu::last_error(), "launch (last error)");
        gpu::check(gpu::event_record(ev[1]), "event record");
        gpu::check(gpu::synchronize(), "synchronize");
        if (r >= warmup) {
            float ms = 0.0f;
            gpu::check(gpu::elapsed_ms(&ms, ev[0], ev[1]), "event elapsed");
            std::printf("fj-runner run %d events_ns %.0f\n", r - warmup, 1e6 * double(ms));
        }
        for (Output& o : outputs) {
            if (r == 0) {
                buffers[o.arg].download(o.host.data.data());
                continue;
            }
            again.resize(o.host.data.size());
            buffers[o.arg].download(again.data());
            if (again != o.host.data) {
                throw std::runtime_error(o.name + ": launch " + std::to_string(r) +
                                         " differs from launch 0 -- not run-to-run "
                                         "deterministic");
            }
        }
    }
    for (gpu::event_t e : ev) (void)gpu::event_destroy(e);
    std::printf("fj-runner launch %s grid %u %u %u block %u %u %u shared %lld warmup %d "
                "runs %d\n",
                kname.c_str(), grid.x, grid.y, grid.z, block.x, block.y, block.z, shm, warmup,
                runs);

    for (Output& o : outputs) {
        npy::write(o.path, o.host);
        std::printf("fj-runner out %s %s\n", o.name.c_str(), o.path.c_str());
    }
    return 0;
}

}  // namespace runner

template <std::size_t N>
int runner_main(int argc, char** argv, const KernelEntry (&table)[N]) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s spec.json\n", argc > 0 ? argv[0] : "runner");
        return 2;
    }
    try {
        return runner::run(argv[1], table, N);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fj-runner: %s\n", e.what());
        return 1;
    }
}

}  // namespace fj
