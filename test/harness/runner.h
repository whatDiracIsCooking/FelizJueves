// runner.h -- the generic runner: launches of generated kernels (#649, #657).
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
// A SEQUENCE (#657) is the same thing for several launches over shared
// arrays -- the ERI drivers' many launches into one `F`, `J`, `K` or `G`:
//
//     {"buffers": [
//        {"name": "D", "type": "f64", "in": "D.in.npy"},
//        {"name": "F", "type": "f64", "shape": [49], "out": "F.out.npy"}],
//      "launches": [
//        {"kernel": "f_0000", "grid": [3], "block": [32], "shared_bytes": 0,
//         "zero": ["W"],
//         "args": [{"name": "n_bra", "type": "i32", "value": 3}, ...,
//                  {"name": "F", "type": "f64", "array": true,
//                   "buffer": "F", "offset": 0}]}, ...]}
//
// A buffer is what an array argument of the single form is (`in`, or a
// zero-filled `shape`; `out` to read it back); an array argument names one
// by `buffer`, `offset` elements in (default 0), at the buffer's own type.
// A launch's optional `zero` buffers are memset to 0 just before it, inside
// its time. The launches run in order on one stream; the single form is the
// one-launch sequence whose buffers are its arrays, one per argument.
//
// Optional `runs` (default 1) and `warmup` (default 0), #654: `warmup + runs`
// runs of the whole sequence, each from the same state (every buffer some
// launch can write -- a non-const pointer names it, or `zero` does --
// restored to its `in`, or zero, outside the clock). `witness`:
//   "identical" (the default) -- every run's outputs required byte-identical
//               to the first's, the run-to-run witness; the first is written.
//   "each"      -- for kernels that add by atomics, whose order is the
//               card's: no comparison; the first run after warmup is written
//               to `out` and run r > 0 of the `runs` to `<out stem>.run<r>.npy`.
// Each of the `runs` prints `fj-runner run <r> events_ns <ns> wall_ns <ns>`
// (device events around the whole sequence, its `zero`s included; the host
// steady clock from before the first launch to the closing synchronize),
// and, for a sequence of more than one launch, `fj-runner run <r> launch <l>
// events_ns <ns>` per launch: an event pair around the kernel alone.
//
// Three optional switches for timing a whole build as one region (#659, the
// J driver's `measured_store.BUILD_TIMED_REGION`):
//   "timed_readback": true  -- every run (warmup included) downloads each
//               `out` buffer, one blocking copy per buffer in spec order,
//               right after the closing synchronize, and the run line ends
//               ` readback_ns <ns>`: the host clock around those copies, so
//               wall_ns + readback_ns is synchronize-to-read-back-done.
//   "launch_events": false  -- no event pair per launch: only the run's two
//               events are recorded, so nothing but the zeros and the
//               launches is between them (no per-launch lines are printed).
//   either way, before the first run: `fj-runner upload_ns <ns>`, the host
//               clock around allocating and uploading every buffer, between
//               two synchronizes. A buffer whose first use in the sequence is
//               a launch's `zero` (and has no `in`) is allocated but neither
//               uploaded nor restored between runs: the zero is its state.
//
// Before anything is allocated every launch is checked against its kernel's
// compiled signature (kernel_table.h): the arity, and per argument scalar or
// pointer, kind and size, that no single-form `out` names a
// pointer-to-const, and that a buffer's type is the argument's. A mismatch
// is an error, not a launch.
//
// Exit 0 with every output written; 1 on any error (stderr says which); 2 on
// bad usage. On success stdout carries the `run` lines, then one
// `fj-runner launch ...` line per launch, then one `fj-runner out <name>
// <path>` line per file written.

#pragma once

#include <chrono>
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

// One device array of the case: an `in` .npy uploaded, or a zero-filled
// `shape`; written back to `out` when it has one. `initial` is what it holds
// before each run -- restored before every run but the first when any launch
// can write it (a non-const pointer names it).
struct Buffer {
    std::string name;
    const TypeInfo* type = nullptr;
    std::string out;                     // empty: never read back
    npy::Array host;                     // the first run's result (shape, dtype)
    std::vector<unsigned char> initial;  // its `in`, or zeros
    std::vector<npy::Array> each;        // witness "each": runs 1.. after warmup
    DeviceBuffer dev;
    bool written = false;
    bool has_in = false;
    bool zeroed_first = false;           // its first use is a `zero`
    std::vector<unsigned char> got;      // timed_readback's copy, per run
};

struct Launch {
    const KernelEntry* entry = nullptr;
    std::string kernel;
    dim3 grid, block;
    long long shm = 0;
    std::vector<std::uint64_t> scalars;
    std::vector<std::size_t> buffer;  // per argument: its buffer (pointers only)
    std::vector<std::size_t> offset;  // per argument: elements into it
    std::vector<std::size_t> zero;    // buffers memset to 0 before the launch
    std::vector<void*> pointers;
    std::vector<void*> args;
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

// A buffer from `{name, type, in | shape, out?}`: a `buffers` entry, or a
// single-launch spec's array argument.
inline Buffer read_buffer(const json::Value& b, const std::string& base, const std::string& where) {
    Buffer buf;
    buf.name = b.at("name").str();
    buf.type = &type_info(b.at("type").str());
    const json::Value* in = b.find("in");
    if (in != nullptr) {
        const std::string path = resolve(base, in->str());
        buf.has_in = true;
        buf.host = npy::read(path);
        if (buf.host.descr != buf.type->descr) {
            throw std::runtime_error(path + ": dtype " + buf.host.descr + ", but " + buf.name +
                                     " is " + buf.type->name + " (" + buf.type->descr + ")");
        }
    } else {
        const json::Value* shape = b.find("shape");
        if (shape == nullptr) {
            throw std::runtime_error(where + ": an array needs `in`, `out` or both");
        }
        buf.host.descr = buf.type->descr;
        for (const json::Value& d : shape->items) {
            const long long x = d.integer();
            if (x < 0) throw std::runtime_error(buf.name + ": negative extent");
            buf.host.shape.push_back(static_cast<std::size_t>(x));
        }
        buf.host.data.assign(buf.host.count() * buf.type->size, 0);
    }
    const json::Value* out = b.find("out");
    if (out != nullptr) buf.out = resolve(base, out->str());
    buf.initial = buf.host.data;
    return buf;
}

inline const KernelEntry* find_kernel(const std::string& name, const KernelEntry* table,
                                      std::size_t n_table) {
    for (std::size_t k = 0; k < n_table; ++k) {
        if (name == table[k].name) return &table[k];
    }
    throw std::runtime_error("no kernel '" + name + "' in this runner");
}

// One launch's description checked against its kernel's compiled signature;
// `single` is the single-launch form, whose arrays ARE its buffers (each
// named after its argument, appended to `buffers` here).
inline Launch read_launch(const json::Value& l, const std::string& base, const KernelEntry* table,
                          std::size_t n_table, std::vector<Buffer>& buffers, bool single) {
    Launch run;
    run.kernel = l.at("kernel").str();
    run.entry = find_kernel(run.kernel, table, n_table);
    const KernelEntry& entry = *run.entry;
    const std::string& kname = run.kernel;
    const json::Value& args = l.at("args");
    if (!args.is(json::Value::Kind::Array) || args.items.size() != entry.arity) {
        throw std::runtime_error(kname + " takes " + std::to_string(entry.arity) +
                                 " arguments; the spec lists " +
                                 std::to_string(args.items.size()));
    }
    run.scalars.assign(entry.arity, 0);
    run.buffer.assign(entry.arity, 0);
    run.offset.assign(entry.arity, 0);
    for (std::size_t k = 0; k < entry.arity; ++k) {
        const json::Value& a = args.items[k];
        const std::string& name = a.at("name").str();
        const TypeInfo& t = type_info(a.at("type").str());
        const ArgType& want = entry.args[k];
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
        if (!is_array) {
            if (a.find("value") == nullptr) {
                throw std::runtime_error(where + ": a scalar needs `value`");
            }
            run.scalars[k] = scalar_bits(t, a.at("value"), name);
            continue;
        }
        if (single) {
            if (a.find("in") == nullptr && a.find("out") == nullptr) {
                throw std::runtime_error(where + ": an array needs `in`, `out` or both");
            }
            if (a.find("out") != nullptr && want.is_const) {
                throw std::runtime_error(where + ": the kernel takes it const; it cannot "
                                                 "be an output");
            }
            buffers.push_back(read_buffer(a, base, where));
            run.buffer[k] = buffers.size() - 1;
        } else {
            const std::string& bname = a.at("buffer").str();
            std::size_t b = buffers.size();
            for (std::size_t j = 0; j < buffers.size(); ++j) {
                if (buffers[j].name == bname) b = j;
            }
            if (b == buffers.size()) throw std::runtime_error(where + ": no buffer '" + bname + "'");
            if (buffers[b].type != &t) {
                throw std::runtime_error(where + ": spec type " + t.name + ", but buffer " +
                                         bname + " is " + buffers[b].type->name);
            }
            const json::Value* off = a.find("offset");
            const long long o = off ? off->integer() : 0;
            if (o < 0 || static_cast<std::size_t>(o) > buffers[b].host.count()) {
                throw std::runtime_error(where + ": offset " + std::to_string(o) +
                                         " is outside buffer " + bname);
            }
            run.buffer[k] = b;
            run.offset[k] = static_cast<std::size_t>(o);
        }
        if (!want.is_const) buffers[run.buffer[k]].written = true;
    }
    run.grid = dims(l.at("grid"), "grid");
    run.block = dims(l.at("block"), "block");
    const json::Value* shared = l.find("shared_bytes");
    run.shm = shared ? shared->integer() : 0;
    if (run.shm < 0) throw std::runtime_error("shared_bytes is negative");
    const json::Value* zero = l.find("zero");
    if (zero != nullptr) {
        for (const json::Value& z : zero->items) {
            std::size_t b = buffers.size();
            for (std::size_t j = 0; j < buffers.size(); ++j) {
                if (buffers[j].name == z.str()) b = j;
            }
            if (b == buffers.size()) {
                throw std::runtime_error(kname + ": `zero` names no buffer '" + z.str() + "'");
            }
            run.zero.push_back(b);
            buffers[b].written = true;
        }
    }
    return run;
}

// `<stem>.npy` -> `<stem>.run<r>.npy`: witness "each"'s later runs.
inline std::string run_path(const std::string& path, int r) {
    const std::string tail = ".npy";
    const bool npy_ext = path.size() >= tail.size() &&
                         path.compare(path.size() - tail.size(), tail.size(), tail) == 0;
    const std::string stem = npy_ext ? path.substr(0, path.size() - tail.size()) : path;
    return stem + ".run" + std::to_string(r) + ".npy";
}

inline int run(const std::string& spec_path, const KernelEntry* table, std::size_t n_table) {
    const json::Value spec = json::read(spec_path);
    const std::size_t slash = spec_path.rfind('/');
    const std::string base = slash == std::string::npos ? "" : spec_path.substr(0, slash);

    // Every launch, and every buffer, checked before the card is touched.
    std::vector<Buffer> buffers;
    std::vector<Launch> launches;
    const json::Value* seq = spec.find("launches");
    if (seq == nullptr) {
        launches.push_back(read_launch(spec, base, table, n_table, buffers, true));
    } else {
        if (spec.find("kernel") != nullptr || spec.find("args") != nullptr) {
            throw std::runtime_error("a spec is one launch (`kernel`, `args`) or a sequence "
                                     "(`buffers`, `launches`), not both");
        }
        const json::Value& bs = spec.at("buffers");
        for (const json::Value& b : bs.items) {
            const std::string where = "buffer " + b.at("name").str();
            for (const Buffer& x : buffers) {
                if (x.name == b.at("name").str()) throw std::runtime_error(where + " is listed twice");
            }
            buffers.push_back(read_buffer(b, base, where));
        }
        if (!seq->is(json::Value::Kind::Array) || seq->items.empty()) {
            throw std::runtime_error("launches must be a non-empty list");
        }
        for (const json::Value& l : seq->items) {
            launches.push_back(read_launch(l, base, table, n_table, buffers, false));
        }
    }
    const int warmup = count(spec, "warmup", 0, 0);
    const int runs = count(spec, "runs", 1, 1);
    const json::Value* wit = spec.find("witness");
    const std::string witness = wit ? wit->str() : "identical";
    if (witness != "identical" && witness != "each") {
        throw std::runtime_error("witness must be \"identical\" or \"each\"");
    }
    const json::Value* trb = spec.find("timed_readback");
    const bool timed_readback = trb != nullptr && trb->boolean;
    const json::Value* lev = spec.find("launch_events");
    const bool launch_events = lev == nullptr || lev->boolean;

    // A buffer zeroed before anything else touches it needs no upload.
    {
        std::vector<bool> seen(buffers.size(), false);
        for (const Launch& l : launches) {
            for (std::size_t z : l.zero) {
                if (!seen[z]) buffers[z].zeroed_first = !buffers[z].has_in;
                seen[z] = true;
            }
            for (std::size_t k = 0; k < l.entry->arity; ++k) {
                if (l.entry->args[k].pointer) seen[l.buffer[k]] = true;
            }
        }
    }

    int devices = 0;
    gpu::check(gpu::device_count(&devices), "device count");
    if (devices < 1) throw std::runtime_error("no device");
    const json::Value* dev = spec.find("device");
    gpu::check(gpu::set_device(dev ? static_cast<int>(dev->integer()) : 0), "set device");

    gpu::check(gpu::synchronize(), "synchronize");
    const auto u0 = std::chrono::steady_clock::now();
    for (Buffer& b : buffers) {
        b.dev = DeviceBuffer(b.initial.size());
        if (!b.zeroed_first) b.dev.upload(b.initial.data());
    }
    gpu::check(gpu::synchronize(), "synchronize");
    std::printf("fj-runner upload_ns %lld\n",
                static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - u0)
                                           .count()));
    for (Buffer& b : buffers) {
        if (timed_readback && !b.out.empty()) b.got.resize(b.host.data.size());
    }
    for (Launch& l : launches) {
        const std::size_t n = l.entry->arity;
        l.pointers.assign(n, nullptr);
        l.args.assign(n, nullptr);
        for (std::size_t k = 0; k < n; ++k) {
            if (!l.entry->args[k].pointer) {
                l.args[k] = &l.scalars[k];
                continue;
            }
            const Buffer& b = buffers[l.buffer[k]];
            l.pointers[k] = static_cast<unsigned char*>(b.dev.get()) + l.offset[k] * b.type->size;
            l.args[k] = &l.pointers[k];
        }
    }

    // ev[0] opens the run, before the first launch's `zero`; launch l is
    // ev[2l + 1] -> ev[2l + 2], the kernel alone; the run is ev[0] -> ev[2n].
    const std::size_t n_launch = launches.size();
    std::vector<gpu::event_t> ev(2 * n_launch + 1);
    for (gpu::event_t& e : ev) gpu::check(gpu::event_create(&e), "event create");
    std::vector<unsigned char> again;
    for (int r = 0; r < warmup + runs; ++r) {
        if (r > 0) {
            for (Buffer& b : buffers) {
                if (b.written && !b.zeroed_first) b.dev.upload(b.initial.data());
            }
        }
        gpu::check(gpu::synchronize(), "synchronize");
        const auto t0 = std::chrono::steady_clock::now();
        gpu::check(gpu::event_record(ev[0]), "event record");
        for (std::size_t i = 0; i < n_launch; ++i) {
            Launch& l = launches[i];
            for (std::size_t z : l.zero) buffers[z].dev.zero();
            if (launch_events) gpu::check(gpu::event_record(ev[2 * i + 1]), "event record");
            gpu::check(gpu::launch(l.entry->fn, l.grid, l.block, l.args.data(),
                                   static_cast<std::size_t>(l.shm), nullptr),
                       "launch");
            gpu::check(gpu::last_error(), "launch (last error)");
            if (launch_events || i + 1 == n_launch) {
                gpu::check(gpu::event_record(ev[2 * i + 2]), "event record");
            }
        }
        gpu::check(gpu::synchronize(), "synchronize");
        const auto t1 = std::chrono::steady_clock::now();
        long long readback = 0;
        if (timed_readback) {
            for (Buffer& b : buffers) {
                if (!b.out.empty()) b.dev.download(b.got.data());
            }
            readback = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - t1)
                           .count();
        }
        // An `out` buffer's contents this run: the timed copy, or a download now.
        auto fetch = [&](const Buffer& b, unsigned char* dst) {
            if (timed_readback) {
                std::memcpy(dst, b.got.data(), b.got.size());
            } else {
                b.dev.download(dst);
            }
        };
        if (r >= warmup) {
            float ms = 0.0f;
            gpu::check(gpu::elapsed_ms(&ms, ev[0], ev[2 * n_launch]), "event elapsed");
            const long long wall =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (timed_readback) {
                std::printf("fj-runner run %d events_ns %.0f wall_ns %lld readback_ns %lld\n",
                            r - warmup, 1e6 * double(ms), wall, readback);
            } else {
                std::printf("fj-runner run %d events_ns %.0f wall_ns %lld\n", r - warmup,
                            1e6 * double(ms), wall);
            }
            for (std::size_t i = 0; launch_events && n_launch > 1 && i < n_launch; ++i) {
                gpu::check(gpu::elapsed_ms(&ms, ev[2 * i + 1], ev[2 * i + 2]), "event elapsed");
                std::printf("fj-runner run %d launch %zu events_ns %.0f\n", r - warmup, i,
                            1e6 * double(ms));
            }
        }
        for (Buffer& b : buffers) {
            if (b.out.empty()) continue;
            if (r == 0) {
                fetch(b, b.host.data.data());
                continue;
            }
            if (witness == "each") {
                if (r <= warmup) continue;
                b.each.push_back(b.host);
                fetch(b, b.each.back().data.data());
                continue;
            }
            again.resize(b.host.data.size());
            fetch(b, again.data());
            if (again != b.host.data) {
                throw std::runtime_error(b.name + ": run " + std::to_string(r) +
                                         " differs from run 0 -- not run-to-run "
                                         "deterministic");
            }
        }
        // witness "each": the first kept run is the first after warmup.
        if (witness == "each" && r == warmup && r > 0) {
            for (Buffer& b : buffers) {
                if (!b.out.empty()) fetch(b, b.host.data.data());
            }
        }
    }
    for (gpu::event_t e : ev) (void)gpu::event_destroy(e);
    for (const Launch& l : launches) {
        std::printf("fj-runner launch %s grid %u %u %u block %u %u %u shared %lld warmup %d "
                    "runs %d\n",
                    l.kernel.c_str(), l.grid.x, l.grid.y, l.grid.z, l.block.x, l.block.y,
                    l.block.z, l.shm, warmup, runs);
    }

    for (Buffer& b : buffers) {
        if (b.out.empty()) continue;
        npy::write(b.out, b.host);
        std::printf("fj-runner out %s %s\n", b.name.c_str(), b.out.c_str());
        for (std::size_t i = 0; i < b.each.size(); ++i) {
            const std::string path = run_path(b.out, static_cast<int>(i + 1));
            npy::write(path, b.each[i]);
            std::printf("fj-runner out %s %s\n", b.name.c_str(), path.c_str());
        }
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
