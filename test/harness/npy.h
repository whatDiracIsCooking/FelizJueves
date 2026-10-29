// npy.h -- NumPy .npy files in and out, host-side only (#649).
//
// The slice the runner needs, written for it rather than copied from
// FreeBreakfast's io/npy (which is float64, column-major, and a C++23 module):
//
//   * READ versions 1.0, 2.0 and 3.0 (they differ only in the width of the
//     header length), any little-endian or byte-order-free numeric dtype --
//     `<f8`, `<f4`, `<i4`, `|u1`, ... -- in C order. Big-endian data and
//     `fortran_order: True` are REFUSED, never reinterpreted: the caller
//     passes `np.ascontiguousarray(a)`, which is what harness_driver.py does.
//   * WRITE what `np.save` writes for a C-ordered array: version 1.0, the
//     same dict spelling (keys sorted, `repr` of the shape tuple), numpy's
//     21 spaces of growth room for the first axis, and padding to 64 bytes.
//     So a file this writes is BYTE-IDENTICAL to `np.save(path, a)` of the
//     equal array -- test/test_harness.py holds the runner's outputs to that.
//
// The payload is copied as bytes, so the host must be little-endian; both
// directions check it and throw rather than swap.

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fj {
namespace npy {

struct Array {
    std::string descr;               // numpy's dtype.str, e.g. "<f8"
    std::vector<std::size_t> shape;  // () is a 0-d array of one element
    std::vector<unsigned char> data; // the payload, C order

    std::size_t count() const {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        return n;
    }
};

inline bool host_little_endian() {
    const std::uint16_t one = 1;
    unsigned char first = 0;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

// The element size a descr spells, after checking it is one we copy as is:
// a byte order of '<' or '|', a kind of f / i / u / b, a positive size.
inline std::size_t itemsize(const std::string& descr) {
    if (descr.size() < 3 || (descr[0] != '<' && descr[0] != '|') ||
        std::strchr("fiub", descr[1]) == nullptr) {
        throw std::runtime_error("npy: unsupported dtype '" + descr +
                                 "' (want little-endian f/i/u/b)");
    }
    std::size_t n = 0;
    for (std::size_t i = 2; i < descr.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(descr[i]))) {
            throw std::runtime_error("npy: unsupported dtype '" + descr + "'");
        }
        n = n * 10 + static_cast<std::size_t>(descr[i] - '0');
    }
    if (n == 0) throw std::runtime_error("npy: zero-size dtype '" + descr + "'");
    return n;
}

namespace detail {

// A cursor over the header's Python-literal dict.
struct Literal {
    const std::string& s;
    std::size_t i = 0;

    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("npy header: " + why + " at offset " +
                                 std::to_string(i) + " in: " + s);
    }
    void ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    bool eat(char c) {
        ws();
        if (i < s.size() && s[i] == c) {
            ++i;
            return true;
        }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail(std::string("expected '") + c + "'");
    }
    std::string str() {
        ws();
        if (i >= s.size() || (s[i] != '\'' && s[i] != '"')) fail("expected a string");
        const char q = s[i++];
        const std::size_t end = s.find(q, i);
        if (end == std::string::npos) fail("unterminated string");
        std::string out = s.substr(i, end - i);
        i = end + 1;
        return out;
    }
    bool boolean() {
        ws();
        if (s.compare(i, 4, "True") == 0) {
            i += 4;
            return true;
        }
        if (s.compare(i, 5, "False") == 0) {
            i += 5;
            return false;
        }
        fail("expected True or False");
    }
    std::size_t integer() {
        ws();
        if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
            fail("expected a non-negative integer");
        }
        std::size_t n = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            n = n * 10 + static_cast<std::size_t>(s[i++] - '0');
        }
        eat('L');  // a Python 2 long, as old writers spelled it
        return n;
    }
    std::vector<std::size_t> tuple() {
        std::vector<std::size_t> out;
        expect('(');
        while (!eat(')')) {
            out.push_back(integer());
            if (!eat(',')) {
                expect(')');
                break;
            }
        }
        return out;
    }
};

inline std::string shape_repr(const std::vector<std::size_t>& shape) {
    std::string out = "(";
    for (std::size_t k = 0; k < shape.size(); ++k) {
        if (k) out += ", ";
        out += std::to_string(shape[k]);
    }
    if (shape.size() == 1) out += ",";
    return out + ")";
}

}  // namespace detail

inline Array parse(const std::vector<unsigned char>& bytes, const std::string& where) {
    static const char magic[] = "\x93NUMPY";
    if (!host_little_endian()) throw std::runtime_error("npy: big-endian host");
    if (bytes.size() < 10 || std::memcmp(bytes.data(), magic, 6) != 0) {
        throw std::runtime_error(where + ": not an .npy file");
    }
    const unsigned major = bytes[6];
    std::size_t hlen = 0, at = 0;
    if (major == 1) {
        hlen = bytes[8] | (static_cast<std::size_t>(bytes[9]) << 8);
        at = 10;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < 12) throw std::runtime_error(where + ": truncated header");
        for (int k = 3; k >= 0; --k) hlen = (hlen << 8) | bytes[8 + k];
        at = 12;
    } else {
        throw std::runtime_error(where + ": npy version " + std::to_string(major) +
                                 " is not 1, 2 or 3");
    }
    if (bytes.size() < at + hlen) throw std::runtime_error(where + ": truncated header");
    const std::string text(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                           bytes.begin() + static_cast<std::ptrdiff_t>(at + hlen));

    Array a;
    bool have_descr = false, have_order = false, have_shape = false;
    detail::Literal lit{text};
    lit.expect('{');
    while (!lit.eat('}')) {
        const std::string key = lit.str();
        lit.expect(':');
        if (key == "descr") {
            a.descr = lit.str();
            have_descr = true;
        } else if (key == "fortran_order") {
            if (lit.boolean()) {
                throw std::runtime_error(where + ": fortran_order is True; pass "
                                         "np.ascontiguousarray(a)");
            }
            have_order = true;
        } else if (key == "shape") {
            a.shape = lit.tuple();
            have_shape = true;
        } else {
            lit.fail("unexpected key '" + key + "'");
        }
        if (!lit.eat(',')) {
            lit.expect('}');
            break;
        }
    }
    if (!have_descr || !have_order || !have_shape) {
        throw std::runtime_error(where + ": header lacks descr, fortran_order or shape");
    }
    const std::size_t want = a.count() * itemsize(a.descr);
    const std::size_t got = bytes.size() - at - hlen;
    if (got != want) {
        throw std::runtime_error(where + ": payload is " + std::to_string(got) +
                                 " bytes, shape and dtype say " + std::to_string(want));
    }
    a.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(at + hlen), bytes.end());
    return a;
}

inline Array read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(path + ": cannot open");
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),
                                           std::istreambuf_iterator<char>());
    return parse(bytes, path);
}

// np.save's header for a C-ordered array of `descr` and `shape`.
inline std::string header(const std::string& descr, const std::vector<std::size_t>& shape) {
    itemsize(descr);  // refuse what parse would refuse
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                       detail::shape_repr(shape) + ", }";
    // numpy's growth room: 21 digits for the first axis, less its own.
    if (!shape.empty()) dict.append(21 - std::to_string(shape[0]).size(), ' ');
    const std::size_t hlen = dict.size() + 1;  // with the closing newline
    const std::size_t pad = 64 - ((6 + 2 + 2 + hlen) % 64);
    if (hlen + pad > 0xffff) throw std::runtime_error("npy: header too long for 1.0");
    dict.append(pad, ' ');
    dict += '\n';
    const std::size_t total = dict.size();
    std::string out("\x93NUMPY\x01\x00", 8);
    out += static_cast<char>(total & 0xff);
    out += static_cast<char>((total >> 8) & 0xff);
    return out + dict;
}

inline void write(const std::string& path, const Array& a) {
    if (!host_little_endian()) throw std::runtime_error("npy: big-endian host");
    if (a.data.size() != a.count() * itemsize(a.descr)) {
        throw std::runtime_error(path + ": payload does not match shape and dtype");
    }
    const std::string head = header(a.descr, a.shape);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(head.data(), static_cast<std::streamsize>(head.size()));
    f.write(reinterpret_cast<const char*>(a.data.data()),
            static_cast<std::streamsize>(a.data.size()));
    if (!f) throw std::runtime_error(path + ": write failed");
}

}  // namespace npy
}  // namespace fj
