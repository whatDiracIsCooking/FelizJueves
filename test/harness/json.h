// json.h -- the JSON the runner's launch spec is written in (#649).
//
// Just enough to read what harness_driver.py writes with `json.dumps`: null,
// true/false, numbers, strings (with the standard escapes; a \u escape only
// below U+0080, since every name the spec carries is an identifier or a
// path), arrays and objects. A number keeps its SPELLING, so an integer is
// read with strtoll and a double with strtod from the same text -- Python's
// float repr round-trips through strtod exactly, and a 64-bit integer never
// passes through a double. Anything else is a parse error naming the offset.

#pragma once

#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fj {
namespace json {

struct Value {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    std::string text;               // a Number's spelling, or a String's value
    std::vector<std::string> keys;  // an Object's keys, in order
    std::vector<Value> items;       // an Array's items, or an Object's values

    bool is(Kind k) const { return kind == k; }

    const Value* find(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        for (std::size_t k = 0; k < keys.size(); ++k) {
            if (keys[k] == key) return &items[k];
        }
        return nullptr;
    }
    const Value& at(const std::string& key) const {
        const Value* v = find(key);
        if (v == nullptr) throw std::runtime_error("json: missing key '" + key + "'");
        return *v;
    }
    const std::string& str() const {
        if (kind != Kind::String) throw std::runtime_error("json: expected a string");
        return text;
    }
    double number() const {
        if (kind != Kind::Number) throw std::runtime_error("json: expected a number");
        return std::strtod(text.c_str(), nullptr);
    }
    long long integer() const {
        if (kind != Kind::Number) throw std::runtime_error("json: expected a number");
        errno = 0;
        char* end = nullptr;
        const long long v = std::strtoll(text.c_str(), &end, 10);
        if (errno != 0 || end == nullptr || *end != '\0') {
            throw std::runtime_error("json: '" + text + "' is not a 64-bit integer");
        }
        return v;
    }
};

namespace detail {

struct Parser {
    const std::string& s;
    std::size_t i = 0;

    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("json: " + why + " at offset " + std::to_string(i));
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
    void word(const char* w) {
        const std::string want(w);
        if (s.compare(i, want.size(), want) != 0) fail("expected " + want);
        i += want.size();
    }

    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            const char c = s[i++];
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size()) fail("unterminated escape");
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (i + 4 > s.size()) fail("short \\u escape");
                    const unsigned long u = std::strtoul(s.substr(i, 4).c_str(), nullptr, 16);
                    if (u >= 0x80) fail("\\u escape beyond ASCII");
                    out += static_cast<char>(u);
                    i += 4;
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    Value value() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        Value v;
        const char c = s[i];
        if (c == '{') {
            v.kind = Value::Kind::Object;
            ++i;
            if (eat('}')) return v;
            do {
                v.keys.push_back(string());
                expect(':');
                v.items.push_back(value());
            } while (eat(','));
            expect('}');
        } else if (c == '[') {
            v.kind = Value::Kind::Array;
            ++i;
            if (eat(']')) return v;
            do {
                v.items.push_back(value());
            } while (eat(','));
            expect(']');
        } else if (c == '"') {
            v.kind = Value::Kind::String;
            v.text = string();
        } else if (c == 't') {
            word("true");
            v.kind = Value::Kind::Bool;
            v.boolean = true;
        } else if (c == 'f') {
            word("false");
            v.kind = Value::Kind::Bool;
        } else if (c == 'n') {
            word("null");
        } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            v.kind = Value::Kind::Number;
            const std::size_t start = i;
            while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) ||
                                    s[i] == '-' || s[i] == '+' || s[i] == '.' ||
                                    s[i] == 'e' || s[i] == 'E')) {
                ++i;
            }
            v.text = s.substr(start, i - start);
        } else {
            fail(std::string("unexpected '") + c + "'");
        }
        return v;
    }
};

}  // namespace detail

inline Value parse(const std::string& text) {
    detail::Parser p{text};
    Value v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing text");
    return v;
}

inline Value read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(path + ": cannot open");
    return parse(std::string((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>()));
}

}  // namespace json
}  // namespace fj
