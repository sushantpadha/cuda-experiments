// runner log: log files, --stable rewriting, the per-line block format
#pragma once

#include <format>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace runner {

// one log file, flushed after every write so a crash keeps every finished line;
// with stable, addresses become va#1, va#2, ... (in order of appearance) and VRAM numbers become -
class LogFile {
public:
    LogFile(const std::string &path, bool stable) : f_(path), stable_(stable) {}
    bool good() const { return f_.good(); }
    // writes s, returns what was written
    std::string put(const std::string &s) {
        std::string t = stable_ ? stabilize(s) : s;
        f_ << t << std::flush;
        return t;
    }

private:
    std::string stabilize(const std::string &in) {
        static const std::regex addr("0x[0-9a-f]{6,}"), vram("VRAM free [0-9]+");
        std::string s = std::regex_replace(in, vram, "VRAM free -"), out;
        size_t last = 0;
        for (std::sregex_iterator it(s.begin(), s.end(), addr), end; it != end; ++it) {
            auto [p, fresh] = ids_.emplace(it->str(), ids_.size() + 1);
            out += s.substr(last, it->position() - last) + std::format("va#{}", p->second);
            last = it->position() + it->length();
        }
        return out + s.substr(last);
    }

    std::ofstream f_;
    bool stable_;
    std::map<std::string, size_t> ids_;   // address -> its va# number
};

inline std::string indent(const std::string &s) {
    std::string r;
    std::istringstream in(s);
    for (std::string l; std::getline(in, l);) r += "       | " + l + "\n";
    return r;
}

// one log block: [line] statement status time result, then the Manager's output indented
inline std::string block(const std::string &tag, const std::string &text, const char *status, const std::string &time,
                         const std::string &result, const std::string &out) {
    std::string head = std::format("[{:>4}] {:<36} {:<5} {:>13}  {}", tag, text, status, time, result);
    head.erase(head.find_last_not_of(' ') + 1);
    return head + "\n" + indent(out);
}

// "<prefix>key      value" lines, for the log header ("# ") and the terminal ("")
using KeyValues = std::vector<std::pair<std::string, std::string>>;
inline std::string keyvalues(const KeyValues &kv, const std::string &prefix) {
    std::string h;
    for (auto &[k, v] : kv) h += std::format("{}{:<8} {}\n", prefix, k, v);
    return h;
}

}  // namespace runner
