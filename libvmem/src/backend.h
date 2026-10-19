// libvmem backend interface: chunk-level memory, one backend per allocator
// a chunk is one allocation of any granularity multiple; the backend picks where it lives
// backends never synchronize: the caller makes sure a chunk is idle before move and destroy
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace vmem {

// ---- shared types ----

enum class Loc { Device, Host };
enum class Status { Ok, NoMemory, Unsupported, Disconnected, Fatal };
enum class Link { Connected, Local, Broken };   // daemon state; backends without one are always Local

inline const char *loc_name(Loc l) { return l == Loc::Device ? "device" : "host"; }
inline const char *status_name(Status s) {
    switch (s) {
    case Status::Ok: return "ok";
    case Status::NoMemory: return "no memory";
    case Status::Unsupported: return "unsupported";
    case Status::Disconnected: return "disconnected";
    default: return "fatal";
    }
}
inline const char *link_name(Link l) {
    return l == Link::Connected ? "connected" : l == Link::Local ? "local" : "broken";
}

struct Error : std::runtime_error {
    Status status;
    Error(Status s, const std::string &what) : std::runtime_error(what), status(s) {}
};

struct Chunk {
    void *ptr = nullptr;
    size_t size = 0;
    Loc loc = Loc::Device;
};

struct Usage {
    size_t device = 0, host = 0;   // bytes this process holds
};

struct BackendConfig {
    int device = 0;
    const char *daemon = nullptr;   // vmemd socket; ignored by backends without a daemon
};

// ---- interface ----

struct Backend {
    virtual ~Backend() = default;

    // lifecycle; destroy every chunk before close
    virtual Status open(const BackendConfig &) = 0;
    virtual void close() noexcept = 0;
    virtual const char *name() const = 0;
    virtual size_t granularity() const = 0;   // chunk sizes are multiples of this
    virtual bool can_move() const = 0;

    // memory
    virtual Status create(size_t size, Chunk &out) = 0;
    virtual Status destroy(Chunk &) = 0;
    virtual Status move(Chunk &, Loc to) = 0;   // same address; libvmem calls it to carry out hints

    // admin
    virtual Link link() const = 0;
    virtual Status reconnect() = 0;
    virtual Usage usage() const = 0;
    virtual const std::string &last_error() const = 0;
};

}  // namespace vmem
