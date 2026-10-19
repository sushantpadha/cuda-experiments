// libvmem backend interface
// - holds memory as chunks
// - one backend per allocator
// - a chunk is one allocation of any granularity multiple
// - backends never synchronize: the caller makes sure a chunk is idle before move and destroy
// - role of backend is purely communicating with cuda runtime/driver layer for memory

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace vmem {

// ---- shared types ----

enum class Loc { Device, Host };  // TODO can make more granular ie different NUMA nodes / GPUs can be allowed in v2/v3; also update used(Loc)
enum class Status { Ok, NoMemory, Unsupported, Disconnected, Fatal };   // used for error handling
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

struct BackendConfig {
    int device = 0;
    const char *daemon = nullptr;   // vmemd socket; ignored by backends without a daemon
};

// ---- generic interface ----

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
    virtual Status move(Chunk &, Loc to) = 0;   // same address; on failure the chunk is unchanged (address, data, loc)
    // TODO: non-blocking move = move_start(chunk, to, stream) + move_finish(chunk). the copy is stream-ordered, but the
    //   swap (cuMemUnmap + cuMemMap) is a host call: it cannot be put in a stream (cuLaunchHostFunc forbids CUDA calls),
    //   so an event does not make it safe. finish must run on the host before any kernel touches the chunk

    // admin
    virtual Link link() const = 0;
    virtual Status reconnect() = 0;
    virtual size_t used() const = 0;          // bytes in live chunks (sizes as created, rounded to granularity)
    virtual size_t used(Loc) const = 0;       // same, on one location; throws Error(Unsupported) if the backend cannot tell (e.g. uvm)
    virtual const std::string &last_error() const = 0;
};

}  // namespace vmem
