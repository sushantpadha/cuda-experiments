// vmemd wire protocol, shared by the vmem backend and daemon/vmemd.cpp
// - we use UDS, with SOCK_SEQPACKET: one message per packet, delivered in order
// - the client blocks on replies; messages without a reply never get one
// - the chunk id is the chunk pointer; the backend reserves the address range before ALLOC, so it is known up front

#pragma once

#include <cstdint>

#include "backend.h"

namespace vmem::proto {

constexpr const char *DEFAULT_SOCKET = "/tmp/vmemd.sock";
constexpr size_t NAME_LEN = 32;

// ---- client to daemon ----

// * defines all relevant request and reply types, authoritative!
enum class Type : uint32_t {
    Hello,      // name                      -> Reply; first message on a connection
    Alloc,      // ptr, size                 -> Reply{Ok, loc} or Reply{NoMemory}; the daemon counts it at loc at once
    AllocFail,  // ptr                       no reply; device create failed, chunk went to host: count it there
    Free,       // ptr                       no reply
    Move,       // ptr, loc (target)         -> Reply{Ok} or Reply{NoMemory}; the daemon reserves the target bytes
    MoveDone,   // ptr, loc (where it is)    no reply; loc == target on success, the old loc on failure
    Resync,     // ptr, size, loc            no reply; one per live chunk after HELLO on a reconnect
};

// single generic struct
struct Msg {
    Type type;
    Loc loc;              // Move, MoveDone, Resync
    uint64_t ptr;         // every type but Hello
    uint64_t size;        // Alloc, Resync
    char name[NAME_LEN];  // Hello; optional, null-terminated
};

// ---- daemon to client ----

struct Reply {
    Status status;  // Ok or NoMemory
    Loc loc;        // Alloc only
};

// static assert
static_assert(sizeof(Msg) == 56 && sizeof(Reply) == 8, "wire layout changed");

}  // namespace vmem::proto
