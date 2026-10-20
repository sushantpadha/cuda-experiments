#!/usr/bin/env python3
"""Two clients share one vmemd at the same time: make test-pair

Starts bin/vmemd --debug in a temp dir, runs two bin/pair_client processes against it (a and b, then a and c after b
is killed) and releases them step by step over stdin. After each phase it reads the broker's log and checks the
per-tenant and total counts; at the end it prints a summary per phase for a human to read.

Phases (first config 24 MiB device, 12 MiB host, 4 MiB context overhead per tenant; 4 MiB chunks):
  1 connect   a and b say Hello; each costs 4 MiB of device once
  2 share     a takes 12 MiB; b gets one device chunk, then spills to host; both see refused Allocs and Moves
  3 reuse     a frees a chunk; once the broker has handled that Free, b's next chunk lands in the freed room
  4 crash     b is killed with 8 MiB device and 12 MiB host held; the broker drops it and a reuses the room
  5 restart   c joins; the broker is killed, both make a chunk alone, a new broker starts with no device limit;
              both reconnect and resync, and the totals add up
  6 allocfail a fills the real VRAM until a chunk lands on host (AllocFail); then c's chunk also lands on host
  7 end       both free everything and exit; the broker's totals are zero
"""
import sys

import harness as h

CLIENT = h.binary("pair_client")
OVH = 4
CONF1 = f"vram_limit_mib = 24\ndram_limit_mib = 12\nctx_overhead_mib = {OVH}\n"
CONF2 = f"dram_limit_mib = 1024\nctx_overhead_mib = {OVH}\n"   # no device limit: CUDA itself runs out


def state_is(b, c, own, what):
    """the broker's latest counts for c equal own (device, host) in MiB, context overhead included"""
    got = h.wait_state(b, c.pid, own)
    h.expect(got is not None and got[:2] == own, f"broker counts {c.label} at {own[0]}/{own[1]} MiB: {what}" +
             ("" if got and got[:2] == own else f" (got {got})"))


def total_is(b, want, what):
    _, total, _ = h.replay(b.lines())
    h.expect(total == want, f"broker total {want[0]}/{want[1]} MiB: {what}" + ("" if total == want else f" (got {total})"))


def end_phase(b, start, names):
    lines = b.lines()
    h.snapshot(lines, names)
    h.events(lines, start)
    return len(lines)


def body(tmp):
    b = h.Broker(tmp, CONF1).start()
    names = {}

    h.phase("1 connect", "a and b connect; each Hello adds 4 MiB of context overhead, once")
    start = 0
    a = h.Client([CLIENT, "a", b.sock], "a")
    bb = h.Client([CLIENT, "b", b.sock], "b")
    a.until("hello")
    bb.until("hello")
    names.update({a.pid: "a", bb.pid: "b"})
    lines = b.lines()
    h.check_hello(lines, a.pid, "pair-a")
    h.check_hello(lines, bb.pid, "pair-b")
    state_is(b, a, (OVH, 0), "a's overhead only")
    state_is(b, bb, (OVH, 0), "b's overhead only")
    total_is(b, (2 * OVH, 0), "two overheads")
    start = end_phase(b, start, names)

    h.phase("2 share", "a takes 12 MiB, b spills to host after one chunk, then both are refused")
    a.go()
    a.until("b fills")
    bb.go()
    bb.until("a freed")
    a.go()
    a.until("b reuses")
    lines = b.lines()
    for c in (a, bb):
        n = sum(f"pid {c.pid}: " in l and "refused" in l for l in lines[start:])
        h.expect(n == 2, f"{c.label} got 2 refusals from the broker (one Alloc, one Move): {n}")
    state_is(b, a, (OVH + 8, 0), "a after its refusals and its Free")
    start = end_phase(b, start, names)

    h.phase("3 reuse", "a's Free reaches the broker, then b's next chunk takes that room on the device")
    # Free has no reply, so b's Alloc could overtake it on the other socket (daemon/README.md, cross-tenant
    # ordering). wait until the broker logged a's Free before releasing b
    h.expect(h.wait_log(b.log, lambda ls: any(f"free: pid {a.pid}," in l for l in ls)), "the broker handled a's Free")
    bb.go()
    bb.until("hold")
    state_is(b, bb, (OVH + 8, 12), "b's new chunk is on the device")
    total_is(b, (24, 12), "device and host limits met exactly")
    start = end_phase(b, start, names)

    h.phase("4 crash", "b is killed with 8 MiB device and 12 MiB host held; a keeps working")
    bb.kill()
    h.expect(h.wait_disconnect(b, bb.pid), "the broker saw b disconnect")
    total_is(b, (OVH + 8, 0), "only a's share is left")
    a.go()
    a.until("b killed")   # a was parked at "b reuses"; this releases it to the crash point
    a.go()
    a.until("c joined")
    start = end_phase(b, start, names)

    h.phase("5 restart", "c joins; the broker is killed, both clients work alone, then both reconnect to a new broker")
    c = h.Client([CLIENT, "c", b.sock], "c")
    c.until("kill")
    names[c.pid] = "c"
    h.check_hello(b.lines(), c.pid, "pair-c")
    a.go()
    a.until("kill")
    b.stop()
    for x in (a, c):
        x.go()
    a.until("restart")
    c.until("restart")
    b.start(CONF2)
    for x in (a, c):
        x.go()
    a.until("resynced")
    c.until("resynced")
    state_is(b, a, (OVH + 12, 0), "a's two old chunks and the one made alone")
    state_is(b, c, (OVH + 8, 4), "c's chunks, one evicted while alone")
    total_is(b, (2 * OVH + 20, 4), "the two resynced tenants add up")
    lines = b.lines()
    for x in (a, c):
        n = h.count(lines, f"resync: pid {x.pid},")
        h.expect(n == 3, f"{x.label} sent one Resync per live chunk: 3 (got {n})")
        hellos = h.count(lines, f'hello: pid {x.pid} is')
        h.expect(hellos == 2, f"{x.label}: one Hello per connection, 2 in all (got {hellos})")
    start = end_phase(b, start, names)

    h.phase("6 allocfail", "no device limit now: a fills the real VRAM, then c allocates too")
    c.go()
    c.until("a filled")   # c parks until a is done
    a.go()
    a.until("a filled")
    h.expect(h.wait_log(b.log, lambda ls: any(f"alloc_fail: pid {a.pid}," in l for l in ls)), "the broker logged a's AllocFail")
    c.go()
    c.until("c filled")
    h.expect(h.wait_log(b.log, lambda ls: any(f"alloc_fail: pid {c.pid}," in l for l in ls)), "the broker logged c's AllocFail")
    state_is(b, c, (OVH + 8, 260), "c's 256 MiB counted on host")
    start = end_phase(b, start, names)

    h.phase("7 end", "both free everything and exit")
    a.go()
    a.finish()
    c.go()
    c.finish()
    h.expect(h.wait_disconnect(b, a.pid) and h.wait_disconnect(b, c.pid), "the broker saw a and c disconnect")
    lines = b.lines()
    for x in (a, bb, c):
        h.check_views(lines, x, OVH)
    h.check_sums(lines)
    h.check_refusals(lines)
    h.check_totals_zero(lines)
    h.expect(b.alive(), "vmemd is still running")
    end_phase(b, start, names)
    b.stop()


if __name__ == "__main__":
    if not h.need(h.VMEMD, CLIENT):
        sys.exit(2)
    sys.exit(h.run(body, "vmemd-pair-", 300))
