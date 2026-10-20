#!/usr/bin/env python3
"""vmemd state test: make test-state

Each scenario gets a fresh bin/vmemd --debug with its own config, socket and log, runs bin/state_client against it
(or, for "raw", talks to the broker directly over the socket), and then reads the log. The [DBG] lines ("state:"
after every message, "totals:" after a disconnect) carry the broker's counts. Every scenario with a broker checks:
  - every state the client reported ("view D H") is a state the broker reached, in the same order;
  - after every message the broker's totals equal the sum over its tenants;
  - every refused request left the counts unchanged;
  - the Hello name is in the log, and the broker's totals are back to zero at the end.
The "config" scenario starts vmemd with four bad config files and checks that it exits with an error and says why.
Some scenarios also check that the log has given lines (a refusal, an AllocFail, the Resync count).
"""
import os
import socket
import struct
import sys

import harness as h

CLIENT = h.binary("state_client")
LIMITS = "vram_limit_mib = 16\ndram_limit_mib = 8\n"


def sub(tmp, name):
    """one directory per scenario, so each broker log holds one scenario"""
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    return d


def common_checks(lines, clients, overhead=0):
    for c in clients:
        h.check_views(lines, c, overhead)
        h.check_hello(lines, c.pid, f"state-{c.label}")
    h.check_sums(lines)
    h.check_refusals(lines)
    h.check_totals_zero(lines)


def single(tmp, scenario, conf, what, overhead=0, on_wait=None, log_text=()):
    h.phase(scenario, what)
    b = h.Broker(sub(tmp, scenario), conf).start()
    try:
        c = h.Client([CLIENT, scenario, b.sock], scenario)
        while (w := c.run_until_wait()) is not None:
            on_wait(b, c, w)
            c.go()
        c.finish()
        h.expect(h.wait_disconnect(b, c.pid), "the broker saw the client disconnect")
    finally:
        b.stop()
    lines = b.lines()
    common_checks(lines, [c], overhead)
    for text in log_text:
        h.expect(h.count(lines, text) > 0, f'log has "{text}"')
    return lines, c


def restart(b, c, _):
    b.stop()
    b.start()


def stop_resume(b, c, w):
    if w == "stop":
        b.pause()
    else:
        b.resume()
        # the old connection must be gone before the client reconnects with the same pid
        h.expect(h.wait_disconnect(b, c.pid), "after SIGCONT the broker dropped the timed-out connection")


def tenants(tmp):
    h.phase("tenants", "one client holds 8 MiB of the 16 MiB device while a second fills the rest")
    b = h.Broker(sub(tmp, "tenants"), LIMITS).start()
    try:
        a = h.Client([CLIENT, "hold", b.sock], "hold")
        a.until("other client")      # a holds 8 MiB of the 16 MiB device
        f = h.Client([CLIENT, "fill", b.sock], "fill")
        f.finish()                   # fill sees the other client's memory in the totals
        a.go()
        a.finish()
        h.wait_disconnect(b, a.pid)
    finally:
        b.stop()
    lines = b.lines()
    common_checks(lines, [a, f])
    h.expect(h.count(lines, "alloc of 4 MiB refused") > 0, "log has the refused alloc")


def nobroker(tmp):
    h.phase("nobroker", "no broker listens: a missing socket file, then a stale one")
    missing = os.path.join(tmp, "none.sock")
    h.Client([CLIENT, "nobroker", missing], "nobroker (missing socket)").finish()
    stale = os.path.join(tmp, "stale.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as s:
        s.bind(stale)   # the file stays after close, and connect gets ECONNREFUSED
    h.Client([CLIENT, "nobroker", stale], "nobroker (stale socket)").finish()


def config(tmp):
    h.phase("config", "vmemd refuses to start on a missing config file, an unknown key, a value below -1 and text after the number")
    d = sub(tmp, "config")
    cases = [("missing file", None, "none.toml"),
             ("unknown key", "vram_limit_mib = 16\nfoo_mib = 1\n", "unknown key foo_mib"),
             ("value below -1", "vram_limit_mib = -2\n", 'expected "key = number"'),
             ("text after the number", "vram_limit_mib = 16 MiB\n", 'expected "key = number"')]
    for what, text, want in cases:
        conf = os.path.join(d, "bad.toml") if text is not None else os.path.join(d, "none.toml")
        if text is not None:
            with open(conf, "w") as f:
                f.write(text)
        log = os.path.join(d, "bad.log")
        if os.path.exists(log):
            os.unlink(log)
        r = h.subprocess.run([h.VMEMD, "--config", conf, "--socket", os.path.join(d, "bad.sock"), "--log", log],
                             stderr=h.subprocess.DEVNULL, timeout=5)
        h.expect(r.returncode != 0, f"{what}: vmemd exits with an error (got {r.returncode})")
        h.expect(want in " ".join(h.read_log(log)), f'{what}: log has "{want}"')


# ---- raw: broker edge cases, sent by hand over the socket ----

MSG = struct.Struct("<IiQQ32s")   # src/protocol.h Msg: type, loc, ptr, size, name (56 bytes)
REPLY = struct.Struct("<ii")      # Reply: status, loc
HELLO, ALLOC, ALLOCFAIL, FREE, MOVE, MOVEDONE, RESYNC = range(7)
DEV, HOST = 0, 1
OK, NOMEM, UNSUP = 0, 1, 2
M = 1 << 20


class Raw:
    """a hand-made tenant; every message the broker handles writes exactly one state line"""

    def __init__(self, b):
        self.b, self.n = b, 0
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.s.settimeout(5)
        self.s.connect(b.sock)

    def send(self, t, ptr=0, mib=0, loc=DEV, name=b""):
        self.s.send(MSG.pack(t, loc, ptr, mib * M, name))
        self.n += 1

    def ask(self, *a, **k):
        self.send(*a, **k)
        return REPLY.unpack(self.s.recv(REPLY.size))

    def state(self):
        """own (device, host) MiB once the broker has handled every message sent so far"""
        lines = h.wait_log(self.b.log, lambda ls: sum(f"state: pid {os.getpid()} " in l for l in ls) >= self.n, 5)
        return h.latest(lines or [], os.getpid())[:2] if lines else None


def raw(tmp):
    h.phase("raw", "edge cases from daemon/README.md, sent by a hand-made client (limits 16/8 MiB, overhead 4 MiB)")
    b = h.Broker(sub(tmp, "raw"), LIMITS + "ctx_overhead_mib = 4\n").start()
    try:
        r = Raw(b)
        p1, p2, p3, p4, unknown = 0x1000000, 0x2000000, 0x3000000, 0x4000000, 0x9000000
        h.expect(r.ask(HELLO, name=b"raw")[0] == OK and r.state() == (4, 0), "Hello: Ok, context overhead 4 MiB")
        h.expect(r.ask(HELLO, name=b"raw")[0] == OK and r.state() == (4, 0), "second Hello on one connection: overhead not added again")
        h.expect(r.ask(ALLOC, p1, 4) == (OK, DEV) and r.state() == (8, 0), "Alloc: device")
        h.expect(r.ask(ALLOC, p1, 4)[0] == UNSUP and r.state() == (8, 0), "Alloc for a pointer it already holds: refused, counts unchanged")
        r.send(FREE, unknown)
        r.send(ALLOCFAIL, unknown)
        r.send(MOVEDONE, unknown, loc=HOST)
        h.expect(r.ask(MOVE, unknown, loc=HOST)[0] == NOMEM and r.state() == (8, 0),
                 "Free, AllocFail, MoveDone for an unknown pointer ignored; Move for it refused")
        h.expect(r.ask(ALLOC, p2, 4) == (OK, DEV) and r.state() == (12, 0), "Alloc: device")
        h.expect(r.ask(MOVE, p1, loc=HOST)[0] == OK and r.state() == (12, 4), "Move granted: target bytes reserved, source still counted")
        h.expect(r.ask(MOVE, p2, loc=HOST)[0] == NOMEM and r.state() == (12, 4), "second Move while one is in flight: refused")
        r.send(MOVEDONE, p1, loc=DEV)
        h.expect(r.state() == (12, 0), "MoveDone at the old location (failed move): reservation released, chunk stays")
        h.expect(r.ask(MOVE, p1, loc=HOST)[0] == OK and r.state() == (12, 4), "Move granted again after the failed one")
        r.send(FREE, p1)
        h.expect(r.state() == (8, 0), "Free during a move: chunk and reservation both dropped")
        h.expect(r.ask(MOVE, p2, loc=HOST)[0] == OK, "Move after the dropped one: granted")
        r.send(MOVEDONE, p2, loc=HOST)
        h.expect(r.state() == (4, 4), "MoveDone at the target: chunk counted on host only")
        r.send(MOVEDONE, p2, loc=DEV)
        h.expect(r.state() == (8, 0), "MoveDone without a grant: the broker records where the tenant says it is")
        r.send(RESYNC, p2, 4, loc=HOST)
        h.expect(r.state() == (4, 4), "Resync for a tracked pointer: overwritten, not counted twice")
        r.send(RESYNC, p3, 8, loc=HOST)
        h.expect(r.state() == (4, 12), "Resync past the host limit: accepted (the tenant holds the memory)")
        h.expect(r.ask(ALLOC, p4, 4) == (OK, DEV) and r.state() == (8, 12), "Alloc: device")
        r.send(ALLOCFAIL, p4)
        h.expect(r.state() == (4, 16), "AllocFail: 4 MiB moved from device to host, even over the host limit")
        r.send(ALLOCFAIL, p4)
        h.expect(r.state() == (4, 16), "second AllocFail for a host chunk: ignored")
        before = h.count(b.lines(), f"disconnect: pid {os.getpid()} ")   # the broker start-up probe counts as one
        r.s.send(b"x" * 10)   # wrong size: the broker drops the tenant
        h.expect(h.wait_disconnect(b, os.getpid(), already=before), "a message of the wrong size drops the tenant")
        r.s.close()
    finally:
        b.stop()
    lines = b.lines()
    h.check_sums(lines)
    h.check_refusals(lines)
    h.check_totals_zero(lines)
    for text in ["already holds; refused", "free for unknown chunk", "alloc_fail for unknown chunk",
                 "move_done for unknown chunk", "move for unknown chunk", "is still moving; refused",
                 "failed, stays on device", "while it is moving; dropping the move", "without a granted move",
                 "already tracked; overwriting", "over limit on host", "already on host; ignored", "-byte message"]:
        h.expect(h.count(lines, text) > 0, f'log has "{text}"')


def main(tmp):
    lines, c = single(tmp, "simple", LIMITS + "ctx_overhead_mib = 4\n",
                      "large chunks, then small objects, one call at a time", overhead=4)
    h.check_overhead_once(lines, c.pid, 4)
    single(tmp, "multi", LIMITS, "fill device then host, hit both limits, move chunks both ways",
           log_text=["alloc of 4 MiB refused", "move of 4 MiB to host refused", "move of 4 MiB to device refused"])
    lines, c = single(tmp, "resync", "vram_limit_mib = 20\ndram_limit_mib = 8\nctx_overhead_mib = 4\n",
                      "the broker is killed and restarted mid-run; the client works alone, then reconnects",
                      overhead=4, on_wait=restart)
    n = h.count(lines, f"resync: pid {c.pid},")
    h.expect(n == 3, f"reconnect sent one Resync per live chunk: 3 (got {n})")
    h.expect(h.count(lines, f'hello: pid {c.pid} is "state-resync"') == 2, "Hello sent once per connection: 2")
    h.check_overhead_once(lines, c.pid, 4)
    lines, c = single(tmp, "autoreconnect", LIMITS, "like resync, but the client never calls reconnect(): auto_reconnect does it at the next create",
                      on_wait=restart)
    n = h.count(lines, f"resync: pid {c.pid},")
    h.expect(n == 3, f"the automatic reconnect sent one Resync per live chunk: 3 (got {n})")
    single(tmp, "allocfail", "dram_limit_mib = 2048\n", "no device limit: VRAM really fills, the next chunk goes to host",
           log_text=["alloc_fail: pid"])
    tenants(tmp)
    single(tmp, "hang", LIMITS, "the broker is stopped with SIGSTOP mid-run, then resumed", on_wait=stop_resume)
    nobroker(tmp)
    config(tmp)
    raw(tmp)


if __name__ == "__main__":
    if not h.need(h.VMEMD, CLIENT):
        sys.exit(2)
    sys.exit(h.run(main, "vmemd-state-", 600))
