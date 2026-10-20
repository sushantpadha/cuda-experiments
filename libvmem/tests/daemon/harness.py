"""Shared code for the vmemd checkers (state_check.py, pair_check.py, workers_check.py).

- Broker: one bin/vmemd --debug with its own socket, log and config in a temp dir
- Client: one test program that prints "view D H" lines and "WAIT x" when it needs the checker
- log helpers: the [DBG] state and totals lines, replayed to rebuild the broker's per-tenant counts
- expect/phase/summary: checks are grouped by phase and printed again at the end
- run(): temp dir, global timeout, Ctrl-C, killing every process group, keeping the files when a check failed
"""
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VMEMD = os.path.join(ROOT, "bin", "vmemd")

STATE = re.compile(r"\[DBG\] state: pid (\d+) device (\d+) MiB host (\d+) MiB, all tenants device (\d+) MiB host (\d+) MiB")
TOTALS = re.compile(r"\[DBG\] totals: all tenants device (\d+) MiB host (\d+) MiB")
HELLO = re.compile(r'hello: pid (\d+) is "([^"]*)"')
DISCONNECT = re.compile(r"disconnect: pid (\d+)")
REFUSED = re.compile(r"pid (\d+): .*refused")

fails = 0
phases = []   # [name, what, [(ok, check)], summary lines]
procs = []    # every Popen we started, killed by run() on the way out


def binary(name):
    return os.path.join(ROOT, "bin", name)


# ---- checks and phases ----


def phase(name, what=""):
    """start a group of checks; the summary prints them under this name"""
    phases.append([name, what, [], []])
    print(f"{name}" + (f": {what}" if what else ""))


def expect(ok, what):
    global fails
    ok = bool(ok)
    print(f"  {'ok' if ok else 'FAIL'}: {what}")
    fails += not ok
    if not phases:
        phase("checks")
    phases[-1][2].append((ok, what))
    return ok


def note(text):
    """a line for the summary of the current phase (counts, events)"""
    print(f"  {text}")
    if phases:
        phases[-1][3].append(text)


def summary():
    print("\n==== summary ====")
    for name, what, checks, notes in phases:
        bad = sum(not ok for ok, _ in checks)
        print(f"\n[{'FAIL' if bad else 'PASS'}] {name}" + (f": {what}" if what else ""))
        for n in notes:
            print(f"    {n}")
        for ok, c in checks:
            print(f"    {'PASS' if ok else 'FAIL'}  {c}")
    total = sum(len(p[2]) for p in phases)
    print(f"\n{total - fails} of {total} checks passed")


# ---- processes ----


def kill_group(proc, sig=signal.SIGKILL):
    """kill the process and everything it started (each runs in its own session)"""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    if sig == signal.SIGKILL:
        proc.wait()


def popen(cmd, **kw):
    p = subprocess.Popen(cmd, start_new_session=True, **kw)
    procs.append(p)
    return p


class Broker:
    """one vmemd with its own files; start() again restarts it on the same log, optionally with a new config"""

    def __init__(self, tmp, conf):
        self.sock = os.path.join(tmp, "vmemd.sock")   # a socket path holds at most 107 characters
        self.log = os.path.join(tmp, "vmemd.log")
        self.conf = os.path.join(tmp, "vmemd.toml")
        self.conf_text = conf
        self.proc = None

    def start(self, conf=None):
        if conf is not None:
            self.conf_text = conf
        with open(self.conf, "w") as f:
            f.write(self.conf_text)
        self.proc = popen([VMEMD, "--config", self.conf, "--socket", self.sock, "--log", self.log, "--debug"],
                          stderr=subprocess.DEVNULL)
        for _ in range(200):   # up to 2 s for the socket
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as s:
                try:
                    s.connect(self.sock)
                    return self
                except OSError:
                    time.sleep(0.01)
        raise RuntimeError("vmemd did not start")

    def stop(self):
        if self.proc:
            kill_group(self.proc)
            self.proc = None
        if os.path.exists(self.sock):
            os.unlink(self.sock)

    def pause(self):
        os.kill(self.proc.pid, signal.SIGSTOP)

    def resume(self):
        os.kill(self.proc.pid, signal.SIGCONT)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def lines(self):
        return read_log(self.log)


class Client:
    """a test client on stdin/stdout: it prints "view D H [ALL_D ALL_H]" after each checked step ("local ..." while
    it has no broker), "WAIT x" when it needs the checker, and its own "ok:"/"FAIL:" checks"""

    def __init__(self, cmd, label):
        self.label = label
        self.proc = popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.pid = self.proc.pid
        self.views = []   # (device, host) or (device, host, all device, all host), in MiB
        self.out = []
        self.reported = 0   # own checks already passed on to expect()

    def run_until_wait(self):
        """read output until the client asks for something; returns what it waits for, or None when it exited"""
        for line in self.proc.stdout:
            line = line.rstrip()
            self.out.append(line)
            if line.startswith("view "):
                self.views.append(tuple(int(x) for x in line.split()[1:]))
            elif line.startswith("WAIT "):
                self.report()
                return line[5:]
        self.report()
        return None

    def report(self):
        """pass the client's own ok/FAIL lines on as checks of the current phase"""
        for line in self.out[self.reported:]:
            s = line.lstrip()
            if s.startswith(("ok:", "FAIL")):
                text = s.split(": ", 1)[-1]
                expect(s.startswith("ok:"), text if text.startswith(self.label + ":") else f"{self.label}: {text}")
            elif line.startswith("  ") and s:   # the client's own remarks, e.g. a measured time
                note(f"{self.label}: {s}")
        self.reported = len(self.out)

    def until(self, want):
        """run the client to its next WAIT, which must be want (call go() first if it is waiting)"""
        got = self.run_until_wait()
        if not expect(got == want, f"{self.label} reached WAIT {want}" + ("" if got == want else f" (got {got})")):
            raise RuntimeError(f"{self.label} did not reach WAIT {want}")

    def go(self):
        try:
            self.proc.stdin.write("go\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            pass   # the client died; its exit code reports it

    def finish(self):
        self.run_until_wait()
        self.proc.wait()
        expect(self.proc.returncode == 0, f"{self.label}: exited with 0 (got {self.proc.returncode})")

    def kill(self):
        kill_group(self.proc)


# ---- log ----


def read_log(path):
    try:
        with open(path) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def wait_log(path, pred, timeout=10):
    """poll the log until pred(lines) holds; the broker writes line-buffered, so no fixed sleep is needed"""
    end = time.time() + timeout
    while True:
        lines = read_log(path)
        if pred(lines):
            return lines
        if time.time() > end:
            return None
        time.sleep(0.02)


def count(lines, text):
    return sum(text in line for line in lines)


def states(lines, pid):
    """the (device, host, all device, all host) states the broker reported for one tenant, repeats removed"""
    out = []
    for line in lines:
        m = STATE.search(line)
        if m and int(m.group(1)) == pid:
            st = tuple(int(x) for x in m.groups()[1:])
            if not out or out[-1] != st:
                out.append(st)
    return out


def latest(lines, pid):
    s = states(lines, pid)
    return s[-1] if s else None


def wait_state(broker, pid, own, timeout=10):
    """wait until the broker's latest state for pid has these own (device, host) MiB; reply-less messages (Free,
    Resync) may still be queued when the client prints its view, so a fixed sleep would race"""
    lines = wait_log(broker.log, lambda ls: (latest(ls, pid) or ())[:2] == tuple(own), timeout)
    return latest(lines or broker.lines(), pid)


def wait_disconnect(broker, pid, timeout=10, already=0):
    """wait until the broker has logged more than `already` disconnects of pid, and the totals line that follows
    each one. already matters when the pid is this script's own: Broker.start leaves one probe disconnect behind"""
    key = f"disconnect: pid {pid} "

    def done(lines):
        seen = 0
        for i, line in enumerate(lines):
            if key in line:
                seen += 1
                if seen > already:
                    return any("[DBG] totals:" in x for x in lines[i + 1:i + 3])
        return False

    return wait_log(broker.log, done, timeout) is not None


def replay(lines):
    """rebuild the broker's per-tenant counts from the log. returns ({pid: [name, device, host]}, (all device, all
    host), errors). errors lists every state or totals line where the global count is not the sum over live tenants
    (K2). a restart ("listening on") clears everything; a disconnect drops the tenant"""
    tenants, total, errors = {}, (0, 0), []
    for line in lines:
        if "listening on" in line:
            tenants, total = {}, (0, 0)
        elif m := HELLO.search(line):
            tenants.setdefault(int(m.group(1)), ["?", 0, 0])[0] = m.group(2)
        elif m := DISCONNECT.search(line):
            tenants.pop(int(m.group(1)), None)   # the totals line right after carries the new sum
        elif m := STATE.search(line):
            pid, d, h, ad, ah = (int(x) for x in m.groups())
            t = tenants.setdefault(pid, ["?", 0, 0])
            t[1], t[2] = d, h
            total = (ad, ah)
        elif m := TOTALS.search(line):
            total = tuple(int(x) for x in m.groups())
        if not (STATE.search(line) or TOTALS.search(line)):
            continue
        s = (sum(t[1] for t in tenants.values()), sum(t[2] for t in tenants.values()))
        if s != total:
            errors.append(f"{line.strip()} (tenants add up to {s[0]} MiB device, {s[1]} MiB host)")
    return tenants, total, errors


def snapshot(lines, names=None):
    """summary line: each live tenant's counts and the totals, as the broker has them now"""
    tenants, total, _ = replay(lines)
    parts = [f"{(names or {}).get(pid, t[0])} {t[1]}/{t[2]}" for pid, t in sorted(tenants.items())]
    note(f"broker counts (device/host MiB): {', '.join(parts) or 'no tenants'}; total {total[0]}/{total[1]}")
    return total


def events(lines, start=0):
    """summary line: what the broker logged since line start"""
    ls = lines[start:]
    note(f"broker events: {count(ls, 'refused')} refused, {count(ls, 'alloc_fail:')} alloc_fail, "
         f"{count(ls, 'resync:')} resync, {count(ls, 'over limit')} over limit, {count(ls, 'disconnect:')} disconnect")


# ---- generic checks ----


def reached_in_order(views, sts, overhead):
    """every view appears among the broker's states, in order. the broker's device count for a tenant includes its
    context overhead, so it is added to the view. returns the first view not found, or None"""
    i = 0
    for v in views:
        while i < len(sts):
            st = sts[i]
            i += 1
            if st[0] == v[0] + overhead and st[1] == v[1] and (len(v) < 4 or (st[2], st[3]) == (v[2], v[3])):
                break
        else:
            return v
    return None


def check_views(lines, client, overhead):
    """the broker passed through every state the client saw while connected (K1: its counts follow each call)"""
    views = [v for i, v in enumerate(client.views) if i == 0 or client.views[i - 1] != v]
    missing = reached_in_order(views, states(lines, client.pid), overhead)
    expect(missing is None, f"{client.label}: the broker reached all {len(views)} states the client reported, in order" +
           ("" if missing is None else f" (not found: {missing})"))
    # a match in order can hit a stale state (e.g. one from a connection the client already dropped); the last
    # state cannot: messages on one socket are handled in order, so it follows everything the client sent
    if views:
        last, want = latest(lines, client.pid), (views[-1][0] + overhead, views[-1][1])
        expect(last is not None and last[:2] == want, f"{client.label}: the broker's last counts match the client's last "
               f"view, {want[0]}/{want[1]} MiB" + ("" if last and last[:2] == want else f" (got {last})"))


def check_totals_zero(lines):
    totals = [tuple(int(x) for x in m.groups()) for m in map(TOTALS.search, lines) if m]
    expect(bool(totals) and totals[-1] == (0, 0), "all clients gone: the broker's totals are 0 MiB device, 0 MiB host" +
           ("" if totals and totals[-1] == (0, 0) else f" (got {totals[-1] if totals else 'no totals line'})"))


def check_sums(lines):
    """K2: after every message the global counts equal the sum of the live tenants' counts"""
    errors = replay(lines)[2]
    expect(not errors, "the broker's totals equal the sum over its tenants after every message" +
           ("" if not errors else f" (first mismatch: {errors[0]})"))


def check_refusals(lines):
    """K3/B3: a refused Alloc or Move leaves the tenant's counts as they were. the broker logs the refusal while
    handling the message and writes the state line right after, so the next own (device, host) for that pid must
    equal the one before. the totals are not compared: other tenants may have changed them in between"""
    last, pending, bad, n = {}, set(), [], 0
    for line in lines:
        if "listening on" in line:
            last, pending = {}, set()
        elif m := REFUSED.search(line):
            pending.add(int(m.group(1)))
            n += 1
        elif m := STATE.search(line):
            pid, st = int(m.group(1)), (int(m.group(2)), int(m.group(3)))
            if pid in pending and last.get(pid) != st:
                bad.append(f"pid {pid}: {last.get(pid)} -> {st}")
            pending.discard(pid)
            last[pid] = st
    expect(not bad, f"each of the {n} refused requests left the counts unchanged" + (f" ({bad[0]})" if bad else ""))
    return n


def check_hello(lines, pid, name):
    expect(any(f'hello: pid {pid} is "{name}"' in l for l in lines), f'the broker logged the Hello name "{name}" for pid {pid}')


def check_overhead_once(lines, pid, overhead):
    """the first state after a tenant's Hello is the context overhead alone"""
    sts = states(lines, pid)
    expect(bool(sts) and sts[0][:2] == (overhead, 0), f"pid {pid}: first state is the context overhead, {overhead} MiB, counted once" +
           ("" if sts and sts[0][:2] == (overhead, 0) else f" (got {sts[:1]})"))


# ---- main ----


def need(*paths):
    missing = [p for p in paths if not os.access(p, os.X_OK)]
    for p in missing:
        print(f"FAIL: {p} is missing (build it with make)")
    return not missing


def run(body, prefix, timeout):
    """run body(tmp) in a fresh temp dir under /tmp. kills every process group on return, error, Ctrl-C or after
    timeout seconds; removes the dir unless a check failed; prints the summary; returns the exit code"""
    global fails
    tmp = tempfile.mkdtemp(prefix=prefix, dir="/tmp")

    def on_alarm(*_):
        raise TimeoutError(f"no result after {timeout} s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(timeout)
    rc = None
    try:
        body(tmp)
    except KeyboardInterrupt:
        print("interrupted")
        fails += 1
        rc = 130
    except Exception as e:   # a stuck or crashed client: report it as a failed check, then clean up
        expect(False, f"the run finished ({type(e).__name__}: {e})")
    finally:
        signal.alarm(0)
        for p in procs:
            kill_group(p)
            if p.stdin:
                p.stdin.close()
            if p.stdout:
                p.stdout.close()
        summary()
        if fails:
            print(f"files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {fails} check(s)")
    return rc or (1 if fails else 0)


if __name__ == "__main__":
    sys.exit("harness.py is a module; run state_check.py, pair_check.py or workers_check.py")
