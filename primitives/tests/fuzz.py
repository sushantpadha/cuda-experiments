#!/usr/bin/env python3
# AI-generated (Claude). Model-based fuzz test for ./runner and vmem::Manager.
#
# Writes random .vm programs, predicts with a small model of the Manager's rules
# which lines must fail, runs each program through the runner and compares the
# runner's ok/ERROR per line with the prediction.
#
# usage: tests/fuzz.py [-n PROGRAMS] [-l LINES] [--seed S] [--runner bin/runner] [-o logs/fuzz]   (or: make fuzz)
import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

MiB = 1 << 20
OPS = (['reserve'] * 2 + ['create'] * 3 + ['map'] * 4 + ['unmap'] * 2 + ['remap'] * 3 + ['release'] * 2 +
       ['free'] + ['fill'] * 3 + ['check'] * 3 + ['va', 'on_device', 'info', 'loc'])


def generate(seed, length):
    """Returns (lines, expected_ok) for one random program."""
    rnd = random.Random(seed)
    res = {}      # reservation variable -> {'size', 'id'}; 'id' is unique, a reassigned variable is a new reservation
    freed = set() # reservation ids already freed: their address range may be reused, so never touched again
    chunks = {}   # chunk variable -> chunk state
    every = []    # every chunk ever created: reassigning a variable does not unmap the old chunk
    lines, ok = [], []

    def live_res():
        return [r for r, v in res.items() if v['id'] not in freed]

    def mapped_in(rid):
        return [c for c in every if not c['released'] and c['mapped'] and c['mapped'][0] == rid]

    for _ in range(length):
        op = rnd.choice(OPS)
        if op in ('map', 'free') and not live_res():
            op = 'reserve'
        if op not in ('reserve', 'create') and not chunks:
            op = 'create'

        if op == 'reserve':
            r, size = rnd.choice(['r0', 'r1', 'r2']), rnd.choice([4, 8, 16, 32]) * MiB
            res[r] = {'size': size, 'id': len(lines)}
            lines.append(f'{r} = reserve({size // MiB}M)'); ok.append(True)
            continue
        if op == 'create':
            c = rnd.choice(['c0', 'c1', 'c2', 'c3'])
            size, loc = rnd.choice([2, 4, 6]) * MiB, rnd.choice(['device', 'host'])
            chunks[c] = {'size': size, 'mapped': None, 'filled': False, 'released': False}
            every.append(chunks[c])
            lines.append(f'{c} = create({size // MiB}M, {loc})'); ok.append(True)
            continue

        c = rnd.choice(list(chunks))
        ch = chunks[c]
        alive = not ch['released']
        mapped = alive and ch['mapped'] is not None

        if op == 'map':
            r = rnd.choice(live_res())
            rid, rsize = res[r]['id'], res[r]['size']
            # past the end of r lies the next reservation (they are contiguous), so stay inside r;
            # a map that runs over the end must still fail
            off = rnd.randrange(0, rsize // MiB, 2) * MiB
            lines.append(f'map({c}, {r} + {off // MiB}M)' if off else f'map({c}, {r})')
            good = (alive and ch['mapped'] is None and off + ch['size'] <= rsize and
                    not any(off < o['mapped'][1] + o['size'] and o['mapped'][1] < off + ch['size'] for o in mapped_in(rid)))
            if good:
                ch['mapped'] = (rid, off)
        elif op == 'free':
            r = rnd.choice(live_res())
            lines.append(f'free({r})')
            good = not mapped_in(res[r]['id'])
            if good:
                freed.add(res[r]['id'])
        elif op == 'unmap':
            lines.append(f'unmap({c})')
            good = mapped
            if good:
                ch['mapped'] = None
        elif op == 'remap':
            lines.append(f'remap({c}, {rnd.choice(["device", "host"])})')
            good = mapped
        elif op == 'release':
            lines.append(f'release({c})')
            good = alive and ch['mapped'] is None
            if good:
                ch['released'] = True
        elif op in ('fill', 'check', 'va'):
            lines.append(f'{op}({c})')
            good = mapped and (op != 'check' or ch['filled'])
            if op == 'fill' and good:
                ch['filled'] = True
        else:   # info, loc, on_device
            lines.append(f'{op}({c})')
            good = alive
        ok.append(good)
    return lines, ok


def main():
    ap = argparse.ArgumentParser(description='Model-based fuzz test for the vmem action runner.')
    ap.add_argument('-n', type=int, default=50, help='number of programs (default 50)')
    ap.add_argument('-l', type=int, default=60, help='lines per program (default 60)')
    ap.add_argument('--seed', type=int, default=0, help='first seed (default 0)')
    ap.add_argument('--runner', default='bin/runner')
    ap.add_argument('-o', default='logs/fuzz', help='folder for the programs and logs (default logs/fuzz)')
    a = ap.parse_args()

    out = Path(a.o)
    out.mkdir(parents=True, exist_ok=True)
    line_re = re.compile(r'^\[ *(\d+)\] .{36} (ok|ERROR|SKIP)', re.M)
    total = bad = errors = 0
    for seed in range(a.seed, a.seed + a.n):
        lines, ok = generate(seed, a.l)
        vm = out / f'fuzz{seed}.vm'
        vm.write_text('\n'.join(lines) + '\n')
        p = subprocess.run([a.runner, '--stable', '-o', str(out), str(vm)], capture_output=True, text=True)
        if p.returncode not in (0, 1):
            print(f'seed {seed}: runner exited {p.returncode}\n{p.stderr}')
            bad += 1
            continue
        log = (out / f'fuzz{seed}.log').read_text()
        got = {int(n): st for n, st in line_re.findall(log)}
        for i, want in enumerate(ok, 1):
            total += 1
            errors += not want
            if got.get(i) != ('ok' if want else 'ERROR'):
                bad += 1
                if bad <= 10:
                    print(f'seed {seed} line {i}: {lines[i - 1]}  expected {"ok" if want else "ERROR"}, got {got.get(i)}')
    print(f'{total} lines in {a.n} programs ({errors} expected errors): {bad} disagreement(s)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
