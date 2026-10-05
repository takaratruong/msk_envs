#!/usr/bin/env python3
"""Detached jobs with script snapshots and identity-checked orphan cleanup.

Process identity is Linux boot ID + /proc start time + PID. Each attempt also
gets an inherited environment nonce so descendants remain attributable after
the worker dies, including the interval before it records child_pid. Signals
use pidfds rather than a numeric PGID, preventing PID/group reuse races. A live
owned process blocks restart; a stale PID reused by another job does not.
Legacy attempts can be controlled while their exact worker command is live;
an orphan lacking either recorded identity or an attempt nonce is not signalled.
"""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time


TOKEN_ENV = 'REAL2SIM_JOB_ATTEMPT_TOKEN'
BOOT_ID = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def process_identity(pid):
    try:
        path = Path(f'/proc/{int(pid)}')
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return dict(pid=int(pid), start_ticks=int(fields[19]), pgid=int(fields[2]),
                    sid=int(fields[3]), boot_id=BOOT_ID, uid=path.stat().st_uid)
    except (OSError, ValueError, IndexError, TypeError):
        return None


def same_process(expected, observed):
    return bool(expected and observed and all(expected.get(k) == observed.get(k)
                for k in ('pid', 'start_ticks', 'boot_id')))


def alive(info):
    observed = process_identity(info.get('pid')) if info else None
    if info and info.get('worker_identity'):
        return same_process(info['worker_identity'], observed)
    # Compatibility for old receipts: require exact argv tokens and a session
    # leader, not the old substring search in arbitrary process command lines.
    try:
        argv = Path(f"/proc/{info['pid']}/cmdline").read_bytes().split(b'\0')
        return bool(observed and observed['sid'] == observed['pgid'] == info['pid']
                    and any(argv[i] == b'_worker' and argv[i + 1] == os.fsencode(info['attempt'])
                            for i in range(len(argv) - 1)))
    except (OSError, KeyError, TypeError):
        return False


def group_members(pgid):
    return {p['pid']: str(p['start_ticks']) for path in Path('/proc').glob('[0-9]*')
            if (p := process_identity(path.name)) and p['pgid'] == pgid}


def has_attempt_token(pid, token):
    if not token:
        return False
    try:
        return os.fsencode(TOKEN_ENV + '=' + token) in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    except OSError:
        return False


def owned_processes(info, state, config, remembered=None):
    """Identify live anchors first, then their current session/process groups."""
    table = {p['pid']: p for path in Path('/proc').glob('[0-9]*')
             if (p := process_identity(path.name))}
    records = [info.get('worker_identity'), (state or {}).get('worker_identity'),
               (state or {}).get('child_identity')]
    records += list((remembered or {}).values())
    anchors = {r['pid']: table[r['pid']] for r in records
               if r and same_process(r, table.get(r.get('pid')))}
    if not info.get('worker_identity') and alive(info):
        anchors[info['pid']] = table[info['pid']]
    token = config.get('attempt_token') if config.get('boot_id') == BOOT_ID else None
    if token:
        for pid, identity in table.items():
            if identity['uid'] == os.getuid() and has_attempt_token(pid, token):
                anchors[pid] = identity
    groups = {(p['sid'], p['pgid']) for p in anchors.values()}
    owned = {pid: p for pid, p in table.items() if (p['sid'], p['pgid']) in groups}
    primary_pgid = info.get('pgid', info['pid'])
    primary_sid = info.get('sid', primary_pgid)
    unverified = {pid: p for pid, p in table.items()
                  if (p['sid'], p['pgid']) == (primary_sid, primary_pgid) and pid not in owned}
    expected_worker = info.get('worker_identity') or (state or {}).get('worker_identity')
    reused = bool(expected_worker and (
        expected_worker.get('boot_id') != BOOT_ID or
        (table.get(info['pid']) and not same_process(expected_worker, table[info['pid']]))))
    if reused:
        # A different live session leader or a different boot cannot be the old
        # group. Ignore it for restart, and never include it in stop signals.
        unverified = {}
    return owned, unverified


def signal_process(expected, sig):
    """Open the process handle, recheck identity, then signal that exact handle."""
    # Some Conda Python builds omit the wrappers despite a capable host kernel.
    # Use libc's named pidfd functions there; no architecture-specific syscall IDs.
    libc = None
    try:
        if hasattr(os, 'pidfd_open'):
            fd = os.pidfd_open(expected['pid'])
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
            libc.pidfd_open.restype = ctypes.c_int
            fd = libc.pidfd_open(expected['pid'], 0)
            if fd < 0:
                raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    except ProcessLookupError:
        return False
    except AttributeError as exc:
        raise RuntimeError('This Linux runtime needs pidfd support for identity-safe signalling') from exc
    try:
        if not same_process(expected, process_identity(expected['pid'])):
            return False
        if hasattr(signal, 'pidfd_send_signal'):
            signal.pidfd_send_signal(fd, sig)
        else:
            libc = libc or ctypes.CDLL(None, use_errno=True)
            libc.pidfd_send_signal.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)
            libc.pidfd_send_signal.restype = ctypes.c_int
            if libc.pidfd_send_signal(fd, sig, None, 0) < 0:
                raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def stop_owned(info, state, config, grace):
    remembered, signalled, signals = {}, set(), []
    owned, unverified = owned_processes(info, state, config)
    if not owned:
        if unverified:
            raise SystemExit('A group survives but its ownership cannot be verified; no processes were signalled.')
        raise SystemExit('No live owned job processes to stop; unrelated processes were not signalled.')
    remembered.update(owned)
    deadline = time.monotonic() + min(55, max(0, grace))
    while owned:
        # Keep a live worker waiting for its child during the graceful period.
        # The child can finish a checkpoint and the worker can record its exit.
        targets = {pid: p for pid, p in owned.items() if pid != info['pid']} or owned
        for pid, identity in targets.items():
            key = (pid, identity['start_ticks'])
            if key not in signalled:
                delivered = signal_process(identity, signal.SIGTERM)
                signals.append(dict(identity=identity, signal='SIGTERM', delivered=delivered))
                signalled.add(key)
        if time.monotonic() >= deadline:
            break
        time.sleep(.1)
        owned, unverified = owned_processes(info, state, config, remembered)
        remembered.update(owned)
    if owned:
        deadline = time.monotonic() + 3
        while owned and time.monotonic() < deadline:
            for identity in owned.values():
                delivered = signal_process(identity, signal.SIGKILL)
                signals.append(dict(identity=identity, signal='SIGKILL', delivered=delivered))
            time.sleep(.1)
            owned, unverified = owned_processes(info, state, config, remembered)
            remembered.update(owned)
    owned, unverified = owned_processes(info, state, config, remembered)
    if owned or unverified:
        raise SystemExit('Processes remain owned or unverified; stop is incomplete and restart stays blocked.')
    # Read the worker's final state after it has exited, avoiding a stale write
    # that would discard its child identity, exit code or completion receipt.
    state_path = Path(info['attempt']) / 'status.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
    save(state_path, dict(state or {}, status='stopped', stopped_at=stamp(), pid=info['pid'],
                         stop_receipt=dict(owned_identities=list(remembered.values()), signals=signals,
                                           remaining_owned=0, remaining_unverified=0)))
    print('Verified all owned job processes have stopped.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('start', 'restart', 'status', 'logs', 'stop'):
        s = sub.add_parser(action)
        s.add_argument('name')
        if action == 'start':
            s.add_argument('--gpu', default='3')
        if action == 'stop':
            s.add_argument('--grace', type=float, default=30,
                           help='Seconds allowed for a graceful save before terminating a stubborn job')
    w = sub.add_parser('_worker')
    w.add_argument('attempt', type=Path)
    # Parse everything after the explicit -- as the child command so options
    # before that separator work on either side of the job name.
    argv = sys.argv[1:]
    tail = None
    if '--' in argv:
        pos = argv.index('--'); tail = argv[pos+1:]; argv = argv[:pos]
    a = p.parse_args(argv)
    if tail is not None:
        if a.action != 'start':
            p.error('Child command is valid only for start.')
        a.command = tail
    root = a.root.resolve()
    if a.action == '_worker':
        attempt = a.attempt.resolve()
        config = json.loads((attempt / 'config.json').read_text())
        state = dict(config, status='running', pid=os.getpid(), started_at=stamp(),
                     worker_identity=process_identity(os.getpid()))
        save(attempt / 'status.json', state)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=config['gpu'], OMP_NUM_THREADS='8',
                   OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1')
        try:
            child = subprocess.Popen(config['command'], cwd=config['cwd'], env=env)
            state['child_pid'] = child.pid
            state['child_identity'] = process_identity(child.pid)
            save(attempt / 'status.json', state)
            code = child.wait()
            state.update(status='succeeded' if code == 0 else 'failed',
                         returncode=code, finished_at=stamp())
        except BaseException as e:
            state.update(status='failed', error=repr(e), finished_at=stamp())
        save(attempt / 'status.json', state)
        return
    job = root / 'jobs' / a.name
    current = job / 'current.json'
    info = json.loads(current.read_text()) if current.exists() else None
    state = None
    if info and (Path(info['attempt']) / 'status.json').exists():
        state = json.loads((Path(info['attempt']) / 'status.json').read_text())
    config = json.loads((Path(info['attempt']) / 'config.json').read_text()) if info else {}
    if a.action in ('start', 'restart'):
        if info:
            owned, unverified = owned_processes(info, state, config)
            if owned:
                raise SystemExit('This job has verified live owned processes; stop it before retrying.')
            if unverified:
                raise SystemExit('A surviving group has unverified ownership; restart cannot safely overlap it.')
        if a.action == 'restart':
            if not info:
                raise SystemExit('No previous attempt.')
            previous = json.loads((Path(info['attempt']) / 'config.json').read_text())
            command, gpu = previous['original_command'], previous['gpu']
            launch_cwd = Path(previous['cwd'])
        else:
            command, gpu = tail or [], a.gpu
            launch_cwd = Path.cwd()
            if command and command[0] == '--':
                command = command[1:]
            if not command:
                raise SystemExit('Supply an executable command after --.')
        attempt = job / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt.mkdir(parents=True)
        original = command[:]
        script_hashes = {}
        for i, token in enumerate(command):
            candidate = Path(token)
            if not candidate.is_absolute(): candidate = launch_cwd / candidate
            if token.endswith('.py') and candidate.is_file():
                data = candidate.read_bytes()
                dest = attempt / candidate.name
                dest.write_bytes(data)
                script_hashes[str(candidate.resolve())] = hashlib.sha256(data).hexdigest()
                command[i] = str(dest)
        config = dict(command=command, original_command=original, gpu=gpu,
                      cwd=str(launch_cwd), script_sha256=script_hashes,
                      created_at=stamp(), attempt=str(attempt), boot_id=BOOT_ID,
                      attempt_token=secrets.token_hex(24))
        save(attempt / 'config.json', config)
        with (attempt / 'log.txt').open('ab', buffering=0) as log:
            worker_env = os.environ.copy()
            worker_env[TOKEN_ENV] = config['attempt_token']
            worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                       '--root', str(root), '_worker', str(attempt)],
                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                       start_new_session=True, env=worker_env)
        info = dict(pid=worker.pid, pgid=worker.pid, sid=worker.pid,
                    worker_identity=process_identity(worker.pid), attempt=str(attempt), name=a.name)
        save(current, info)
        print(json.dumps(info, indent=2))
    elif a.action == 'status':
        if not info:
            print('No job with this name.'); return
        owned, unverified = owned_processes(info, state, config)
        if state and state['status'] == 'running' and not alive(info):
            state['observed_status'] = 'orphaned_running' if owned else 'worker_missing'
        print(json.dumps(dict(pointer=info, worker_alive=alive(info), state=state,
                             owned_processes=list(owned.values()), unverified_primary_group=list(unverified.values())), indent=2))
    elif a.action == 'logs':
        if not info:
            raise SystemExit('No job with this name.')
        lines = (Path(info['attempt']) / 'log.txt').read_text(errors='replace').splitlines()
        print('\n'.join(lines[-55:]))
    elif a.action == 'stop':
        if not info:
            raise SystemExit('No job with this name.')
        stop_owned(info, state, config, a.grace)


if __name__ == '__main__':
    main()
