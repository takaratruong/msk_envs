#!/usr/bin/env python3
"""Read-only force-direction audit of the recorded 500 Hz task contacts.

This is a contact diagnostic, not a grasp-force-closure or full-task certificate.
MuJoCo contact normals point from geom1 to geom2. The returned force acts on
geom2; transpose the stored world-to-contact frame, and negate for geom1.
https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjcontact
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duration(mask, time):
    best = 0.0
    begin = None
    for i, flag in enumerate(mask):
        if flag and begin is None:
            begin = i
        if begin is not None and (not flag or i == len(mask) - 1):
            end = i if flag else i - 1
            best = max(best, float(time[end] - time[begin]))
            begin = None
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.rollout.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a fresh output directory')
    raw = json.loads((run / 'receipt.json').read_text())
    source = run / 'task_contacts_substeps.jsonl'
    assert sha(source) == raw['artifacts'][source.name]
    assert sha(run / 'rollout.npz') == raw['artifacts']['rollout.npz']
    inp = json.loads((run / 'inputs.json').read_text())
    assert sha(run / 'inputs.json') == raw['inputs_receipt']['sha256']
    task_path = Path(inp['arguments']['task'])
    assert sha(task_path) == inp['inputs'][str(task_path)]
    task = json.loads(task_path.read_text())
    z = np.load(run / 'rollout.npz', allow_pickle=False)
    time, hand, support, forces, normals, depths, count, support_load = [], [], [], [], [], [], [], []
    for text in source.read_text().splitlines():
        line = json.loads(text)
        hand_force, support_force = np.zeros(3), np.zeros(3)
        digit_forces, digit_normals = np.zeros(3), np.zeros((3, 3))
        depth, contacts, nonhand_load = 0.0, 0, 0.0
        for c in line['contacts']:
            bs = [c['body1'], c['body2']]
            if 'task_bottle' not in bs:
                continue
            frame = np.asarray(c['frame_world_to_contact'], dtype=float)
            local = np.asarray(c['force_contact_frame'], dtype=float)
            assert frame.shape == (3, 3) and local.shape == (6,)
            assert np.isfinite(frame).all() and np.isfinite(local).all() and np.isfinite(c['distance'])
            assert np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-6)
            sign = 1 if bs[1] == 'task_bottle' else -1
            world = sign * (frame.T @ local[:3])
            other = bs[0] if sign == 1 else bs[1]
            if other.startswith('right_hand_'):
                hand_force += world
                depth = min(depth, float(c['distance']))
                contacts += int(local[0] > 0.05)
                for j, name in enumerate(('thumb', 'index', 'middle')):
                    if name in other:
                        digit_forces[j] += max(0.0, local[0])
                        digit_normals[j] += sign * frame[0] * max(0.0, local[0])
            else:
                support_force += world
                nonhand_load += max(0.0, float(local[0]))
        time.append(line['time'])
        hand.append(hand_force)
        support.append(support_force)
        forces.append(digit_forces)
        normals.append(digit_normals)
        depths.append(depth)
        count.append(contacts)
        support_load.append(nonhand_load)
    t, h, s, f, normal = map(np.asarray, (time, hand, support, forces, normals))
    assert len(t) == raw['contact_sampling']['task_contact_substeps']
    assert np.all(np.diff(t) > 0) and np.allclose(np.diff(t), 0.002, atol=1e-9, rtol=0)
    thumb, other = normal[:, 0], normal[:, 1:].sum(axis=1)
    denominator = np.linalg.norm(thumb, axis=1) * np.linalg.norm(other, axis=1)
    cosine = np.full(len(t), np.nan)
    valid = denominator > 1e-10
    cosine[valid] = np.einsum('ij,ij->i', thumb[valid], other[valid]) / denominator[valid]
    simultaneous = (f[:, 0] > .05) & (np.max(f[:, 1:], axis=1) > .05)
    opposed = simultaneous & (cosine < -.8)
    weight = float(task['object']['mass']) * 9.81
    # Equal-and-opposite environmental contacts can have zero net wrench while
    # remaining active supports; use their summed positive normal load.
    nonhand_load = np.asarray(support_load)
    loaded = opposed & (h[:, 2] > .8 * weight) & (nonhand_load < .05)
    phase = z['phase'][np.minimum(np.searchsorted(z['time'], t), len(z['time']) - 1)]
    summaries = []
    for name in dict.fromkeys(phase.tolist()):
        at = phase == name
        summaries.append(dict(phase=str(name), seconds=float(t[at][-1]-t[at][0]),
            opposing_normal_fraction=float(np.mean(opposed[at])),
            hand_force_world_mean_N=np.mean(h[at], axis=0).tolist(),
            support_force_world_mean_N=np.mean(s[at], axis=0).tolist(),
            digit_normal_force_mean_N=np.mean(f[at], axis=0).tolist()))
    out.mkdir(parents=True)
    np.savez_compressed(out / 'forces.npz', time=t, hand_force_world=h,
        support_force_world=s, digit_normal_force=f, thumb_other_normal_cosine=cosine,
        simultaneous_digit_contacts=simultaneous, opposing_normals=opposed,
        loaded_without_world_support=loaded, phase=phase,
        nonhand_positive_normal_load_N=nonhand_load,
        hand_contact_distance=np.asarray(depths), positive_hand_contacts=np.asarray(count))
    report = dict(schema='g1-task-contact-force-audit/v1', produced_at=datetime.now(timezone.utc).isoformat(),
        inputs={str(p): sha(p) for p in [source, run/'receipt.json', run/'inputs.json', run/'rollout.npz', task_path, Path(__file__).resolve()]},
        source_scope=raw['scope'], task_success_claim=False, samples=len(t), rate_hz=500,
        object_weight_N=weight, digit_order=['thumb', 'index', 'middle'],
        simultaneous_contact_longest_s=duration(simultaneous,t),
        opposing_normal_contact_longest_s=duration(opposed,t),
        loaded_opposition_without_support_longest_s=duration(loaded,t),
        hand_peak_normal_force_N=float(np.max(f)),
        hand_minimum_contact_distance_m=float(min(depths)), phases=summaries,
        limitations=['Contact-direction and load diagnostic only; no friction-cone wrench/force-closure certificate.',
            'Recorded 500 Hz physics-substep samples; no continuous-time contact bound.',
            'Phase labels use the enclosing 50 Hz control sample.',
            'No object-state interpolation or prescribed reference pose is used to infer a physical lift.'],
        force_convention_source='https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjcontact',
        artifacts={'forces.npz': sha(out/'forces.npz')})
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['inputs','phases']},indent=2))


if __name__ == '__main__':
    main()
