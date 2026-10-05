"""Accepted lower-close preparation using only existing motor targets.

An eligible exact lower-close request is accepted while standing at the far
stance. AMO holds there while its named upper targets park the right arm. The
same accepted request then enters ordinary alignment and ScaleBFM crouching.
No actual robot, joint, contact, model or controller-history state is assigned.
The caller owns decoded-frame/action authorization before step(). This wrapper
then owns a bounded accepted action, so later input expiry does not cancel its
preparation. Public candidates stay busy until ordinary native dispatch.

The close-entry arm park persists in that instance's reference template; later
attempts use it until the Engine is destroyed. Native handback still captures
the actual recovered posture in the base Engine. Reopening is a separate test.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ARM = tuple('right_'+name+'_joint' for name in (
    'shoulder_pitch','shoulder_roll','shoulder_yaw','elbow',
    'wrist_roll','wrist_pitch','wrist_yaw'))
PARK = np.array([.8,-.25,-.2,.3,0.,0.,0.])

def extend_engine(base):
    class PreparedLowerClose(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.pending_lower_close = None
            self.lower_preparation_rows = []
            self.lower_preparation_events = []
            self.lower_preparation_source = str(Path(__file__).resolve())
            self.lower_preparation_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            self.preparation_indices = np.array([self.i.index[n] for n in ARM])
            if np.any(PARK < self.i.ranges[self.preparation_indices,0]) or np.any(PARK > self.i.ranges[self.preparation_indices,1]):
                raise ValueError('Preparation arm target exceeds original source range')

        def candidates(self, *, include_distant=False):
            choices = base.candidates(self, include_distant=include_distant)
            if self.pending_lower_close is not None:
                return [{**item, 'eligible': False,
                    'reason': 'Preparing the arm for closing; interaction in progress'} for item in choices]
            return choices

        def status(self, wall_seconds=0.):
            result = super().status(wall_seconds)
            if self.pending_lower_close is not None and not self.paused:
                result.update(mode='preparing_close', control_mode=self.mode,
                    message='Preparing the arm before moving toward the open lower drawer.',
                    preparation={'id': self.pending_lower_close['id'],
                        'accepted_at': self.pending_lower_close['accepted_at'],
                        'age_s': float(self.data.time)-self.pending_lower_close['accepted_at']})
            return result

        def _preparation_failed(self, reason, now):
            pending = self.pending_lower_close
            self.lower_preparation_events.append({'event': 'preparation_failed',
                'time': now, 'id': pending['id'], 'reason': reason,
                'accepted_request': pending['accepted_request']})
            self.pending_lower_close = None
            self.failed = reason
            self.paused = True
            self.change('paused', self.failed)

        def step(self,command):
            if self.paused:
                return super().step(command)
            now = float(self.data.time)
            if (self.pending_lower_close is None and self.mode == 'walking'
                    and command.get('interact') and command.get('interaction_action') == 'close'):
                identity = command.get('interaction_candidate_id')
                candidate = self.candidate(identity)
                if (candidate and candidate['id'] == identity and candidate.get('eligible')
                        and candidate.get('action') == 'close' and candidate.get('standing_recovery')):
                    age_s = command.get('age_s')
                    if type(age_s) not in (int,float) or not np.isfinite(age_s) or not 0. <= age_s <= .35:
                        self.lower_preparation_events.append({'event':'stale_close_request_discarded','time':now,'id':identity})
                        return super().step(dict(command,interact=False))
                    measured = self.i.measured(self.data)
                    accepted_request = {key: command.get(key) for key in (
                        'seq', 'client_id', 'age_s', 'interaction_candidate_id',
                        'interaction_action', 'interaction_frame_seq',
                        'interaction_view_session', 'look_revision', 'view_session')}
                    self.pending_lower_close = {'id':identity,'accepted_at':now,
                        'source_sequence':command.get('seq',-1),'start_arm':measured[self.i.rq[self.preparation_indices]].copy(),
                        'target':measured[self.i.rq[self.preparation_indices]].copy(),'dwell':0.,
                        'accepted_request': accepted_request}
                    self.lower_preparation_events.append({'event':'eligible_close_request_accepted',
                        'time':now,'id':identity,'source_sequence':command.get('seq',-1),
                        'source_candidate':candidate, 'accepted_request': accepted_request})
            pending = self.pending_lower_close
            if pending is not None:
                age = now-pending['accepted_at']
                u = self.h.smooth(age/.8)
                desired = (1-u)*pending['start_arm']+u*PARK
                pending['target'] += np.clip(desired-pending['target'],-.03,.03)
                # These are reference/command arrays already consumed by AMO.
                # Runtime torque clipping and native stepping are unchanged.
                self.posture[self.i.rq[self.preparation_indices]] = pending['target']
                for name,value in zip(ARM,pending['target']):
                    if name in self.upper:self.upper[name] = float(value)
                measured = self.i.measured(self.data)
                arm_error = float(np.max(abs(measured[self.i.rq[self.preparation_indices]]-PARK)))
                arm_speed = float(np.max(abs(self.data.qvel[self.i.v[self.preparation_indices]])))
                _,heading,tilt,speed = self.actual()
                ready = (age >= 1. and arm_error < .08 and arm_speed < .20
                    and speed < .08 and tilt < 10. and np.all(self.i.floor_force(self.data,self.floor)>20.))
                pending['dwell'] = pending['dwell']+.02 if ready else 0.
                self.lower_preparation_rows.append({'time':now,'age_s':age,'id':pending['id'],
                    'target':pending['target'].tolist(),'actual_arm_error_rad':arm_error,
                    'actual_arm_speed_rad_s':arm_speed,'ready_dwell_s':pending['dwell'],
                    'root_position_m':measured[:3].tolist()})
                forwarded = dict(command,active=False,interact=False)
                if pending['dwell'] >= .25:
                    # Public candidates are busy while preparing. Recheck the
                    # exact actual base conditions without that UI suppression.
                    candidate = next((item for item in base.candidates(self)
                        if item['id'] == pending['id']), None)
                    if not candidate or not candidate['eligible'] or candidate.get('action') != 'close':
                        self._preparation_failed('Accepted lower close lost its actual preconditions during preparation', now)
                        return
                    entry = self.instance_entry_templates[pending['id']].copy()
                    entry[self.i.rq[self.preparation_indices]] = PARK
                    self.instance_entry_templates[pending['id']] = entry
                    forwarded.update(interact=True,interaction_candidate_id=pending['id'],interaction_action='close')
                    self.lower_preparation_events.append({'event':'accepted_close_dispatched_after_preparation',
                        'time':now,'id':pending['id'],'accepted_at':pending['accepted_at'],
                        'source_sequence':pending['source_sequence'],'dispatch_sequence':command.get('seq',-1),
                        'arm_error_rad':arm_error,'arm_speed_rad_s':arm_speed,
                        'accepted_request': pending['accepted_request'],
                        'persistent_instance_entry_arm': PARK.tolist()})
                    self.pending_lower_close = None
                elif age >= 4.:
                    self._preparation_failed('Lower close arm preparation did not settle within 4 seconds', now)
                    return
                return super().step(forwarded)
            return super().step(command)

        def finish(self, wall_seconds):
            super().finish(wall_seconds)
            source = Path(self.lower_preparation_source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != self.lower_preparation_source_sha256:
                raise ValueError('Preparation helper changed during execution')
            files = {'preparation_events.json': self.lower_preparation_events,
                'preparation_samples.json': self.lower_preparation_rows}
            for name, value in files.items():
                (self.out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            receipt = {'schema': 'g1-drawer-preparation/v1',
                'produced_at': datetime.now(timezone.utc).isoformat(),
                'source': {'path': str(source), 'sha256': digest},
                'base_inputs_sha256': sha(self.out/'inputs.json'),
                'base_physics_receipt_sha256': sha(self.out/'receipt.json'),
                'artifacts': {name: sha(self.out/name) for name in files},
                'preparation_count': sum(e['event']=='eligible_close_request_accepted' for e in self.lower_preparation_events),
                'dispatch_count': sum(e['event']=='accepted_close_dispatched_after_preparation' for e in self.lower_preparation_events),
                'pending_at_finish': self.pending_lower_close is not None,
                'control_mode_during_preparation': 'walking (AMO hold)',
                'public_mode_during_preparation': 'preparing_close',
                'instance_reference_park_persists': True,
                'scope': 'Accepted-action preparation only. Original native motors and guards remain in the base Engine. Frame authorization is owned by the caller; missing frame metadata denotes a headless driver, not a verified browser F. This receipt does not award physical task success.'}
            (self.out/'preparation.receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return PreparedLowerClose
