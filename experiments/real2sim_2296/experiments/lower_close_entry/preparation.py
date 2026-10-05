"""Experimental accepted-action preparation using only existing motor targets.

An eligible exact lower-close request is accepted while standing at the far
stance. AMO holds there while its named upper targets park the right arm. The
same accepted request then enters ordinary alignment and ScaleBFM crouching.
No actual robot, joint, contact, model or controller-history state is assigned.
"""
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
            self.preparation_indices = np.array([self.i.index[n] for n in ARM])
            if np.any(PARK < self.i.ranges[self.preparation_indices,0]) or np.any(PARK > self.i.ranges[self.preparation_indices,1]):
                raise ValueError('Preparation arm target exceeds original source range')

        def step(self,command):
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
                    self.pending_lower_close = {'id':identity,'accepted_at':now,
                        'source_sequence':command.get('seq',-1),'start_arm':measured[self.i.rq[self.preparation_indices]].copy(),
                        'target':measured[self.i.rq[self.preparation_indices]].copy(),'dwell':0.}
                    self.lower_preparation_events.append({'event':'eligible_close_request_accepted',
                        'time':now,'id':identity,'source_sequence':command.get('seq',-1),
                        'source_candidate':candidate})
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
                    candidate = self.candidate(pending['id'])
                    if not candidate or not candidate['eligible'] or candidate.get('action') != 'close':
                        self.failed = 'Accepted lower close lost its actual preconditions during preparation'
                        self.paused = True;self.change('paused',self.failed);return
                    entry = self.instance_entry_templates[pending['id']].copy()
                    entry[self.i.rq[self.preparation_indices]] = PARK
                    self.instance_entry_templates[pending['id']] = entry
                    forwarded.update(interact=True,interaction_candidate_id=pending['id'],interaction_action='close')
                    self.lower_preparation_events.append({'event':'accepted_close_dispatched_after_preparation',
                        'time':now,'id':pending['id'],'accepted_at':pending['accepted_at'],
                        'source_sequence':pending['source_sequence'],'dispatch_sequence':command.get('seq',-1),
                        'arm_error_rad':arm_error,'arm_speed_rad_s':arm_speed})
                    self.pending_lower_close = None
                elif age >= 4.:
                    self.failed = 'Lower close arm preparation did not settle within4s'
                    self.paused = True;self.change('paused',self.failed);return
                return super().step(forwarded)
            return super().step(command)
    return PreparedLowerClose
