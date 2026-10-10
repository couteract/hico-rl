"""Stateful, bounded technical collector with independently checkable WALs."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import base64, hashlib, json, os, time
from pathlib import Path
from typing import Any, Callable

class CollectorMode(str, Enum):
    QUALIFY_ZERO_STEP='QUALIFY_ZERO_STEP'; EXECUTE_BOUNDED='EXECUTE_BOUNDED'
class TechnicalStatus(str, Enum):
    TECHNICAL_CERTIFIED='TECHNICAL_CERTIFIED'; TECHNICAL_BUDGET_COMPLETE='TECHNICAL_BUDGET_COMPLETE'; TERMINAL='TERMINAL'; FAILED='FAILED'
PERMANENT_ROLE='PERMANENTLY_EXCLUDED_TECHNICAL_FIXTURE'
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _native(v):
    try:
        import numpy as np
        if isinstance(v,np.ndarray):
            a=np.ascontiguousarray(v); b=a.tobytes()
            return {'encoding':'ndarray_base64','dtype':a.dtype.str,'shape':list(a.shape),'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
        if isinstance(v,np.generic): return v.item()
    except ImportError: pass
    if isinstance(v,dict): return {str(k):_native(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_native(x) for x in v]
    if v is None or isinstance(v,(str,int,float,bool)): return v
    raise TypeError(f'WAL_NON_NATIVE:{type(v).__name__}')
def action_hash(action):
    try:
        import numpy as np
        if isinstance(action,dict) and action.get('encoding')=='ndarray_base64':
            raw=base64.b64decode(action['data']); return hashlib.sha256(raw).hexdigest()
        a=np.ascontiguousarray(action,dtype=np.float32); return hashlib.sha256(a.tobytes()).hexdigest()
    except (ImportError,TypeError,ValueError): return hashlib.sha256(canonical(_native(action))).hexdigest()
@dataclass(frozen=True)
class TechnicalFixtureIdentity:
    physical_group_id:str; snapshot_path:Path; role:str=PERMANENT_ROLE
    def __post_init__(self):
        technical=self.physical_group_id.startswith(('q35_harness_only_','q40_harness_only_'))
        scientific=self.physical_group_id.startswith('q30r3_train_') and self.role=='q30r3_training_identity'
        if not (scientific or (technical and self.role==PERMANENT_ROLE)): raise ValueError('TYPED_IDENTITY_ROLE_REQUIRED')
@dataclass
class WalSink:
    root:Path; identity:str; role:str=PERMANENT_ROLE
    def __post_init__(self): self.root=Path(self.root);self.root.mkdir(parents=True,exist_ok=True);self.head={k:None for k in ('FRAME','ENV','TRANSITION')};self.seq={k:0 for k in self.head}
    def append(self,kind,payload):
        if kind not in self.head: raise ValueError('WAL_KIND')
        self.seq[kind]+=1
        body=_native({'schema':'B02Q41_'+kind+'_V1','sequence':self.seq[kind],'identity':self.identity,'role':self.role,'parent_sha256':self.head[kind],**payload})
        digest=hashlib.sha256(canonical(body)).hexdigest();row={**body,'commit_sha256':digest};p=self.root/f'{kind}_WAL.jsonl'
        with p.open('ab',buffering=0) as f:f.write(canonical(row)+b'\n');os.fsync(f.fileno())
        fd=os.open(self.root,os.O_DIRECTORY);os.fsync(fd);os.close(fd);self.head[kind]=digest;return row

@dataclass
class CollectorSession:
    identity:TechnicalFixtureIdentity; binding:Any; actor:Any; normalizer:Any; provider:Any; recorder:Any; wal:WalSink
    generator:Any=None; frame:int=0; held_visual:Any=None; current_decision:Any=None; provider_record:Any=None
    previous_rows:list=field(default_factory=list); pending_transition:Any=None; decision_boundary:Any=None; state_hashes:dict=field(default_factory=dict)
    initial_prepare_count:int=0; next_prepare_count:int=0
    @property
    def instance_token(self): return f'{id(self)}:{id(self.binding)}:{id(self.actor)}:{id(self.provider)}:{id(self.recorder)}:{id(self.generator)}'

def _transition(session, pending, *, next_prepared=None, terminal=None, budget=False):
    env=pending['env']; frame=pending['frame']; payload={'frame':frame,'frame_commit_sha256':pending['frame_commit_sha256'],'env_commit_sha256':env['commit_sha256'],'current_identity':pending['current_identity'],'next_live':next_prepared is not None,'next_identity':({'frame':frame+1,'identity':session.identity.physical_group_id,'memory_version':next_prepared.get('memory_version')} if next_prepared is not None else None),'next_current1981':next_prepared.get('current1981') if next_prepared is not None else None,'next_raw_memory256':next_prepared.get('raw_memory256') if next_prepared is not None else None,'next_normalized_memory256':next_prepared.get('normalized_memory256') if next_prepared is not None else None,'terminal':terminal,'technical_budget':bool(budget)}
    row=session.wal.append('TRANSITION',payload);p=pending['prepared'];session.previous_rows.append({'frame':frame,'submitted7':p.get('submitted7'),'env_action7_sha256':env.get('env_action7_sha256'),'visual_pooled_2x960':p.get('visual_pooled_2x960'),'state8':p.get('state8')});session.pending_transition=None;return row

def independent_validate(root, identity):
    """Recompute raw WAL chains and cross-links; no execution status is trusted."""
    root=Path(root); docs={}
    for kind in ('FRAME','ENV','TRANSITION'):
        rows=[]; head=None
        p=root/f'{kind}_WAL.jsonl'
        if not p.exists(): raise ValueError(f'MISSING_{kind}_WAL')
        for i,line in enumerate(p.read_bytes().splitlines(),1):
            row=json.loads(line); commit=row.pop('commit_sha256'); digest=hashlib.sha256(canonical(row)).hexdigest()
            if digest!=commit or row['sequence']!=i or row['parent_sha256']!=head or row['identity']!=identity.physical_group_id or row['role']!=identity.role: raise ValueError(f'{kind}_CHAIN_OR_ROLE_{i}')
            row['commit_sha256']=commit;head=commit;rows.append(row)
        docs[kind]=rows
    frames={r['frame']:r for r in docs['FRAME']}; envs={r['frame']:r for r in docs['ENV']}; ts={r['frame']:r for r in docs['TRANSITION']}
    if set(frames)!=set(envs) or set(frames)!=set(ts): raise ValueError('WAL_FRAME_COUNTS')
    for i in sorted(frames):
        f,e,t=frames[i],envs[i],ts[i]
        if e['frame_commit_sha256']!=f['commit_sha256'] or t['frame_commit_sha256']!=f['commit_sha256'] or t['env_commit_sha256']!=e['commit_sha256']: raise ValueError('WAL_CROSS_REFERENCE')
        if f['action_sha256']!=e['env_action7_sha256']: raise ValueError('ACTION_HASH_MISMATCH')
        if action_hash(f['current']['submitted7'])!=f['action_sha256']: raise ValueError('FRAME_SUBMITTED_ACTION_MISMATCH')
        if i+1 in frames and (not t['next_live'] or t['next_identity']['frame']!=i+1): raise ValueError('NEXT_FRAME_IDENTITY')
        if i+1 in frames:
            n=frames[i+1]['current_identity']
            if n['frame']!=t['next_identity']['frame'] or n['identity']!=t['next_identity']['identity'] or n.get('memory_version')!=t['next_identity'].get('memory_version'): raise ValueError('NEXT_MEMORY_VERSION_OR_IDENTITY')
        if i+1 not in frames and t['next_live']: raise ValueError('MISSING_NEXT_FRAME')
    return {'status':'PASS','frames':len(frames),'envs':len(envs),'transitions':len(ts),'last_transition_sha256':docs['TRANSITION'][-1]['commit_sha256'] if ts else None}

class ProductionFramePreparer:
    """Frame-zero and causal next-frame preparation over one live production session."""
    def __init__(self, *, entry, run_dir, source_hashes=None):
        self.entry=entry; self.run_dir=Path(run_dir); self.source_hashes=source_hashes or {}; self.initial_calls=0; self.next_calls=0
    def prepare_initial(self, session):
        if self.initial_calls: raise RuntimeError('INITIAL_PREPARER_MUST_RUN_ONCE')
        from scripts.b02q30r3_online import prepare_first_step
        self.initial_calls+=1
        result=prepare_first_step(binding=session.binding,entry=self.entry,actor=session.actor,normalizer=session.normalizer,
          provider_session=session.provider,run_dir=self.run_dir,recorder=session.recorder,rows=session.previous_rows,
          pending=None,generator=session.generator,source_hashes=self.source_hashes)
        session.held_visual=result['visual'];session.current_decision=session.recorder.current
        session.provider_record=result['provider_record'];session.decision_boundary=0
        return result['steprecord']
    def prepare_next(self, session):
        if session.frame<=0: raise ValueError('NEXT_PREPARER_REQUIRES_FRAME_GT_ZERO')
        if session.pending_transition is None: raise RuntimeError('NEXT_PREPARER_PENDING_STEP_REQUIRED')
        import numpy as np, torch
        from scripts.b02q30r3_online import current_from_binding
        from scripts.b02q30_core import project_torch
        from scripts.b02q30_tensor_hash import module_state_hash
        task=int(self.entry['canonical_task_id']); frame=int(session.frame)
        cur,raw,visual,side,fields,qbefore,qafter=current_from_binding(session.binding,task,frame,session.held_visual)
        session.held_visual=visual
        condition=cur.copy();condition[:1928]=session.normalizer.transform(cur[None,:1928])[0]
        remaining=int(fields['queue_remaining'][0]); boundary=(qbefore==0)
        if boundary: session.recorder.begin(frame,max(1,remaining+1))
        if session.recorder.current is None: raise RuntimeError('RECORDED_DECISION_QUEUE_BOUNDARY_MISMATCH')
        provider_record=session.provider.current_record if session.provider is not None else None
        interval_audit=None
        if session.provider is not None and boundary:
            p=session.pending_transition['prepared']
            executed={'frame':int(session.pending_transition['frame']),'submitted7':p['submitted7'],
                      'visual_pooled_2x960':p['visual_pooled_2x960'],'state8':p['state8'],
                      'env_action7_sha256':(session.pending_transition.get('env') or {}).get('env_action7_sha256',action_hash(p['submitted7']))}
            prior=session.previous_rows+[executed]
            decision=session.recorder.decisions[-1]
            start=int(decision['chunk_start_frame']);end=int(decision['chunk_end_frame'])
            by_frame={int(x['frame']):x for x in prior}
            if any(j not in by_frame for j in range(start,end)): raise RuntimeError('PREVIOUS_INTERVAL_ROWS_MISSING')
            interval=np.asarray([by_frame[j]['submitted7'] for j in range(start,end)],np.float32)
            if len(interval)!=int(decision['executed_action_count']) or end!=frame: raise RuntimeError('PREVIOUS_INTERVAL_END_EXCLUSIVE_MISMATCH')
            interval_audit={'start_inclusive':start,'end_exclusive':end,'executed_action_count':len(interval),'frame_ids':list(range(start,end)),
              'mean7':interval.mean(axis=0).tolist(),'last7':interval[-1].tolist(),'action_hashes':[by_frame[j]['env_action7_sha256'] for j in range(start,end)]}
            provider_record=session.provider.produce_decision(episode_id=self.entry['physical_group_key'],frame=frame,
              fold=int(self.entry['fold']),decision_index=int(session.recorder.current['decision_index']),visual=np.asarray(visual,np.float32),
              state=np.asarray(fields['state'],np.float32),previous_decision=session.recorder.decisions[-1] if session.recorder.decisions else None,
              previous_rows=prior)
        if self.entry.get('require_provider_record') and (provider_record is None or provider_record.get('status')!='PASS'):
            raise RuntimeError('ONLINE_PROVIDER_RECORD_REQUIRED')
        device=next(session.actor.parameters()).device
        noise=torch.randn((1,6),generator=session.generator,device=device);xt=torch.as_tensor(condition,device=device).reshape(1,1981)
        with torch.no_grad():u,logpi,_=session.actor(xt,noise=noise);submitted,bounded=project_torch(torch.as_tensor(raw,device=device).reshape(1,7),u)
        submitted_np=submitted[0].detach().cpu().numpy().astype(np.float32)
        record={'status':'PREPARED','frame':frame,'current1981':condition.tolist(),'visual_pooled_2x960':visual.tolist(),
          'state8':np.asarray(fields['state'],np.float32).tolist(),'raw_base7':raw.tolist(),
          'bounded_base7':bounded[0].detach().cpu().numpy().tolist(),'latent6':u[0].detach().cpu().numpy().tolist(),
          'log_pi_u':float(logpi.item()),'submitted7':submitted_np.tolist(),
          'queue':{'cursor':int(fields['queue_cursor'][0]),'remaining':remaining,'actions':fields['queue_actions'].tolist(),'mask':fields['queue_mask'].tolist()},
          'fold':int(self.entry['fold']),'memory_version':int(session.recorder.current['decision_index']),
          'provider_hash':self.entry.get('provider_hash'),'source_hashes':self.source_hashes}
        if provider_record is not None:
            record.update({'provider_record':provider_record,'raw_memory256':session.provider.raw_memory.tolist(),
              'normalized_memory256':session.provider.normalized_memory.tolist(),'memory_provenance':session.provider.provenance})
        if interval_audit is not None: record['provider_previous_interval']=interval_audit
        session.provider_record=provider_record;session.current_decision=session.recorder.current;session.decision_boundary=frame if boundary else session.decision_boundary
        self.next_calls+=1
        return record

def collect_technical(*,identity,binding,actor,normalizer,provider,recorder,wal,mode,max_steps,prepare:Callable[[],dict],freeze_callback=None,prepare_next:Callable[[CollectorSession],dict]|None=None,generator=None,session:CollectorSession|None=None):
    if not isinstance(identity,TechnicalFixtureIdentity): raise TypeError('TYPED_TECHNICAL_IDENTITY_REQUIRED')
    if mode==CollectorMode.QUALIFY_ZERO_STEP and max_steps!=0: raise ValueError('QUALIFICATION_MUST_BE_ZERO_STEP')
    if mode==CollectorMode.EXECUTE_BOUNDED and not 1<=max_steps<=520: raise ValueError('BOUNDED_STEPS_1_TO_520')
    legacy_static_prepare=session is None
    if session is None: session=CollectorSession(identity,binding,actor,normalizer,provider,recorder,wal,generator)
    if session.identity!=identity or session.binding is not binding or session.actor is not actor or session.provider is not provider or session.recorder is not recorder: raise ValueError('SESSION_OBJECT_DRIFT')
    start=time.monotonic(); prepared=prepare();session.initial_prepare_count+=1
    if binding.env_step!=0: raise RuntimeError('PRESTEP_ALREADY_ADVANCED')
    result={'collector_entry_count':1,'prepare_count':session.initial_prepare_count,'initial_prepare_count':session.initial_prepare_count,'next_prepare_count':0,'freeze_callback_count':0,'environment_steps':0,'mode':mode.value,'same_instance_token':session.instance_token}
    prestep_frame=wal.append('FRAME',{'frame':0,'current':prepared,'action_sha256':action_hash(prepared['submitted7']),'current_identity':{'frame':0,'identity':identity.physical_group_id,'memory_version':prepared.get('memory_version')}})
    if mode==CollectorMode.QUALIFY_ZERO_STEP:return {**result,'status':TechnicalStatus.TECHNICAL_CERTIFIED.value,'prestep_frame_commit_sha256':prestep_frame['commit_sha256'],'elapsed':time.monotonic()-start}
    # Legacy bounded harnesses may supply a static preparation callback. Scientific
    # sessions always provide prepare_next and a freeze callback before stepping.
    if freeze_callback is None: raise ValueError('EXECUTION_REQUIRES_FREEZE')
    if prepare_next is None:
        if not legacy_static_prepare or max_steps == 1: raise ValueError('EXECUTION_REQUIRES_NEXT_PREPARER')
        prepare_next=lambda _session: prepare()
    result['freeze_callback_count']=1;freeze_callback({**result,'prestep_prepared':prepared,'prestep_frame_commit_sha256':prestep_frame['commit_sha256'],'session_token':session.instance_token})
    for i in range(max_steps):
        frame=session.frame; fc=prestep_frame if frame==0 else wal.append('FRAME',{'frame':frame,'current':prepared,'action_sha256':action_hash(prepared['submitted7']),'current_identity':{'frame':frame,'identity':identity.physical_group_id,'memory_version':prepared.get('memory_version')}})
        action=prepared['submitted7']; obs,reward,terminated,truncated,info=binding.step(action);result['environment_steps']+=1
        eh=info.get('env_action7_sha256') if isinstance(info,dict) else None; ah=action_hash(action)
        if eh is not None and eh!=ah: raise RuntimeError('SUBMITTED_ENV_ACTION_HASH_MISMATCH')
        env=wal.append('ENV',{'frame':frame,'frame_commit_sha256':fc['commit_sha256'],'reward':float(reward),'raw_done':bool(info.get('raw_done',terminated or truncated)),'terminated':bool(terminated),'truncated':bool(truncated),'success':info.get('success'),'env_action7_sha256':ah,'info':info})
        terminal=bool(terminated or truncated); pending={'frame':frame,'frame_commit_sha256':fc['commit_sha256'],'env':env,'prepared':prepared,'current_identity':fc['current_identity']}
        nxt=None
        if not terminal and i+1<max_steps:
            session.pending_transition=pending
            # Expose the just executed row to a boundary provider before durable transition commit.
            session.frame=frame+1;nxt=prepare_next(session);session.next_prepare_count+=1
            _transition(session,pending,next_prepared=nxt);prepared=nxt
        else:
            if hasattr(recorder,'finish') and (terminal or i+1==max_steps): recorder.finish(frame+1,'terminal' if terminated else ('timeout' if truncated else 'technical_budget'))
            _transition(session,pending,terminal=('terminated' if terminated else 'truncated' if truncated else None),budget=(not terminal and i+1==max_steps))
        session.frame=frame+1;result['closeout_count']=i+1
        if terminal:return _final_result(result,session,provider,recorder,binding,TechnicalStatus.TERMINAL.value,wal.head['TRANSITION'])
    return _final_result(result,session,provider,recorder,binding,TechnicalStatus.TECHNICAL_BUDGET_COMPLETE.value,wal.head['TRANSITION'])

def _final_result(result,session,provider,recorder,binding,status,last):
    return {**result,'status':status,'last_transition_sha256':last,
      'initial_prepare_count':session.initial_prepare_count,'next_prepare_count':session.next_prepare_count,
      'provider_call_count':getattr(provider,'call_count',None),'decision_count':len(getattr(recorder,'decisions',[])),
      'env_steps':int(getattr(binding,'env_step',result['environment_steps'])),'closeout_count':result.get('closeout_count',0)}
