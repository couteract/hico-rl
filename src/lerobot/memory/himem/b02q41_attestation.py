"""Durable pre/post model and RNG attestations for B02Q41 technical execution."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
from typing import Any

def _raw_tensor_bytes(t):
    import torch
    x=t.detach().contiguous().cpu()
    return x.view(torch.uint8).numpy().tobytes(),str(x.dtype),list(x.shape)

def encode_generator_state(generator):
    b,dtype,shape=_raw_tensor_bytes(generator.get_state())
    return {'schema':'B02Q41_GENERATOR_STATE_V1','device':str(generator.device),'dtype':dtype,'shape':shape,'length':len(b),'sha256':hashlib.sha256(b).hexdigest(),'bytes_hex':b.hex()}

def decode_generator_state(doc, *, device):
    import numpy as np,torch
    if doc.get('schema')!='B02Q41_GENERATOR_STATE_V1' or doc.get('device')!=str(torch.device(device)):raise ValueError('RNG_STATE_DEVICE')
    try:b=bytes.fromhex(doc['bytes_hex'])
    except Exception as e:raise ValueError('RNG_STATE_ENCODING') from e
    if len(b)!=doc.get('length') or hashlib.sha256(b).hexdigest()!=doc.get('sha256'):raise ValueError('RNG_STATE_HASH_OR_LENGTH')
    if doc.get('dtype')!='torch.uint8' or doc.get('shape')!=[len(b)]:raise ValueError('RNG_STATE_DTYPE_OR_SHAPE')
    state=torch.from_numpy(np.frombuffer(b,dtype=np.uint8).copy())
    g=torch.Generator(device=device)
    try:g.set_state(state)
    except Exception as e:raise ValueError('RNG_STATE_INVALID_FOR_DEVICE') from e
    return g

def atomic_json(path:Path,doc:dict[str,Any]):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);dfd=os.open(path.parent,os.O_DIRECTORY);os.fsync(dfd);os.close(dfd)
    except Exception:
        try:os.unlink(tmp)
        except FileNotFoundError:pass
        raise

def tensor_hash(items):
    h=hashlib.sha256()
    for name,t in sorted(items):
        b,dtype,shape=_raw_tensor_bytes(t);h.update(name.encode()+b'\0'+dtype.encode()+b'\0'+repr(shape).encode()+b'\0'+b)
    return h.hexdigest()

def module_attestation(module):
    return {'parameters_sha256':tensor_hash(module.named_parameters()),'buffers_sha256':tensor_hash(module.named_buffers()),
      'training':bool(module.training),'grad_tensors':sum(int(p.grad is not None) for p in module.parameters())}

def validate_parameter_buffer_invariance(pre,post):
    for model in ('actor','vla','provider'):
        if pre['models'][model]['parameters_sha256']!=post['models'][model]['parameters_sha256']:raise ValueError('POST_PARAMETER_HASH_MISMATCH')
        if pre['models'][model]['buffers_sha256']!=post['models'][model]['buffers_sha256']:raise ValueError('POST_BUFFER_HASH_MISMATCH')

def validate_rng_attribution(pre,post):
    if not post.get('rng_attribution',{}).get('match'):raise ValueError('RNG_ATTRIBUTION_MISMATCH')
    if post['rng_attribution'].get('replayed_proposal_count')!=post['proposal_count']-post['proposal_count_at_pre']:raise ValueError('RNG_PROPOSAL_COUNT_MISMATCH')
    for stream in ('python','numpy_legacy','torch_cpu','torch_cuda_global'):
        if pre['rng'][stream]!=post['rng'][stream]:raise ValueError('UNATTRIBUTED_GLOBAL_RNG_DRIFT:'+stream)
    if post['rng_attribution'].get('expected_actor_generator_sha256')!=post['rng']['actor_generator']['sha256']:raise ValueError('RNG_ATTRIBUTION_MISMATCH')

def write_post_raw_then_attribution(directory,raw_post,attribute):
    """Commit raw POST before executing attribution; attribution failure preserves it."""
    directory=Path(directory);raw_path=directory/'POST_STATE_RAW.json';atomic_json(raw_path,raw_post)
    att=attribute(raw_post);atomic_json(directory/'POST_RNG_ATTRIBUTION.json',att)
    return raw_path,directory/'POST_RNG_ATTRIBUTION.json'
