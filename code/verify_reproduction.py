"""Check the experiment's invariants and numerical edge cases."""
from pathlib import Path
import json, hashlib
import torch
from smoke_experiment import SmokeNet,loss_fn,boundary,evaluate,group_id

torch.set_num_threads(2)
reference=None
for ms,edge in [(False,False),(True,False),(False,True),(True,True)]:
    torch.manual_seed(42); net=SmokeNet(ms,edge)
    shared={k:v.clone() for k,v in net.state_dict().items() if not k.startswith(('ms.','edge.'))}
    if reference is None: reference=shared
    else: assert all(torch.equal(reference[k],v) for k,v in shared.items())
    x=torch.rand(2,3,32,32); y=torch.zeros(2,1,32,32); y[0,:,8:24,8:24]=1
    pred,ed=net(x); assert pred.shape==y.shape
    loss=loss_fn(pred,ed,y,.2); assert torch.isfinite(loss)
    loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    if edge: assert net.edge[-1].weight.grad.abs().sum()>0
    print('PASS shape, initialization, finite gradient:',ms,edge)

assert group_id('1528757586_+00180')==group_id('1528758126_+00720')
assert boundary(torch.zeros(1,1,16,16)).sum()==0
class Perfect(torch.nn.Module):
    def forward(self,x): return (x[:,0:1]-.5)*40,None
y=torch.zeros(2,1,32,32); y[0,:,8:24,8:24]=1
m=evaluate(Perfect(),y.repeat(1,3,1,1),y)
assert all(abs(m[k]-1)<1e-6 for k in ['iou','dice','precision','recall','boundary_f1'])
print('PASS exact region/boundary metrics and empty-mask loss handling')

root=Path(__file__).resolve().parent
s=json.loads((root/'splits.json').read_text(encoding='utf8'))
groups={k:{r['group'] for r in s[k]} for k in ['train','validation','test']}
assert not groups['train']&groups['validation']
assert not groups['train']&groups['test']
assert not groups['validation']&groups['test']
data=root/'data'/'smoke5k-main'
hashes={}
for split in groups:
    for r in s[split]:
        assert (data/r['image']).is_file() and (data/r['mask']).is_file()
        digest=hashlib.sha256((data/r['image']).read_bytes()).hexdigest()
        hashes.setdefault(digest,set()).add(split)
duplicates=[v for v in hashes.values() if len(v)>1]
assert not duplicates, 'Identical image files occur across splits'
print('PASS groups, paths, and image byte hashes across all three splits')
