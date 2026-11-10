"""Controlled small-budget smoke segmentation experiment (PyTorch).

This is an independently implemented compact U-Net, not a reproduction of
Trans-BVM or FoSp. Four ablations share backbone initialization, sample order,
data augmentation, optimizer and model-selection protocol for each seed.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, random, time, platform
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent

class Conv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(4,co),nn.ReLU(),nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(4,co),nn.ReLU())
    def forward(self,x): return self.net(x)

class MultiScale(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(c,c,3,padding=d,dilation=d,groups=c,bias=False),nn.Conv2d(c,c//2,1,bias=False),nn.GroupNorm(4,c//2),nn.ReLU()) for d in (1,2,4)])
        self.fuse = nn.Conv2d(c//2*3,c,1,bias=False)
    def forward(self,x): return x+self.fuse(torch.cat([b(x) for b in self.branches],1))

class SmokeNet(nn.Module):
    def __init__(self, ms=False, edge=False, width=8):
        super().__init__()
        c=[width*2**i for i in range(4)]
        self.e1=Conv(3,c[0]); self.e2=Conv(c[0],c[1]); self.e3=Conv(c[1],c[2]); self.e4=Conv(c[2],c[3])
        self.d3=Conv(c[3]+c[2],c[2]); self.d2=Conv(c[2]+c[1],c[1]); self.d1=Conv(c[1]+c[0],c[0]); self.seg=nn.Conv2d(c[0],1,1)
        # Construct optional modules AFTER shared layers so shared initialization
        # is exactly equal when the random seed is reset for each ablation.
        self.ms=MultiScale(c[3]) if ms else nn.Identity()
        self.edge=nn.Sequential(nn.Conv2d(c[0],c[0],3,padding=1),nn.ReLU(),nn.Conv2d(c[0],1,1)) if edge else None
    def forward(self,x):
        a=self.e1(x); b=self.e2(F.max_pool2d(a,2)); c=self.e3(F.max_pool2d(b,2)); d=self.ms(self.e4(F.max_pool2d(c,2)))
        up=lambda z,skip: F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        z=self.d3(torch.cat([up(d,c),c],1)); z=self.d2(torch.cat([up(z,b),b],1)); z=self.d1(torch.cat([up(z,a),a],1))
        return self.seg(z), self.edge(z) if self.edge is not None else None

def boundary(y):
    return (F.max_pool2d(y,3,1,1)-(-F.max_pool2d(-y,3,1,1))).clamp(0,1)

def loss_fn(logits,edge,y,weight):
    p=logits.sigmoid()
    dice=1-((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean()
    loss=F.binary_cross_entropy_with_logits(logits,y)+dice
    if edge is not None:
        b=boundary(y)
        raw=F.binary_cross_entropy_with_logits(edge,b,reduction='none')
        # Equal positive / negative boundary contribution; denominators protect
        # all-background masks without inventing boundary pixels.
        pos=(raw*b).sum()/b.sum().clamp_min(1)
        neg=(raw*(1-b)).sum()/(1-b).sum().clamp_min(1)
        loss=loss+weight*.5*(pos+neg)
    return loss

def group_id(stem):
    # Camera-video frames use "timestamp_+offset" or "timestamp_-offset".
    import re
    m=re.match(r'^(\d{9,})_[+-]?\d+$',stem)
    if m:
        timestamp,offset=stem.split('_',1)
        return 'video_'+str(int(timestamp)-int(offset))
    # Synthetic images often reuse a background/source identifier before '_'.
    # Conservatively keep that identifier together when splitting.
    return 'source_'+stem.split('_')[0] if '_' in stem else stem

def pairs(root,split):
    images=root/split/'img'
    if not images.exists(): images=root/split/'image'
    gt=root/split/'gt'
    records=[]
    for f in sorted(images.iterdir()):
        if f.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        mask=gt/(f.stem+'.png')
        if mask.exists(): records.append({'image':str(f.relative_to(root)).replace('\\','/'),'mask':str(mask.relative_to(root)).replace('\\','/'),'group':group_id(f.stem)})
    return records

def prepare(root,train_n,val_n):
    path=HERE/'splits.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    pool=pairs(root,'train'); test=pairs(root,'test')
    test_groups={r['group'] for r in test}
    overlap_pool=[r for r in pool if r['group'] in test_groups]
    eligible=[r for r in pool if r['group'] not in test_groups]
    groups={}
    for r in eligible: groups.setdefault(r['group'],[]).append(r)
    ids=sorted(groups); rng=random.Random(20261110); rng.shuffle(ids)
    val=[]; train=[]
    for g in ids:
        target=val if len(val)<val_n else train
        target.extend(groups[g])
    rng.shuffle(train); rng.shuffle(val)
    train=train[:train_n]; val=val[:val_n]
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    # Report source overlap with official test, rather than silently hiding it.
    cross=sorted({r['group'] for r in train}&{r['group'] for r in test})
    data={'split_seed':20261110,'official_train':len(pool),'official_test':len(test),'eligible_train':len(eligible),'excluded_train_with_test_group':len(overlap_pool),'train':train,'validation':val,'test':test,'train_test_group_overlap':cross}
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return data

def load_set(root,records,size):
    xs=[]; ys=[]
    for r in records:
        im=Image.open(root/r['image']).convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
        mask=Image.open(root/r['mask']).convert('L').resize((size,size),Image.Resampling.NEAREST)
        xs.append(np.asarray(im).copy()); ys.append((np.asarray(mask)>127).astype(np.uint8))
    return torch.from_numpy(np.stack(xs)).permute(0,3,1,2).float()/255,torch.from_numpy(np.stack(ys))[:,None].float()

@torch.inference_mode()
def evaluate(model,x,y,batch=16,device='cpu',corrupt=None,save=False):
    model.eval(); tp=fp=fn=0; pm=gm=predmatch=gtmatch=0; per=[]; saved=[]
    for i in range(0,len(x),batch):
        xx=x[i:i+batch].to(device); yy=y[i:i+batch].to(device)
        if corrupt=='dark': xx=xx*.5
        if corrupt=='blur': xx=F.avg_pool2d(F.pad(xx,(2,2,2,2),mode='reflect'),5,1)
        pp=model(xx)[0].sigmoid()>.5; truth=yy>.5
        a=(pp&truth).sum((1,2,3)); b=(pp&~truth).sum((1,2,3)); c=(~pp&truth).sum((1,2,3))
        tp+=int(a.sum()); fp+=int(b.sum()); fn+=int(c.sum())
        per.extend(((a.float()+1e-7)/(a+b+c+1e-7)).cpu().tolist())
        pb=boundary(pp.float())>0; gb=boundary(yy)>0
        # Boundary F1: 1-pixel Chebyshev tolerance on 3x3 morphological bands.
        pd=F.max_pool2d(pb.float(),3,1,1)>0; gd=F.max_pool2d(gb.float(),3,1,1)>0
        pm+=int(pb.sum()); gm+=int(gb.sum()); predmatch+=int((pb&gd).sum()); gtmatch+=int((gb&pd).sum())
        if save: saved.append(pp.cpu().numpy().astype(np.uint8))
    prec=tp/max(1,tp+fp); recall=tp/max(1,tp+fn); bp=predmatch/max(1,pm); br=gtmatch/max(1,gm)
    out={'iou':tp/max(1,tp+fp+fn),'dice':2*tp/max(1,2*tp+fp+fn),'precision':prec,'recall':recall,'boundary_f1':2*bp*br/max(1e-12,bp+br),'macro_iou':float(np.mean(per)),'per_image_iou':per}
    return (out,np.concatenate(saved)) if save else out

def run(args):
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    root=Path(args.data).resolve(); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    split=prepare(root,args.train_n,args.val_n)
    print('SPLIT', {k:len(split[k]) for k in ('train','validation','test')},'official',split['official_train'],split['official_test'],'overlap',len(split['train_test_group_overlap']),flush=True)
    sets={k:load_set(root,split[k],args.size) for k in ('train','validation','test')}
    configs=[('baseline',False,False),('multiscale',True,False),('boundary',False,True),('combined',True,True)]
    metadata={'arguments':vars(args),'device':device,'torch':torch.__version__,'python':platform.python_version(),'processor':platform.processor(),'platform':platform.platform(),'split_sha256':hashlib.sha256((HERE/'splits.json').read_bytes()).hexdigest()}
    envfile=out/'environment.json'
    if envfile.exists():
        old=json.loads(envfile.read_text(encoding='utf-8'))
        keys=['size','epochs','train_n','val_n','batch','threads','lr','edge_weight','seeds','cpu']
        if old.get('split_sha256')!=metadata['split_sha256'] or any(old['arguments'].get(k)!=vars(args).get(k) for k in keys):
            raise ValueError('Existing results use a different configuration. Move results/ aside before a new experiment.')
    else:
        envfile.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    records=[]
    for seed in args.seeds:
      for name,ms,edge in configs:
        resultfile=out/f'{name}_{seed}.json'
        if resultfile.exists():
            records.append(json.loads(resultfile.read_text())); print('RESUME',name,seed,flush=True); continue
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model=SmokeNet(ms,edge).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        generator=torch.Generator().manual_seed(seed+10000); aug=torch.Generator().manual_seed(seed+20000)
        x,y=sets['train']; best=-1; history=[]; started=time.perf_counter()
        for ep in range(1,args.epochs+1):
            t=time.perf_counter(); model.train(); order=torch.randperm(len(x),generator=generator); total=0
            for start in range(0,len(x),args.batch):
                idx=order[start:start+args.batch]; xx=x[idx].to(device); yy=y[idx].to(device)
                if torch.rand((),generator=aug).item()<.5: xx=xx.flip(-1); yy=yy.flip(-1)
                gain=.8+.4*torch.rand((),generator=aug).item(); xx=(xx*gain).clamp(0,1)
                opt.zero_grad(set_to_none=True); logits,ed=model(xx); loss=loss_fn(logits,ed,yy,args.edge_weight); loss.backward(); opt.step(); total+=float(loss.detach())*len(idx)
            val=evaluate(model,*sets['validation'],device=device)
            history.append({'epoch':ep,'train_loss':total/len(x),'validation_iou':val['iou'],'seconds':time.perf_counter()-t})
            if val['iou']>best:
                best=val['iou']; bestep=ep
                torch.save({'state_dict':model.state_dict(),'config':{'ms':ms,'edge':edge},'epoch':ep,'seed':seed,'size':args.size},out/f'{name}_{seed}.pt')
            print(f'{name} seed={seed} epoch={ep}/{args.epochs} loss={total/len(x):.4f} valIoU={val["iou"]:.4f} sec={time.perf_counter()-t:.1f}',flush=True)
        model.load_state_dict(torch.load(out/f'{name}_{seed}.pt',map_location=device,weights_only=True)['state_dict'])
        clean,pred=evaluate(model,*sets['test'],device=device,save=True)
        np.savez_compressed(out/f'{name}_{seed}_predictions.npz',masks=pred)
        dark=evaluate(model,*sets['test'],device=device,corrupt='dark'); blur=evaluate(model,*sets['test'],device=device,corrupt='blur')
        model.eval(); sample=sets['test'][0][:1].to(device)
        with torch.inference_mode():
            for _ in range(10): model(sample)
            times=[]
            for _ in range(50):
                if device=='cuda': torch.cuda.synchronize()
                t=time.perf_counter(); model(sample)
                if device=='cuda': torch.cuda.synchronize()
                times.append((time.perf_counter()-t)*1000)
        record={'name':name,'seed':seed,'best_epoch':bestep,'best_validation_iou':best,'parameters':sum(p.numel() for p in model.parameters()),'median_latency_ms':float(np.median(times)),'train_seconds':time.perf_counter()-started,'history':history,'clean':clean,'dark':dark,'blur':blur}
        resultfile.write_text(json.dumps(record,indent=2),encoding='utf-8'); records.append(record)
        print('RESULT',name,seed,{k:round(v,4) for k,v in clean.items() if k!='per_image_iou'},flush=True)
    (out/'all_results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['model','seed','epoch','IoU','Dice','Precision','Recall','BoundaryF1','DarkIoU','BlurIoU','Parameters','LatencyMs'])
        for r in records: w.writerow([r['name'],r['seed'],r['best_epoch'],*[r['clean'][k] for k in ['iou','dice','precision','recall','boundary_f1']],r['dark']['iou'],r['blur']['iou'],r['parameters'],r['median_latency_ms']])
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(HERE/'data'/'smoke5k-main'))
    p.add_argument('--output-dir',default=str(HERE/'results'),help='Directory for checkpoints, logs and metrics; use a new directory to retrain.')
    p.add_argument('--size',type=int,default=128); p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--train-n',type=int,default=400); p.add_argument('--val-n',type=int,default=100)
    p.add_argument('--batch',type=int,default=16); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--edge-weight',type=float,default=.2)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,2026]); p.add_argument('--cpu',action='store_true')
    run(p.parse_args())
