"""Export real model probabilities and data distributions for research figures."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from smoke_experiment import SmokeNet

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'smoke5k-main'
OUT=ROOT/'results'
split=json.loads((ROOT/'splits.json').read_text(encoding='utf8'))
fractions={}
test_masks=[]
for name in ['train','validation','test']:
    arr=[]
    for r in split[name]:
        mask=np.asarray(Image.open(DATA/r['mask']).convert('L').resize((128,128),Image.Resampling.NEAREST))>127
        arr.append(float(mask.mean()))
        if name=='test':test_masks.append(mask)
    fractions[name]=np.asarray(arr,dtype=np.float32)
order=sorted(range(len(test_masks)),key=lambda i:(fractions['test'][i],i))
quantiles=[.1,.5,.9,.99]
indices=[order[round(q*(len(order)-1))] for q in quantiles]
images=[]
for i in indices:
    images.append(np.asarray(Image.open(DATA/split['test'][i]['image']).convert('RGB').resize((128,128),Image.Resampling.BILINEAR)).copy())
images=np.stack(images)
x=torch.from_numpy(images).permute(0,3,1,2).float()/255
torch.set_num_threads(4)
outputs={}
for name in ['baseline','combined']:
    ck=torch.load(OUT/f'{name}_42.pt',map_location='cpu',weights_only=True)
    net=SmokeNet(**ck['config']);net.load_state_dict(ck['state_dict']);net.eval()
    with torch.inference_mode():
        seg,edge=net(x)
        outputs[name+'_probability']=seg.sigmoid()[:,0].numpy()
        if edge is not None:outputs['edge_probability']=edge.sigmoid()[:,0].numpy()
np.savez_compressed(OUT/'research_visuals.npz',images=images,selected_indices=np.asarray(indices),selected_ground_truth=np.stack(test_masks)[indices].astype(np.uint8),**outputs,**{k+'_area':v for k,v in fractions.items()})
protocol={'seed':42,'quantiles':quantiles,'indices':indices,'probability_range':[0,1],'outputs':'sigmoid probabilities from segmentation and auxiliary boundary heads','samples':[{'image':split['test'][i]['image'],'mask':split['test'][i]['mask'],'foreground_fraction':float(fractions['test'][i]),'quantile':q} for i,q in zip(indices,quantiles)]}
(OUT/'visualization_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
print('Exported model probabilities and all split foreground-area fractions',indices,flush=True)
