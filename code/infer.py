from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
from smoke_experiment import SmokeNet

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--image',required=True)
p.add_argument('--output',default='prediction.png')
a=p.parse_args()
torch.set_num_threads(4)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
net=SmokeNet(**ck['config']); net.load_state_dict(ck['state_dict']); net.eval()
im=Image.open(a.image).convert('RGB'); size=ck['size']
x=torch.from_numpy(np.asarray(im.resize((size,size),Image.Resampling.BILINEAR)).copy()).permute(2,0,1)[None].float()/255
with torch.inference_mode(): mask=(net(x)[0].sigmoid()[0,0].numpy()>.5).astype(np.uint8)*255
Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST).save(a.output)
print('Saved',Path(a.output).resolve())
