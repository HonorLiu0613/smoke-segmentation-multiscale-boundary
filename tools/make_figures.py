from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
RESULTS=ROOT/'code'/'results'
data=np.load(RESULTS/'research_visuals.npz')
records=json.loads((RESULTS/'all_results.json').read_text())
KEYS=['baseline','multiscale','boundary','combined']
LABELS=['B','B+M','B+E','B+M+E']
COLORS=['#464646','#2876ad','#d38a35','#23836f']
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.labelsize':13,'axes.linewidth':.8,'xtick.labelsize':11,'ytick.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','pdf.fonttype':42})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=350,bbox_inches='tight',pad_inches=.05)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.05)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,2.9))
bins=np.geomspace(.005,100,27)
for name,label,color in zip(['train','validation','test'],['Training','Validation','Test'],[COLORS[1],COLORS[2],COLORS[3]]):
    area=data[name+'_area']*100;positive=area[area>0]
    ax.hist(positive,bins=bins,weights=np.full(len(positive),100/len(positive)),histtype='step',lw=1.8,label=f'{label} (n={len(positive)})',color=color)
ax.set_xscale('log');ax.set_xlim(.005,100);ax.set_xticks([.01,.1,1,10,100],labels=['0.01','0.1','1','10','100'])
ax.set_xlabel('Smoke foreground area (%)');ax.set_ylabel('Images per bin (%)');ax.legend(frameon=False,ncol=3,fontsize=12,loc='upper left');ax.grid(axis='y',alpha=.15)
fig.tight_layout();save(fig,'foreground_distribution')

fig,ax=plt.subplots(figsize=(8.2,2.9))
for key,label,color in zip(KEYS,LABELS,COLORS):
    rr=sorted([r for r in records if r['name']==key],key=lambda r:r['seed'])
    values=np.asarray([[h['validation_iou']*100 for h in r['history']] for r in rr]);epochs=np.arange(1,values.shape[1]+1)
    for row in values:ax.plot(epochs,row,lw=.8,alpha=.22,color=color)
    ax.plot(epochs,values.mean(0),lw=1.8,marker='o',ms=3,label=label,color=color)
ax.set_xlabel('Epoch');ax.set_ylabel('Validation smoke IoU (%)');ax.set_xticks(range(1,13));ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.legend(ncol=4,frameon=False,loc='upper left',fontsize=12)
fig.tight_layout();save(fig,'validation_curves')

fig,axes=plt.subplots(4,5,figsize=(10,7.8))
titles=['(a) RGB image','(b) Ground truth','(c) B smoke P','(d) Joint smoke P','(e) Joint edge P']
for row,idx in enumerate(data['selected_indices']):
    imgs=[data['images'][row],data['selected_ground_truth'][row],data['baseline_probability'][row],data['combined_probability'][row],data['edge_probability'][row]]
    for col,arr in enumerate(imgs):
        ax=axes[row,col]
        if col==0:ax.imshow(arr)
        elif col==1:ax.imshow(arr,cmap='gray',vmin=0,vmax=1)
        else:ax.imshow(arr,cmap='inferno',vmin=0,vmax=1,interpolation='nearest')
        if row<2:
            yy,xx=np.where(data['selected_ground_truth'][row]>0)
            side=max(32,int(max(xx.max()-xx.min(),yy.max()-yy.min()))+12)
            x0=int(np.clip((xx.max()+xx.min()-side)/2,0,128-side));y0=int(np.clip((yy.max()+yy.min()-side)/2,0,128-side))
            ax.add_patch(Rectangle((x0,y0),side,side,fill=False,edgecolor='#edcf5b',linewidth=.65,linestyle='--'))
            if col>0:
                inset=ax.inset_axes([.60,.025,.375,.375])
                inset.imshow(arr[y0:y0+side,x0:x0+side],cmap='gray' if col==1 else 'inferno',vmin=0,vmax=1,interpolation='nearest')
                inset.set_xticks([]);inset.set_yticks([])
                for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#edcf5b');spine.set_linewidth(.7)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    axes[row,0].set_ylabel(f'Test #{int(idx)+1}\nArea {data["test_area"][idx]*100:.2f}%',fontsize=14)
for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=13,pad=7)
fig.subplots_adjust(left=.09,right=.92,top=.95,bottom=.03,wspace=.04,hspace=.07)
cax=fig.add_axes([.94,.18,.013,.63]);bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='inferno'),cax=cax);bar.set_label('Predicted probability',fontsize=13);bar.set_ticks([0,.25,.5,.75,1]);bar.ax.tick_params(labelsize=11)
save(fig,'probability_heatmaps')

valid=data['test_area']>0
paired={key:np.mean([r['clean']['per_image_iou'] for r in records if r['name']==key],axis=0)[valid]*100 for key in KEYS}
area=data['test_area'][valid]*100
delta=paired['combined']-paired['baseline']
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.4))
for key,label,color in zip(KEYS,LABELS,COLORS):
    x=np.sort(paired[key]);y=np.arange(1,len(x)+1)/len(x)
    ax1.step(np.r_[0,x],np.r_[0,y],where='post',lw=1.8,label=label,color=color)
ax1.set_xlim(0,100);ax1.set_ylim(0,1.02);ax1.set_xlabel('Per-image smoke IoU (%)');ax1.set_ylabel('Empirical cumulative fraction');ax1.legend(frameon=False,fontsize=11,loc='lower right');ax1.grid(alpha=.15);ax1.set_title('(a) Test-image performance distribution',fontsize=13)
order=np.argsort(area,kind='stable');parts=np.array_split(order,4);vectors=[delta[part] for part in parts]
bp=ax2.boxplot(vectors,widths=.5,showfliers=False,patch_artist=True,medianprops={'color':'#222222','linewidth':1.5},boxprops={'linewidth':1},whiskerprops={'linewidth':1},capprops={'linewidth':1})
rng=np.random.default_rng(20261110)
quartile_summary=[]
for j,(part,box) in enumerate(zip(parts,bp['boxes'])):
    box.set_facecolor('#d7e8e2');box.set_edgecolor(COLORS[3]);ax2.scatter(j+1+rng.uniform(-.19,.19,len(part)),delta[part],s=7,color=COLORS[3],alpha=.28,lw=0,zorder=2)
    quartile_summary.append({'quartile':j+1,'n':len(part),'min_area_percent':float(area[part].min()),'max_area_percent':float(area[part].max()),'median_delta_iou_pp':float(np.median(delta[part])),'mean_delta_iou_pp':float(np.mean(delta[part]))})
ax2.axhline(0,color='#777777',ls='--',lw=.9);ax2.set_xticks(range(1,5),labels=[f'Q{i+1}\n(n={len(p)})' for i,p in enumerate(parts)]);ax2.set_xlabel('Ground-truth smoke-area rank quartile');ax2.set_ylabel('Joint minus baseline IoU (pp)');ax2.set_title('(b) Paired differences by smoke size',fontsize=13);ax2.grid(axis='y',alpha=.15)
fig.tight_layout(w_pad=2);save(fig,'performance_distribution')
summary={'positive_test_images':int(valid.sum()),'aggregation':'per-image IoU averaged across seeds 42 and 2026 before plotting','improved_images':int((delta>1e-6).sum()),'degraded_images':int((delta< -1e-6).sum()),'tied_images':int((np.abs(delta)<=1e-6).sum()),'quartiles':quartile_summary}
(RESULTS/'distribution_analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
(OUT/'prediction_samples.json').write_text((RESULTS/'visualization_protocol.json').read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(summary,indent=2),flush=True)
