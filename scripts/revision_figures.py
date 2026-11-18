"""Make compact print-readable revised figures using the archived full results."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.plot_outputs import severity_matrix
REF=ROOT/'reference_results'/'tables'
OUT=ROOT/'revision_results'/'figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none'})
# Figure 4: abbreviated axis labels preserve readable type at a 3.5-inch width.
frames=[pd.read_csv(REF/'baseline_summary.csv'),pd.read_csv(REF/'ablation_summary.csv'),pd.read_csv(REF/'hidden_width_summary.csv'),pd.read_csv(REF/'regularization_summary.csv')]
frames[1]=frames[1][frames[1].analysis_type=='cumulative']
labels=[['LDA4','MLP4','LR28','SVM28','RF28','MLP28'],frames[1].feature_count.astype(str).tolist(),frames[2].hidden_units.astype(str).tolist(),['None','L2','L2+ES']]
fig,axes=plt.subplots(2,2,figsize=(3.5,2.65))
for i,(ax,f,l,title) in enumerate(zip(axes.flat,frames,labels,['(a) Classifiers','(b) Feature count','(c) Hidden width','(d) Regularization'])):
    x=np.arange(len(f));ax.bar(x,f.test_accuracy_mean*100,color=['#3975a5','#65936b','#ae7f3b','#866a9e'][i],width=.65)
    ax.errorbar(x,f.test_accuracy_mean*100,yerr=f.test_accuracy_sd*100,fmt='none',color='black',capsize=1.8,elinewidth=.6)
    ax.set_ylim(72,100);ax.set_yticks([75,85,95]);ax.set_xticks(x,l,rotation=60 if i==0 else 0,ha='right' if i==0 else 'center')
    ax.tick_params(axis='x',labelsize=6.5 if i==0 else 7,length=2,pad=1)
    ax.set_title(title,loc='left',pad=3);ax.spines[['top','right']].set_visible(False)
    if i in [0,2]:ax.set_ylabel('Accuracy (%)',labelpad=2)
fig.tight_layout(pad=.5,h_pad=.7,w_pad=.7)
fig.savefig(OUT/'fig4_revised.png',dpi=600);fig.savefig(OUT/'fig4_revised.svg');plt.close(fig)
# Figure 5: use class IDs (defined in Table I) instead of long rotated labels.
# Exact count matrix remains in reference_results/tables/pooled_confusion_matrix.csv.
c=pd.read_csv(REF/'pooled_confusion_matrix.csv',index_col=0).to_numpy(dtype=int)
_,pct=severity_matrix(c)
fig=plt.figure(figsize=(3.5,2.3))
a=fig.add_axes([.095,.24,.51,.68]);b=fig.add_axes([.70,.34,.28,.47])
im=a.imshow(c,cmap='Blues',vmin=0,vmax=240,aspect='equal')
a.set_xticks(range(13),[str(i) for i in range(13)],rotation=90);a.set_yticks(range(13))
a.tick_params(length=0,pad=1,labelsize=7)
a.set_xlabel('Predicted class',labelpad=2);a.set_ylabel('True class',labelpad=1)
a.set_title('(a) Pooled counts',loc='left',pad=4)
cbax=fig.add_axes([.125,.08,.45,.035]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[0,120,240]);cb.ax.tick_params(labelsize=7,length=2,pad=1)
b.imshow(pct,cmap='Blues',vmin=0,vmax=100,aspect='equal')
b.set_xticks(range(4),['I','10','20','30']);b.set_yticks(range(4),['I','10','20','30'])
b.tick_params(length=0,pad=1,labelsize=7)
b.set_xlabel('Predicted (%)',labelpad=2);b.set_title('(b) Severity',loc='left',pad=4)
for i in range(4):
    for j in range(4):
        val=pct[i,j];b.text(j,i,f'{val:.0f}' if val>=99.95 or val==0 else f'{val:.1f}',ha='center',va='center',fontsize=6.8,color='white' if val>=50 else 'black')
fig.text(.70,.19,'Rows: true severity\nI: intact; cells: %',fontsize=7,va='top')
for ax in [a,b]:
    for sp in ax.spines.values():sp.set_visible(False)
fig.savefig(OUT/'fig5_revised.png',dpi=600);fig.savefig(OUT/'fig5_revised.svg');plt.close(fig)
