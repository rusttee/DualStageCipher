"""Stage-by-stage analysis, including comparison with values printed in the paper."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common import ROOT, image_change_stats

COLOR_PAPER_NAME={"4.2.03":"Mandril","4.2.07":"Peppers","4.2.06":"Lake","4.2.05":"Airplane"}


def _entropy(ch):
    x=np.asarray(ch,dtype=np.uint8).ravel(); h=np.bincount(x,minlength=256).astype(float); p=h[h>0]/x.size
    return float(-np.sum(p*np.log2(p)))

def _chi(ch):
    x=np.asarray(ch,dtype=np.uint8); h=np.bincount(x.ravel(),minlength=256).astype(float); e=x.size/256.0
    return float(np.sum((h-e)**2/e))

def _corr(x,y):
    x=np.asarray(x,dtype=float).ravel(); y=np.asarray(y,dtype=float).ravel()
    if x.size<2 or x.std()==0 or y.std()==0:return float("nan")
    return float(np.corrcoef(x,y)[0,1])

def _adj(a):
    a=np.asarray(a)
    return {"horizontal":_corr(a[:,:-1],a[:,1:]),"vertical":_corr(a[:-1,:],a[1:,:]),"diagonal":_corr(a[:-1,:-1],a[1:,1:])}

def _chan(a): return {"entropy":_entropy(a),"chi_square":_chi(a),"correlation":_adj(a)}

def _img(a):
    x=np.asarray(a)
    if x.ndim==2:return {"gray":_chan(x)}
    return {n:_chan(x[...,i]) for i,n in enumerate(("R","G","B"))}

def _mse_psnr(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if a.shape!=b.shape:return None,None
    mse=float(np.mean((a-b)**2)); psnr=float("inf") if mse==0 else 10*math.log10(255.0**2/mse)
    return mse,psnr


def _paper_rows():
    p=ROOT/'paper_reference_values.csv'
    if not p.exists(): return []
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def _paper_lookup(source):
    image_id=(source or {}).get('image_id')
    if not image_id:return {}
    target=image_id if image_id.startswith('5.') else COLOR_PAPER_NAME.get(image_id)
    out={}
    for r in _paper_rows():
        image_name=r['image']
        hit=(image_name.split(' ')[0]==image_id) if image_id.startswith('5.') else (image_name==target)
        if hit: out[(r['metric'],r['channel'] or 'gray')]=float(r['paper_value'])
    return out


def _paper_compare(stages,source):
    refs=_paper_lookup(source)
    if not refs:return {}
    orig=np.asarray(stages['original']); cipher=np.asarray(stages['cipher'])
    chans=[('gray',orig,cipher)] if orig.ndim==2 else [(n,orig[...,i],cipher[...,i]) for i,n in enumerate(('R','G','B'))]
    rows=[]
    for name,a,c in chans:
        current={
            'plain_entropy':_entropy(a),'cipher_entropy':_entropy(c),'chi_square':_chi(c),
            'plain_corr_horizontal':_adj(a)['horizontal'],'plain_corr_vertical':_adj(a)['vertical'],'plain_corr_diagonal':_adj(a)['diagonal'],
            'cipher_corr_horizontal':_adj(c)['horizontal'],'cipher_corr_vertical':_adj(c)['vertical'],'cipher_corr_diagonal':_adj(c)['diagonal'],
        }
        mse,psnr=_mse_psnr(a,c); current['MSE']=mse; current['PSNR']=psnr
        for metric,ours in current.items():
            key=(metric,name)
            if key in refs:
                rows.append({'channel':name,'metric':metric,'ours':ours,'paper':refs[key],'delta':None if ours is None else ours-refs[key]})
    return {'note':'Paper comparisons are descriptive only. Exact equality is not expected while the Level-2 confusion indexing and Eq.(5) byte quantization remain under-specified; color 512->256 preprocessing is also not published.', 'rows':rows}


def run(stages:dict,output_dir:Path,source_info:dict|None=None):
    output_dir.mkdir(parents=True,exist_ok=True)
    order=['original','confusion1','confusion2','diffusion1','diffusion2','cipher']
    report={'source':source_info or {},'stage_changes':{},'metrics':{}}
    for k in order:
        if k in stages:report['metrics'][k]=_img(stages[k])
    for a,b in zip(order,order[1:]):
        if a in stages and b in stages:report['stage_changes'][f'{a}_to_{b}']=image_change_stats(stages[a],stages[b])
    mse,psnr=_mse_psnr(stages['original'],stages['cipher']);report['plain_vs_cipher']={'MSE':mse,'PSNR_dB':psnr}
    report['paper_comparison']=_paper_compare(stages,source_info or {})
    (output_dir/'analysis_report.json').write_text(json.dumps(report,indent=2,allow_nan=True),encoding='utf-8')

    lines=['NATURE CIPHER — STAGE ANALYSIS','='*72]
    for name,s in report['stage_changes'].items():
        lines+=['',name]
        if s.get('shape_changed'):lines.append(f"  shape changed: {s['before_shape']} -> {s['after_shape']}")
        else:
            lines.append(f"  changed pixels: {s['changed_pixels']} / {s['total_pixels']} ({s['changed_pixels_percent']:.6f}%)")
            lines.append(f"  changed components: {s['changed_components']} / {s['total_components']} ({s['changed_components_percent']:.6f}%)")
    lines+=['','Original vs cipher',f"  MSE: {mse}",f"  PSNR: {psnr} dB"]
    comp=report['paper_comparison']
    if comp:
        lines+=['','PAPER VALUE COMPARISON',comp['note']]
        for r in comp['rows']:
            lines.append(f"  {r['channel']:>4} {r['metric']:<24} ours={r['ours']:.8f} paper={r['paper']:.8f} delta={r['delta']:+.8f}")
    lines+=['','Full entropy / chi-square / correlation data are in analysis_report.json.']
    (output_dir/'analysis_report.txt').write_text('\n'.join(lines),encoding='utf-8')

    # comparison CSV
    if comp:
        with (output_dir/'paper_comparison.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['channel','metric','ours','paper','delta']);w.writeheader();w.writerows(comp['rows'])

    orig=np.asarray(stages['original']); cipher=np.asarray(stages['cipher'])
    fig,axes=plt.subplots(2,1,figsize=(10,8))
    if orig.ndim==2:
        axes[0].hist(orig.ravel(),bins=256,range=(0,256));axes[1].hist(cipher.ravel(),bins=256,range=(0,256))
    else:
        for i,label in enumerate(('R','G','B')):
            axes[0].hist(orig[...,i].ravel(),bins=256,range=(0,256),histtype='step',label=label)
            axes[1].hist(cipher[...,i].ravel(),bins=256,range=(0,256),histtype='step',label=label)
        axes[0].legend();axes[1].legend()
    axes[0].set_title('Original image histogram');axes[1].set_title('Cipher image histogram')
    for ax in axes:ax.grid(True,alpha=.25);ax.set_xlim(0,255)
    fig.tight_layout();fig.savefig(output_dir/'analysis_histograms.png',dpi=170);plt.close(fig)
    return report
