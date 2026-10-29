"""MATLAB-style visualizations and a generated MATLAB replay script."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common import difference_image


def _gray_for_view(a):
    x=np.asarray(a,dtype=np.uint8)
    return x


def run(stages: dict, trace: dict, output_dir: Path, show: bool=False):
    output_dir.mkdir(parents=True,exist_ok=True)
    labels=[("original","I0 original"),("confusion1","I1 confusion-1"),
            ("confusion2","I2 confusion-2"),("diffusion1","I3 diffusion-1"),
            ("diffusion2","I4 diffusion-2"),("cipher","C cipher")]

    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for ax,(key,label) in zip(axes.ravel(),labels):
        a=_gray_for_view(stages[key])
        if a.ndim==2: ax.imshow(a,cmap="gray",vmin=0,vmax=255)
        else: ax.imshow(a)
        ax.set_title(label); ax.axis("off")
    fig.suptitle("Paper cipher stages — MATLAB-style stage view")
    fig.tight_layout(); fig.savefig(output_dir/"matlab_stage_view.png",dpi=180)
    if show: plt.show()
    else: plt.close(fig)

    # Difference maps from one stage to the next.
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    pairs=[("original","confusion1","I0 -> I1"),("confusion1","confusion2","I1 -> I2"),
           ("confusion2","diffusion1","I2 -> I3"),("diffusion1","diffusion2","I3 -> I4")]
    for ax,(a,b,title) in zip(axes,pairs):
        d=difference_image(stages[a],stages[b])
        ax.imshow(d,cmap="gray",vmin=0,vmax=255); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(output_dir/"matlab_change_maps.png",dpi=180); plt.close(fig)

    seq1=np.asarray(trace["seq1"]); seq2=np.asarray(trace["seq2"]); seq3=np.asarray(trace["seq3"])
    idx=np.asarray(trace["perm_idx_matlab"]); sorted1=np.asarray(trace["sorted_seq1"])
    for name,arr in (("yn1",seq1),("yn2",seq2),("yn3",seq3),("yn1_sorted",sorted1),("y1ind",idx)):
        np.savetxt(output_dir/f"{name}.csv",arr,delimiter=",")

    fig,axes=plt.subplots(3,1,figsize=(11,10))
    axes[0].plot(np.arange(1,len(seq1)+1),seq1,linewidth=.55); axes[0].set_title("yn1")
    axes[1].plot(np.arange(1,len(seq2)+1),seq2,linewidth=.7); axes[1].set_title("yn2")
    axes[2].plot(np.arange(1,len(seq3)+1),seq3,linewidth=.55); axes[2].set_title("yn3")
    for ax in axes: ax.grid(True,alpha=.3); ax.set_xlabel("Iteration")
    fig.tight_layout(); fig.savefig(output_dir/"matlab_chaotic_sequences.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(11,8))
    axes[0].plot(np.arange(1,len(sorted1)+1),sorted1,linewidth=.7); axes[0].set_title("yn1 sorted ascending")
    axes[1].plot(np.arange(1,len(idx)+1),idx,".",markersize=1); axes[1].set_title("y1ind (MATLAB 1-based index)")
    for ax in axes: ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(output_dir/"matlab_sort_and_index.png",dpi=180); plt.close(fig)

    matlab = r'''%% Auto-generated plotting companion for the Python run
clear; clc; close all;
yn1 = readmatrix('yn1.csv');
yn2 = readmatrix('yn2.csv');
yn3 = readmatrix('yn3.csv');
yn1_sorted = readmatrix('yn1_sorted.csv');
y1ind = readmatrix('y1ind.csv');

figure; plot(yn1); grid on; title('yn1'); xlabel('Iteration'); ylabel('Value');
figure; plot(yn2); grid on; title('yn2'); xlabel('Iteration'); ylabel('Value');
figure; plot(yn3); grid on; title('yn3'); xlabel('Iteration'); ylabel('Value');
figure; plot(yn1_sorted); grid on; title('yn1 sorted ascending');
figure; plot(y1ind,'.'); grid on; title('y1ind permutation index');

I0 = imread('../01_original.png');
I1 = imread('../02_confusion_1.png');
I2 = imread('../03_confusion_2.png');
I3 = imread('../04_diffusion_1.png');
I4 = imread('../05_diffusion_2.png');
C  = imread('../06_cipher_image.png');
figure;
subplot(2,3,1); imshow(I0); title('I0 original');
subplot(2,3,2); imshow(I1); title('I1 confusion-1');
subplot(2,3,3); imshow(I2); title('I2 confusion-2');
subplot(2,3,4); imshow(I3); title('I3 diffusion-1');
subplot(2,3,5); imshow(I4); title('I4 diffusion-2');
subplot(2,3,6); imshow(C); title('C cipher');
'''
    (output_dir/"matlab_replay.m").write_text(matlab,encoding="utf-8")
