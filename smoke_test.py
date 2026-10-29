import importlib.util, sys
from pathlib import Path
import numpy as np
from common import CipherConfig
ROOT=Path(__file__).resolve().parent

def load(f,n):
    s=importlib.util.spec_from_file_location(n,ROOT/f); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
c1=load('confusion-1.py','c1'); c2=load('confusion-2.py','c2'); d1=load('diffusion-1.py','d1'); d2=load('diffusion-2.py','d2')
a=np.arange(16*16,dtype=np.uint8).reshape(16,16)
cfg=CipherConfig(discard_first_100=False)
i1,t1=c1.run(a,cfg); i2,t2=c2.run(i1,t1['seq1'],cfg); i3,t3=d1.run(i2,cfg); i4,t4=d2.run(i3)
assert all(x.dtype==np.uint8 for x in (i1,i2,i3,i4))
assert i1.shape==a.shape and i2.shape==a.shape and i3.shape==a.shape and i4.shape==a.shape
assert np.array_equal(np.sort(a.ravel()),np.sort(i1.ravel()))
print('smoke test passed',a.shape,i4.shape)
