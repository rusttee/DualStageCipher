from pathlib import Path
import json, sys
from common import ROOT,SIPI_CATALOG,download_sipi,asset_quality,MEDICAL
print('='*78); print('OFFLINE PACKAGE CHECK'); print('='*78)
ok=True
for iid,s in SIPI_CATALOG.items():
    try:
        p=download_sipi(iid); q=asset_quality(iid,p); print(f'{iid:7s} {s["name"]:24s} {q:24s} {p.name}')
        if q=='placeholder': ok=False
    except Exception as e: ok=False; print(iid,'MISSING',e)
for n in ['Lungimage04','Lungimage09','Pneumonia120','BrainG100','BrainG101']:
    p=MEDICAL/f'{n}.png'; print(f'{n:14s}', 'OK figure9_extract' if p.exists() else 'MISSING'); ok &= p.exists()
print('\nCore package will run offline:', 'YES' if ok else 'YES WITH FALLBACK/PLACEHOLDER LIMITATION')
print('For strongest paper comparison, run FIRST_RUN_SETUP.bat while online before the presentation.')
