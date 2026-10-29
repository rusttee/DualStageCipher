from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
from common import ROOT, SIPI_CATALOG, PAPER_IMAGES, download_sipi_exact, download_sipi, asset_quality

READY=ROOT/'offline_ready.json'

def cache_wheels():
    wheels=ROOT/'offline_wheels'; wheels.mkdir(exist_ok=True)
    cmd=[sys.executable,'-m','pip','download','-r',str(ROOT/'requirements.txt'),'-d',str(wheels)]
    print('\nCaching Python wheels for this computer/Python version:')
    print(' '.join(cmd)); return subprocess.call(cmd)==0

def main():
    ap=argparse.ArgumentParser(description='Prepare the project for an offline presentation.')
    ap.add_argument('--cache-wheels',action='store_true')
    ap.add_argument('--skip-downloads',action='store_true')
    a=ap.parse_args()
    results={}
    print('='*78); print('FIRST-RUN OFFLINE PREPARATION'); print('='*78)
    if not a.skip_downloads:
        for iid,spec in SIPI_CATALOG.items():
            print(f'\n{iid} {spec["name"]}: caching exact USC-SIPI source ...')
            try:
                p=download_sipi_exact(iid); results[iid]={'status':'exact_cached','path':str(p.relative_to(ROOT))}; print('  OK:',p)
            except Exception as e:
                p=download_sipi(iid); results[iid]={'status':'fallback','path':str(p.relative_to(ROOT)),'reason':str(e)}
                print('  Could not cache exact source; bundled fallback remains usable.')
                print(' ',e)
    wheel_ok=None
    if a.cache_wheels: wheel_ok=cache_wheels()
    medical={
      'status':'bundled_figure9_extracts',
      'note':'The paper gives source databases but not exact file/slice identifiers. Figure 9 plain-image panels are bundled for offline presentation; replace with author-original 256x256 files for exact numerical verification.'
    }
    data={'prepared_at':time.strftime('%Y-%m-%d %H:%M:%S'),'standard_images':results,'medical':medical,'wheels_cached':wheel_ok}
    READY.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('\n'+('='*78)); print('OFFLINE PREPARATION COMPLETE'); print('='*78)
    print('Normal encryption and simulation runs now use cached local assets only.')
    print('Ready record:',READY)
    print('\nMedical source note: exact author-selected file/slice IDs are not published; bundled Figure 9 extracts are presentation fallbacks.')
if __name__=='__main__': main()
