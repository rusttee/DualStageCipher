import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; FULL=ROOT/'simulation_results'/'10_full_paper_comparison.py'
print('='*78); print('SIMULATION RESULTS & DISCUSSION'); print('='*78)
print('1. Run ALL comparisons (USC-SIPI + medical if supplied + chosen plaintext + runtime)')
print('2. Run all except runtime')
print('3. Quick verification: grayscale + chosen plaintext only')
print('4. Exit')
c=input('Choose: ').strip(); cmd=[sys.executable,str(FULL)]
if c=='2': cmd+=['--skip-runtime']
elif c=='3': cmd+=['--skip-runtime','--skip-color','--skip-medical']
elif c=='4': raise SystemExit(0)
elif c!='1': raise SystemExit('Unknown choice')
raise SystemExit(subprocess.call(cmd,cwd=ROOT))
