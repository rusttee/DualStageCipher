from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    ROOT, OUTPUT, MEDICAL, SIPI_CATALOG, CipherConfig,
    build_black_white, config_summary, download_sipi, load_image, asset_quality,
    save_image, image_change_stats,
)


def load_hyphen_module(filename: str, module_name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


C1 = load_hyphen_module("confusion-1.py", "stage_confusion_1")
C2 = load_hyphen_module("confusion-2.py", "stage_confusion_2")
D1 = load_hyphen_module("diffusion-1.py", "stage_diffusion_1")
D2 = load_hyphen_module("diffusion-2.py", "stage_diffusion_2")
CG = load_hyphen_module("cipher-generation.py", "stage_cipher_generation")
MV = load_hyphen_module("matlab-view.py", "matlab_view")
AN = load_hyphen_module("analysis.py", "analysis_view")


def choose_config() -> CipherConfig:
    print("\nNUMERICAL / TRANSIENT MODE")
    print("1. Legacy reproduction mode — NO first-100 discard [default]")
    print("2. Paper mode — generate extra 100 and discard first 100")
    choice=input("Choose 1/2: ").strip() or "1"
    discard=(choice=="2")

    print("\nCOLOR IMAGE SIZE")
    print("1. Keep official USC-SIPI source size (color images are 512x512) [default]")
    print("2. Resize color input to 256x256 for paper-table-sized experiments")
    cp=input("Choose 1/2: ").strip() or "1"
    color_preprocess="paper256" if cp=="2" else "official"

    return CipherConfig(discard_first_100=discard,color_preprocess=color_preprocess)


def choose_image(config: CipherConfig):
    print("\nIMAGE TO ENCRYPT — ALL NORMAL OPTIONS WORK OFFLINE")
    ids=list(SIPI_CATALOG.keys())
    for i,image_id in enumerate(ids,1):
        s=SIPI_CATALOG[image_id]
        try:
            p=download_sipi(image_id); q=asset_quality(image_id,p)
        except Exception:
            q="missing"
        print(f"{i:2d}. {image_id} — {s['name']} — {q}")
    base=len(ids)
    medical=["Lungimage04","Lungimage09","Pneumonia120","BrainG100","BrainG101"]
    for j,name in enumerate(medical,base+1):
        print(f"{j:2d}. {name} — bundled from paper Figure 9 (presentation fallback)")
    b2=base+len(medical)
    print(f"{b2+1:2d}. Black 256x256 chosen-plaintext image")
    print(f"{b2+2:2d}. White 256x256 chosen-plaintext image")
    print(f"{b2+3:2d}. Input a NEW image path")

    while True:
        raw=input("Selection: ").strip()
        try: sel=int(raw)
        except ValueError:
            print("Enter a menu number."); continue
        if 1 <= sel <= len(ids):
            image_id=ids[sel-1]
            try: path=download_sipi(image_id)
            except Exception as exc: print(exc); continue
            q=asset_quality(image_id,path)
            if q != 'exact_cached_sipi':
                print(f"NOTE: using {q}. For strongest paper verification run FIRST_RUN_SETUP.bat while online.")
            arr=load_image(path,config)
            return arr,{"kind":"USC-SIPI/offline-cache","image_id":image_id,"name":SIPI_CATALOG[image_id]["name"],"path":str(path),"asset_quality":q}
        if base+1 <= sel <= base+len(medical):
            name=medical[sel-(base+1)]
            path=MEDICAL/f"{name}.png"
            if not path.exists(): print("Bundled medical image missing:",path); continue
            arr=load_image(path,config)
            print("NOTE: this is extracted from the plain-image panel in paper Figure 9. The paper names the source databases but does not publish the exact source file/slice IDs.")
            return arr,{"kind":"paper Figure 9 extract","name":name,"path":str(path),"asset_quality":"figure9_extract"}
        if sel==b2+1:
            arr,name=build_black_white("black"); return arr,{"kind":"generated paper chosen-plaintext","name":name}
        if sel==b2+2:
            arr,name=build_black_white("white"); return arr,{"kind":"generated paper chosen-plaintext","name":name}
        if sel==b2+3:
            p=Path(input("Enter full image path: ").strip().strip('"')).expanduser()
            if not p.exists(): print("File not found:",p); continue
            arr=load_image(p,config); return arr,{"kind":"custom","name":p.stem,"path":str(p)}
        print("Unknown choice.")

def stage_line(name,before,after):
    s=image_change_stats(before,after)
    print(f"\n{name}")
    if s.get("shape_changed"):
        print("  shape:",s["before_shape"],"->",s["after_shape"])
    else:
        print(f"  changed pixels: {s['changed_pixels']}/{s['total_pixels']} = {s['changed_pixels_percent']:.6f}%")
        print(f"  changed byte-components: {s['changed_components']}/{s['total_components']} = {s['changed_components_percent']:.6f}%")


def main():
    print("="*78)
    print("CHAOTIC COSINE-LOGISTIC IMAGE ENCRYPTION — MODULAR REPRODUCTION PIPELINE")
    print("="*78)
    config=choose_config()
    print("\nRUN CONFIGURATION\n"+config_summary(config))
    image,source=choose_image(config)

    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    label=str(source.get("image_id") or source.get("name") or "image").replace("/","_").replace(" ","_")
    out=OUTPUT/f"{stamp}_{label}"
    out.mkdir(parents=True,exist_ok=True)
    (out/"run_config.json").write_text(json.dumps({"config":config.to_dict(),"source":source},indent=2),encoding="utf-8")

    stages={"original":image}
    save_image(image,out/"01_original.png")
    print(f"\nOriginal: shape={image.shape}, dtype={image.dtype}")

    i1,t1=C1.run(image,config); stages["confusion1"]=i1; save_image(i1,out/"02_confusion_1.png"); stage_line("CONFUSION 1: I0 -> I1",image,i1)
    i2,t2=C2.run(i1,t1["seq1"],config); stages["confusion2"]=i2; save_image(i2,out/"03_confusion_2.png"); stage_line("CONFUSION 2: I1 -> I2",i1,i2)
    i3,t3=D1.run(i2,config); stages["diffusion1"]=i3; save_image(i3,out/"04_diffusion_1.png"); stage_line("DIFFUSION 1: I2 -> I3",i2,i3)
    i4,t4=D2.run(i3); stages["diffusion2"]=i4; save_image(i4,out/"05_diffusion_2.png"); stage_line("DIFFUSION 2: I3 -> I4",i3,i4)
    cipher,meta=CG.run(i4,out); stages["cipher"]=cipher

    # Save mathematically useful stage internals.
    np.save(out/"R_diffusion_matrix.npy",t3["R"])
    save_image(t3["R"],out/"R_diffusion_matrix.png")
    np.save(out/"IMSB.npy",t4["IMSB"]); np.save(out/"ILSB.npy",t4["ILSB"])
    np.savetxt(out/"confusion2_actions.txt",np.asarray(t2["actions"],dtype=str),fmt="%s")

    trace={**t1,**t2,**t3,**t4}
    analysis=AN.run(stages,out/"analysis",source)
    show=(input("\nOpen the MATLAB-style stage figure now? [y/N]: ").strip().lower()=="y")
    MV.run(stages,trace,out/"matlab_view",show=show)

    print("\n"+"="*78)
    print("ENCRYPTION COMPLETE")
    print("="*78)
    print("Cipher image     :",out/"06_cipher_image.png")
    print("Cipher hex text  :",out/"cipher_hex.txt")
    print("Cipher raw bytes :",out/"cipher_bytes.bin")
    print("Analysis         :",out/"analysis"/"analysis_report.txt")
    print("MATLAB views     :",out/"matlab_view")
    print("Full run folder  :",out)
    print("\nNOTE: the paper's output C is an IMAGE ciphertext matrix. cipher_hex.txt is the textual hex representation of those bytes.")


if __name__=="__main__":
    main()
