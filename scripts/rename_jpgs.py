#!/usr/bin/env python3
"""rename_jpgs.py -- copiaza doar fisierele .jpg dintr-un director (care poate contine
si alte tipuri de fisiere) intr-un folder nou, redenumindu-le img1.jpg, img2.jpg, ...
Fisierele originale raman neschimbate in directorul sursa.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
    if not files:
        print(f"Niciun fisier .jpg gasit in {in_dir}")
        return

    for i, src in enumerate(files, start=1):
        dst = out_dir / f"img{i}.jpg"
        if dst.exists():
            raise FileExistsError(f"{dst} exista deja -- opresc, ca sa nu suprascriu nimic")
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst}")

    print(f"\n{len(files)} fisiere .jpg copiate in {out_dir}/")


if __name__ == "__main__":
    main()
