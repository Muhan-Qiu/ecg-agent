# -*- coding: utf-8 -*-
"""下载 MIT-BIH Arrhythmia Database 到 data/。

数据许可：Open Data Commons Attribution License v1.0 (ODC-BY 1.0)
PhysioNet: https://physionet.org/content/mitdb/1.0.0/

用法（仓库根目录）：
    python scripts/download_mitbih.py
    python scripts/download_mitbih.py --dir data
"""
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data", help="下载目录（默认 data/）")
    a = ap.parse_args()
    Path(a.dir).mkdir(parents=True, exist_ok=True)
    import wfdb
    print(f"downloading MIT-BIH to {a.dir} ...")
    wfdb.dl_database("mitdb", dl_dir=a.dir)
    print("done. 数据许可见 https://physionet.org/content/mitdb/1.0.0/")


if __name__ == "__main__":
    main()
