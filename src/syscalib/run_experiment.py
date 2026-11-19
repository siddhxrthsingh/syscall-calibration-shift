"""Command-line runner:  python -m syscalib.run_experiment --seeds 1000-1029 --out results/prelim.csv"""
import argparse, csv, json, time
from multiprocessing import Pool
from pathlib import Path
from .config import load_config
from .experiment import run_seed, COLUMNS, ROC_COLUMNS


def parse_seeds(s, cfg):
    if s == "dev":
        return list(cfg["seeds"]["dev"])
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _work(args):
    cfg, seed = args
    return run_seed(cfg, seed, roc=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", required=True, help="'dev', 'a-b' or comma list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seeds = parse_seeds(a.seeds, cfg)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    roc_path = out.with_name(out.stem + "_oracle_roc.csv")
    with Pool(a.jobs) as pool, open(out, "w", newline="") as f, open(roc_path, "w", newline="") as f2:
        w, w2 = csv.writer(f), csv.writer(f2)
        w.writerow(COLUMNS); w2.writerow(ROC_COLUMNS)
        for i, (rows, roc_rows) in enumerate(pool.imap(_work, [(cfg, s) for s in seeds])):
            w.writerows(rows); w2.writerows(roc_rows)
            print(f"seed {seeds[i]} done ({i+1}/{len(seeds)}), {time.time()-t0:.0f}s", flush=True)
    meta = {"seeds": seeds, "config": cfg, "elapsed_s": time.time() - t0}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
