"""Turn the raw ncu CSVs from profile/ncu.sh into one markdown table."""
import argparse
import glob
import io
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--M", type=int)
ap.add_argument("--N", type=int)
a = ap.parse_args()

COLS = {
    "gpu__time_duration.sum": "time",
    "lts__t_sectors_op_read_lookup_miss.sum": "l2miss_rd_sect",
    "lts__t_sectors_op_write_lookup_miss.sum": "l2miss_wr_sect",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem_%peak",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "ld_sectors",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "ld_requests",
    "smsp__sass_inst_executed_op_global_ld.sum": "ld_instr",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum": "local_ld_B",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum": "local_st_B",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_%",
    "launch__registers_per_thread": "regs",
    "launch__block_size": "block",
    "launch__grid_size": "grid",
}


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return float("nan")


def to_bytes(v, unit):
    mult = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    return num(v) * mult.get(str(unit).strip(), 1)


def to_us(v, unit):
    mult = {"nsecond": 1e-3, "usecond": 1, "msecond": 1e3, "ns": 1e-3, "us": 1, "ms": 1e3}
    return num(v) * mult.get(str(unit).strip(), 1)


rows = []
for path in sorted(glob.glob(os.path.join(a.dir, "*.csv"))):
    if path.endswith("metrics.csv"):
        continue
    text = open(path).read()
    start = text.find('"ID"')
    if start < 0:
        print(f"skip {path}: no ncu data (permission error?)")
        continue
    df = pd.read_csv(io.StringIO(text[start:]))
    units, df = df.iloc[0], df.iloc[1:]
    name = os.path.basename(path)[:-4]
    for kname, g in df.groupby("Kernel Name", sort=False):
        if "distribution" in kname:  # torch.randn for dy/ds in run_one.py, not part of the op
            continue
        r = {"run": name, "kernel": (kname.replace("(anonymous namespace)::", "").replace("<unnamed>::", "").replace("void ", "")
                                   .replace("at::native::", "").replace("at::", "")
                                   .split("(")[0].split("<")[0])[:34], "calls": len(g) // 1}
        for c, short in COLS.items():
            if c not in g:
                continue
            vals = g[c]
            if short in ("local_ld_B", "local_st_B"):
                r[short] = sum(to_bytes(v, units[c]) for v in vals) / len(vals)
            elif short == "time":
                r[short] = sum(to_us(v, units[c]) for v in vals) / len(vals)
            else:
                r[short] = sum(num(v) for v in vals) / len(vals)
        rows.append(r)

out = pd.DataFrame(rows)
if out.empty:
    raise SystemExit("no profiles parsed")
out["sect/req"] = out.ld_sectors / out.ld_requests
# Memory traffic ~ L2-miss sectors x 32 B (each miss is a memory transaction).
out["MB_moved"] = (out.l2miss_rd_sect + out.l2miss_wr_sect) * 32 / 1e6
out["GB/s"] = out.MB_moved * 1e6 / (out.time * 1e3)
show = ["run", "kernel", "time", "MB_moved", "GB/s", "mem_%peak", "sect/req", "ld_instr",
        "local_ld_B", "regs", "block", "grid", "occupancy_%"]
out = out[[c for c in show if c in out]]
md = out.to_markdown(index=False, floatfmt=".1f")
title = f"# Nsight Compute summary (M={a.M}, N={a.N})\n\ntime in µs, per kernel launch (averaged)\n\n"
open(os.path.join(a.dir, "summary.md"), "w").write(title + md + "\n")
print(md)
