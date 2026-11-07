#!/usr/bin/env bash
# Profile the kernels with Nsight Compute.
#
#   bash profile/ncu.sh                # M=8192 N=4096 bf16
#   M=1 N=4096 bash profile/ncu.sh     # decode-sized shape
#
# Writes results/ncu/<name>.ncu-rep (open in the Nsight Compute GUI, Source page
# shows SASS next to .cu lines thanks to -lineinfo) and results/ncu/metrics.csv.
#
# Needs GPU performance-counter access. If you see ERR_NVGPUCTRPERM, the driver
# has RmProfilingAdminOnly=1: run as root, or give the container
# --cap-add SYS_ADMIN, or set NVreg_RestrictProfilingToAdminUsers=0 on the host.
set -euo pipefail
M=${M:-8192}; N=${N:-4096}; DTYPE=${DTYPE:-bfloat16}
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1 || which ncu)}
OUT=results/ncu; mkdir -p "$OUT"

# The metrics that tell the memory-bound story:
#  lts__t_sectors_op_{read,write}_lookup_miss  L2-miss sectors (x32 B) = memory traffic.
#                                          GB10 (unified LPDDR5x) exposes no dram__ counters.
#  gpu__compute_memory_throughput...pct   % of peak memory-subsystem throughput
#  ..._sectors / _requests (global ld)    sectors per request; 4 sectors (128B) per
#                                          warp-request = perfectly coalesced 32-bit,
#                                          16 = coalesced 128-bit (LDG.128)
#  smsp__sass_inst_executed_op_global_ld  number of load instructions (vectorization
#                                          cuts this 8x for bf16)
#  local_*                                register spills to local memory
#  achieved occupancy, duration
METRICS=gpu__time_duration.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__block_size,launch__grid_size

run() {  # name impl pass
  local name=$1 impl=$2 pass=$3
  echo "== $name"
  "$NCU" --target-processes all --metrics "$METRICS" --csv --page raw \
      --launch-skip 4 --launch-count 8 \
      python profile/run_one.py --impl "$impl" --pass "$pass" --M "$M" --N "$N" --dtype "$DTYPE" --iters 4 \
      > "$OUT/$name.csv"
  [ -n "${SKIP_FULL:-}" ] && return
  "$NCU" --target-processes all --set full --import-source yes \
      --launch-skip 4 --launch-count 4 -f -o "$OUT/$name" \
      python profile/run_one.py --impl "$impl" --pass "$pass" --M "$M" --N "$N" --dtype "$DTYPE" --iters 4 \
      > /dev/null
}

run "cuda_naive_fwd_M${M}_N${N}" cuda_naive fwd
run "cuda_fwd_M${M}_N${N}"       cuda       fwd
run "triton_fwd_M${M}_N${N}"     triton     fwd
run "cuda_bwd_M${M}_N${N}"       cuda       bwd
run "triton_bwd_M${M}_N${N}"     triton     bwd
run "eager_fwd_M${M}_N${N}"      eager      fwd
python profile/summarize_ncu.py "$OUT" --M "$M" --N "$N"
