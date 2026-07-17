# Results: NVIDIA Jetson Orin Nano Developer Kit

Date: 2026-07-17. Runtime: Python 3.12.3, numpy 1.26.4, Linux 6.8.12-1021-tegra
(aarch64), 6 cores, L4T R39.2.0.

Power mode: **15W (nvpmodel mode 0)**, CPU governor `schedutil`, clocks **not**
pinned (`jetson_clocks` not applied), matching the default-governor conditions of
the Raspberry Pi rows. SoC thermals held at 46-48 C before and after the
benchmark, so no thermal throttling affected the measurement.

## Smoke tests (`python3 test_memory_qc_edge.py`)

All 5 tests pass.

## Latency benchmark (`python3 benchmark_latency.py`)

```
== device ==
board            : NVIDIA Jetson Orin Nano Developer Kit
platform         : Linux-6.8.12-1021-tegra-aarch64-with-glibc2.39
machine          : aarch64
processor        : aarch64
cpu count        : 6
python           : 3.12.3
numpy            : 1.26.4

Slices: 1008  |  positives: 296  |  k=15
Initial memory table: 68.4 KB (0.0668 MB)
After online update:  182.6 KB (0.1783 MB)

Operation                 Total s (mean±std)        s/slice       slices/s
fixed score               0.088230 ± 0.001100       0.00008753    11424.6
score + online update     0.088026 ± 0.000160       0.00008733    11451.2
```

Per-slice scoring latency is ~0.088 ms (~11.4k slices/s) — roughly 4x faster than
the Raspberry Pi 4 (0.354 ms) and within 1.35x of the x86-64 workstation
(0.065 ms), on a device with a 15W power envelope. The memory table footprint is
identical to every other device (68.4 KB initial, 182.6 KB after appending the
full 1008-slice stream), as expected for a deterministic numpy table.

Adding the online update is free at this scale: `score + online update`
(0.088026 s) is within run-to-run noise of `fixed score` (0.088230 s), i.e. the
append-and-rescore path costs nothing measurable relative to scoring alone.

## Power draw and sustained throughput

Unlike the Raspberry Pi rows, the Orin exposes on-board INA3221 rails, so power
was measured directly (P = V x I from `curr*_input` x `in*_input`; the
`power*_input` nodes are not populated on this board). Sampled at 20 Hz over a
5 s idle baseline and a 15 s sustained scoring loop:

```
                    idle       load     delta
VDD_IN (board)      3.321 W    5.180 W  +1.859 W
VDD_CPU_GPU_CV      0.552 W    1.345 W  +0.793 W
VDD_SOC             1.084 W    1.530 W  +0.446 W

throughput under sustained load : 11,372.9 slices/s
energy/slice @ full board power : 455.4 uJ
energy/slice @ active delta     : 163.4 uJ
```

The whole board draws 5.18 W under sustained scoring, well inside its 15 W
budget; the marginal power cost of running the QC layer is 1.86 W.

**No thermal throttling over a 60 s soak.** Throughput is flat and Tj rises 1.3 C:

```
window     slices/s    VDD_IN W   Tj C
0-10s      10996.2     5.127      48.8
10-20s     10983.2     5.188      48.9
20-30s     11015.3     5.183      49.1
30-40s     11004.1     5.175      49.5
40-50s     11014.1     5.192      49.7
50-60s     11012.4     5.203      50.1
```

Soak throughput (~11.0k slices/s) sits ~3.6% below the benchmark figure
(11.4k slices/s) because the soak loop reads two sysfs power nodes per iteration;
it is a slight underestimate, not a throttling effect. Drift across the six
windows is <0.3%, and Tj stays ~45 C below the Orin's throttle point.

## Online-adaptation experiment (`python3 experiment_online_update.py`)

First half of the Africa stream (arrival order) is treated as reviewed and
appended to the local memory; the second half is held out and scored before and
after the update.

```
Slices: 1008  |  positives: 296  |  k=15
Reviewed (update) half: 504 slices, 117 positives
Held-out half:          504 slices, 179 positives

AUROC, frozen scorer, full stream:          0.8960
AUROC, frozen scorer, held-out half:        0.9754
AUROC, after online update, held-out half:  0.9870  (delta +0.0116)
```

Online memory update improves held-out AUROC by +0.0116 with no controller or
segmenter retraining.

**Cross-device determinism:** these AUROC values are bit-identical to the
Raspberry Pi 4 and Pi 400 runs (0.8960 / 0.9754 / 0.9870, delta +0.0116). The
numpy-only scorer is therefore reproducible across aarch64 devices and across
numpy versions (1.24.2 on the Pi vs 1.26.4 here) — only latency varies by device,
not the ranking result.
