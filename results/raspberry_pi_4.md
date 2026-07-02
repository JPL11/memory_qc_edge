# Results: Raspberry Pi 4 Model B Rev 1.4

Date: 2026-07-01. Runtime: Python 3.11.2, numpy 1.24.2, Linux 6.18.37-v8+ (aarch64), 4 cores.

## Smoke tests (`python test_memory_qc_edge.py`)

All 5 tests pass.

## Latency benchmark (`python benchmark_latency.py`)

```
== device ==
board            : Raspberry Pi 4 Model B Rev 1.4
platform         : Linux-6.18.37-v8+-aarch64-with-glibc2.36
machine          : aarch64
processor        : n/a
cpu count        : 4
python           : 3.11.2
numpy            : 1.24.2

Slices: 1008  |  positives: 296  |  k=15
Initial memory table: 68.4 KB (0.0668 MB)
After online update:  182.6 KB (0.1783 MB)

Operation                 Total s (mean±std)        s/slice       slices/s    
fixed score               0.357228 ± 0.063495       0.00035439    2821.7      
score + online update     0.349480 ± 0.035158       0.00034671    2884.3      
```

Per-slice scoring latency is ~0.35 ms (~2.8k slices/s); the memory table stays
under 200 KB even after appending the full 1008-slice stream.

## Online-adaptation experiment (`python experiment_online_update.py`)

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
