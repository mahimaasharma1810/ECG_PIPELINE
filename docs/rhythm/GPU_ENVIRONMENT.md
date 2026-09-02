# GPU environment

The home directory is NFS-mounted (`172.16.0.3:/archive/home2`), 83% full, and
slow. The GPU environment therefore lives on the local SSD.

```
venv     /ssd_scratch/mahimakopalley/venv-gpu     (5.0 GB)
tmp      /ssd_scratch/mahimakopalley/tmp
pip cache/ssd_scratch/mahimakopalley/pip-cache
setup    /ssd_scratch/mahimakopalley/setup_gpu.sh
```

Run GPU work with:

```bash
/ssd_scratch/mahimakopalley/venv-gpu/bin/python scripts_rhythm/s28_train_fusion.py
```

## What is installed

| | |
|---|---|
| torch | **2.5.1+cu121** |
| GPU | NVIDIA GeForce GTX 1080 Ti, 11 GB |
| Driver | 570.211.01 (CUDA 12.8) |
| Compute capability | **sm_61** (Pascal) |
| Measured throughput | **8,281 GFLOP/s** FP32 matmul |

Plus numpy 1.26.4, scipy, pandas, scikit-learn, wfdb, neurokit2, matplotlib,
pytest.

## Two things worth knowing

**sm_61 is not in the build's arch list.** `torch.cuda.get_arch_list()` returns
`['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` — the GPU's sm_61 is
absent. Kernels nevertheless run correctly (matmul and conv1d both verified, at
full expected throughput). **The mechanism was not verified** — only that it
works. If a future build stops running, this is the first thing to check.

**cu121 was chosen deliberately.** Newer CUDA wheel series have been dropping
Pascal support. If torch is ever upgraded here, re-run the kernel test rather
than trusting `torch.cuda.is_available()`, which returned `True` before any
kernel had been executed.

## A data-integrity issue this surfaced

LTAFDB record **110** has a truncated `.dat` file — 7.0 MB against an expected
44 MB (16%). The download was interrupted several times.

**Annotation-only work is unaffected**: Track A's threshold, Track B, and every
RR-based result read `.atr`, which is complete. The truncation only appears when
WAVEFORMS are read, which is why it went unnoticed until GPU work began.

Our earlier "16 complete records" check verified only that `.hea`/`.atr`/`.dat`
*exist*, not that the `.dat` is whole. **15 of 16 LTAFDB records have usable
waveforms**; `rhythm/fusion/dataset.py` skips unreadable ones.
