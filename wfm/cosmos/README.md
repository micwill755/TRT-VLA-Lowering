# Cosmos3 policy implementation comparison

Both Torch-TRT approaches are colocated here so their implementation size and
mechanics can be compared directly. Each export follows the same flow shape as
``export_wfm_cosmos_edge.py`` (``export_engines`` + numbered stages + CLI), but
writes the ``Cosmos3Runtime`` UND-KV-cached layout instead of the WFM bundle.

```text
Policy (Cosmos3Runtime):
  image ──► vae_encoder ──► cond_latent ─┐
  text  ──► und_prefill ──► und_k/v ─────┼──► gen (×N steps) ──► action chunk
                                         │
                                         └─ (no VAE decode)

WFM (WFMInferenceRuntime, for reference):
  text ──► packing ──► und_seq ─────────────────────────────┐
  noise ──► embed ──► gen_seq ──► mot_backbone (full MoT) ──┼──► denoise_head
       ▲              (×N steps; UND path rerun every step) │
       └──────────────────── UniPC update ──────────────────┘
  latents ──► visual_decode ──► video
```

## Approaches and line counts

### `edge/edge-llm/` — model rebuild

Builds new standalone UND and GEN PyTorch module trees, splits checkpoint
weights, maps the weights into those trees, then exports them.

| File | Lines |
|---|---:|
| `export_wfm_full_model_e2e.py` | 470 |
| `policy.py` | 465 |
| `config.py` | 285 |
| `weights.py` | 140 |
| **Total** | **1,360** |

```bash
python wfm/cosmos/edge/edge-llm/export_wfm_full_model_e2e.py \
  --engine-dir /tmp/cosmos3_policy_full --height 256 --width 256 --num-frames 5
```

### `edge/trt/` — live MoT (recommended)

Loads the Diffusers Edge transformer once and wraps its existing MoT layers.
UND runs once and emits frozen K/V; GEN reuses those K/V without reconstructing
the transformer.

| File | Lines |
|---|---:|
| `export_wfm_split_mot_backbone_e2e.py` | 705 |
| `edge_functions.py` | 114 |
| **Total** | **819** |

```bash
python wfm/cosmos/edge/trt/export_wfm_split_mot_backbone_e2e.py \
  --engine-dir /tmp/cosmos3_policy_split --height 256 --width 256 --num-frames 5
```

Live MoT is **541 lines smaller** than the model-rebuild path. Shared project
infrastructure (`trt.compile`, packing, parity, plugin loading) is excluded
from both totals.

### Official ONNX export (`tensorrt-edgellm-export --task policy`)

Hand-written UND / GEN / VAE modules plus weight remapping for the Edge-LLM
ONNX path (GitLab `tensorrt_edgellm/models/cosmos3/`):

| File | Lines |
|---|---:|
| `modeling_gen.py` | 762 |
| `modeling_vae_encoder.py` | 470 |
| `modeling_und_prefill.py` | 322 |
| `export.py` | 200 |
| `weights.py` | 142 |
| **Cosmos3-specific total** | **1,896** |

That path also depends on shared ONNX infrastructure used across Edge-LLM
tasks (not counted above): `models/ops.py` (~1.6k), `onnx/dynamo_translations.py`
(~1.2k), `onnx/onnx_custom_schemas.py` (~1.7k). C++ `Cosmos3Runtime` /
`Cosmos3PolicyRunner` (~2.2k) is shared by both Torch-TRT and ONNX engines.

| Policy export implementation | Lines | Notes |
|---|---:|---|
| Torch-TRT live MoT | **819** | wrap Diffusers MoT; no module rebuild |
| Torch-TRT model rebuild | 1,360 | split weights + re-instantiate UND/GEN |
| Official ONNX cosmos3 package | 1,896 | + shared ONNX op / schema stack |

So the live-MoT Torch-TRT export is about **2.3× smaller** than the cosmos3
ONNX package alone, before counting the shared ONNX translator stack.

## Why splitting MoT and freezing UND K/V is faster

In packed MoT attention the two streams are asymmetric:

```text
UND (causal):  Q_und × (K_und, V_und)          ← UND only talks to itself
GEN (full):    Q_gen × (K_und∥K_gen, V_und∥V_gen)  ← GEN reads UND + itself
```

UND never depends on GEN. For a fixed prompt, every layer's `K_und` / `V_und`
is identical at every diffusion step.

| Path | UND work per denoise step |
|---|---|
| WFM single MoT (`mot_backbone`) | Re-run full UND path through all layers |
| Policy (`und_prefill` + `gen`) | **None** — K/V frozen after one prefill |

That is the main architectural win vs WFM:

- **Policy:** 1 TRT enqueue per denoise step (`gen`), UND prefill once.
- **WFM:** 3 enqueues per step (`embed` + `mot_backbone` + `denoise_head`), and
  the backbone still recomputes UND QKV every time.

Measured at the same 256² smoke geometry:

| Runtime | Task | Steps | Median |
|---|---|---:|---:|
| Cosmos3Runtime policy (Torch-TRT + pad fuse) | image+text → action | 4 | **21.8 ms** |
| WFMInferenceRuntime single MoT | text → video 9×256² | 2 | **88.1 ms** |

Different products (action vs video encode/decode), but the gap matches the
factorization: policy pays for UND once; WFM pays for the joint MoT every step
and includes VAE decode (~10.2 GB peak GPU vs ~5.7 GB for policy).

## Performance

End-to-end `cosmos3_policy_inference` steady-state median, batch 1, 4 denoise
steps, 256×256, 5 frames, `und_len` 121, median of 20 iterations after 5 warmup
(unless noted):

| Bundle | Median | Action mean_abs vs baseline |
|---|---:|---:|
| Torch-TRT live MoT + pad fusion | **21.81 ms** | 0.00072 |
| Torch-TRT model rebuild + pad fusion | 21.85 ms | 0.00075 |
| ONNX → TensorRT (`tensorrt-edgellm-export`) | 22.69 ms | 0.00073 |
| Torch-TRT UND+GEN fast (before pad fusion) | 23.85 ms | 0.00068 |
| Torch-TRT GEN-fast only | 24.30 ms | — |
| Torch-TRT split original | 28.66 ms | baseline |
| WFMInferenceRuntime single MoT (2 video steps) | 88.09 ms | n/a (video) |

Per-stage engine latency after pad fusion (CUDA events, no profiler attached):

| Stage | Runs / request | ONNX | Torch-TRT |
|---|---:|---:|---:|
| VAE encoder | 1 | 5.50 ms | 5.50 ms |
| UND prefill | 1 | 3.31 ms | 2.92 ms |
| GEN denoise | 4 | 3.48 ms | 3.35 ms |

Optimization ladder that brought Torch-TRT from 28.7 ms → 21.8 ms:

1. **MoT builder settings** — UND and GEN build with `use_fp32_acc=False` and
   `decompose_attention=False` (`MOT_TRT_SETTINGS_EXPORT`), matching the ONNX
   builder's fp16 matmul accumulation and fused attention. → ~23.9 ms.
2. **VAE pad→conv fusion** — fold asymmetric causal pads into TRT
   `pre_padding` / `post_padding`. → **21.8 ms** (ahead of ONNX).

VAE keeps the conservative base settings: `decompose_attention=False` fails to
build there with a Myelin SSA assertion, and fp16 accumulation alone does not
help the VAE.

Native RoPE / Attention custom ops (`cosmos_native_ops.py`, 150 lines) were
tried and **not** kept in the export path — no latency win, slightly worse
action parity. File retained for reference.

## Pad→conv fusion (upstream Torch-TensorRT)

Wan causal 3D convolutions zero their own `padding` and apply it themselves,
asymmetric in time — `(2 * pad_t, 0)`. TensorRT's ONNX parser already folds
`Pad → Conv` into the convolution's `pre_padding` / `post_padding`. Torch-TRT
only set symmetric `padding_nd`, so those pads stayed as materialized copies
(45 extra kernels, **+1.96 ms** on the VAE: 7.46 → 5.50 ms once fixed).

The fix lives upstream on branch
[`fuse-pad-into-convolution`](https://github.com/pytorch/TensorRT/tree/fuse-pad-into-convolution):

- Post-lowering pass: rewrite `constant_pad_nd → convolution` into
  `tensorrt::conv_asym_pad`
- Converter: emit `IConvolutionLayer` with explicit `pre_padding` /
  `post_padding` (via extended `convNd`)
- Tests: `tests/py/dynamo/lowering/test_fuse_pad_into_convolution.py` (19 cases)

```bash
cd /home/micwilliams/workspace/Torch-TensorRT/TensorRT
# branch fuse-pad-into-convolution — editable install already points at this tree
```

No WFM-side import is required; both policy exports pick the pass up through
Torch-TRT's normal `post_lowering` registry.
