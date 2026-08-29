# Plugin export examples

Torch-TRT shims for Edge-LLM plugins other than attention. Same four-layer stack as the VLA path in `trt/plugin/` and `vla/test_vla_pi05_e2e_one_shot.py`:

1. **Custom op** (`ops.py`) — `torch.library.custom_op` + `register_fake`
2. **Module wrapper** (`modules.py`) — keep projections in PyTorch, call the custom op
3. **Dynamo converter** (`converter.py`) — `@dynamo_tensorrt_converter` → `IPluginV3`
4. **Export script** — `torch.export` then `torch_tensorrt.dynamo.compile`

| Example | Custom op | TRT plugin name | Kernels |
|---|---|---|---|
| `ssm/export_ssm.py` | `trt_edgellm::causal_conv1d` | `causal_conv1d` | depthwise causal conv + state |
| | `trt_edgellm::update_ssm_state` | `update_ssm_state` | Mamba selective scan / SSD |
| | `trt_edgellm::gated_delta_net` | `gated_delta_net` | Qwen3.5 GDN (K=V=128) |
| `moe/export_moe.py` | `trt_edgellm::Fp16MoePlugin` | `Fp16MoePlugin` | softmax + top-k + grouped GEMM |
| `spec_decode/export_spec_decode.py` | `trt_edgellm::dflash_target_kv_cache_update` | `DFlashTargetKVCacheUpdate` | RoPE + paged KV scatter |
| | `trt::attention_plugin` (tree) | `AttentionPlugin` | XQA tree attention (VLA converter) |

Most speculative-decoding kernels (EAGLE accept, DDTree build, MTP scatter) stay in the C++ runtime, not in the engine. The two plugin-shaped pieces are DFlash KV update and tree attention.

## Run

From the Test repo root. `--no-compile` only traces the graph (no `.so` required):

```bash
python -m plugins.ssm.export_ssm --no-compile
python -m plugins.ssm.export_ssm --example gdn --no-compile
python -m plugins.moe.export_moe --no-compile
python -m plugins.spec_decode.export_spec_decode --example dflash --no-compile
python -m plugins.spec_decode.export_spec_decode --example tree_attn --no-compile
```

To lower through the real plugins:

```bash
export EDGE_LLM_PLUGIN_SO=/path/to/libNvInfer_edgellm_plugin.so
python -m plugins.ssm.export_ssm
python -m plugins.moe.export_moe
python -m plugins.spec_decode.export_spec_decode --example all
```

MoE compile needs CuTe DSL FP16 MoE artifacts in the plugin build (`ENABLE_CUTE_DSL` including `f16_moe`). GDN needs the `gdn` group. SSM SSD prefill (`seq_len >= 128`) needs `ssd`; this toy mixer uses `seq_len=8` so it hits the single-step scan loop.

## Converter checklist

Each converter in this folder does the same work as `trt/plugin/plugin_converter.py`:

1. Unpack tensor args vs scalar plugin fields
2. `create_plugin(name, fields)` via `plugins.common`
3. `as_trt_tensors` / `add_plugin_layer`
4. Return the plugin outputs in the same order as the custom op

`Fp16MoePlugin` requires `num_experts ∈ {128, 256}`, `hidden_size % 128 == 0`, `moe_inter_size % 64 == 0`. The example uses `E=128, H=128, I=64` so a real engine build is in-contract.

DFlash KV is **paged** `[2, num_pages, 128, Hkv, D]`. Tree attention in this folder uses the **linear** VLA cache `[B, 2, Hkv, cap, D]` because that is what `convert_llm_attention_plugin` already emits.
