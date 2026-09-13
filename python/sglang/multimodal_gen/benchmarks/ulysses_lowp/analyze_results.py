"""Build a Markdown report from completed captures without retaining exports."""

import argparse
import json
from html import escape
from pathlib import Path

import attention_module
import pre_attention
import timeline
from quality import decoded_hash, ssim
from run import PROMPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--up", type=int, choices=[4, 8], default=8)
    parser.add_argument("--baseline-videos", type=Path)
    args = parser.parse_args()
    root, work = args.output.resolve(), args.scratch.resolve()
    analysis = work / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    timeline.SCRATCH = pre_attention.SCRATCH = attention_module.SCRATCH = analysis
    names = {"bf16": "BF16", "sage2": "Sage2 CUDA", "lowp": "Lowp-Sage2"}
    records = [
        r
        for p in work.glob(f"up{args.up}_*/done.json")
        for r in json.loads(p.read_text())
    ]
    assert len(records) == 12, "Complete all 9 videos and 3 traces first"
    lines = [
        f"# UP{args.up} Video quality and performance",
        "",
        "50 timesteps / 49 updates; 3 timesteps / 2 updates. Seed=2101, FSDP OFF, AdaLN online ON, compile OFF. Each backend is warmed up and measured once; no repeated-run confidence interval.",
        "",
        "SSIM measures similarity, not perceptual quality. Interpret cross-architecture comparisons separately from same-architecture code regression.",
    ]
    for case, prompt in enumerate(PROMPTS, 1):
        paths = {
            b: root / "videos" / f"case{case}_up{args.up}_{b}_50steps.mp4"
            for b in names
        }
        hashes = {b: decoded_hash(p) for b, p in paths.items()}
        lines += [
            "",
            f"## Case {case}",
            "",
            prompt,
            "",
            "| Backend / video | Pipeline / denoise s | SSIM vs local BF16 | Decoded SHA-256 |",
            "|---|---:|---:|---|",
        ]
        for b, path in paths.items():
            r = next(
                r
                for r in records
                if not r["trace"] and r["case"] == case and r["backend"] == b
            )
            assert r["steps"] == 50
            similarity = 1.0 if b == "bf16" else ssim(path, paths["bf16"])
            lines.append(
                f"| [{names[b]}](videos/{path.name}) | {r['pipeline']:.2f} / {r['denoise']:.4f} | {similarity:.6f} | `{hashes[b]}` |"
            )
        if args.baseline_videos:
            baseline = args.baseline_videos / paths["lowp"].name
            lines += [
                "",
                f"Lowp vs specified reference `{baseline}`: SSIM={ssim(paths['lowp'], baseline):.6f}; decoded frames equal={hashes['lowp'] == decoded_hash(baseline)}. Bitwise equality is not required across SM90/SM120.",
            ]
        lines += [
            "",
            "Manual observations: pending. Inspect frames at 0.5 / 2 / 4 seconds and play the full videos.",
        ]
    lines += [
        "",
        "## 3-step CUDA/NVTX",
        "",
        "GPU0: 2 actual updates / 100 DiT attention calls. A2A is the NVTX GPU projection including pack and gaps, not pure network time.",
        "",
        "| Trace | Pipeline / denoise s | GPU0 loop ms | Input A2A ms |",
        "|---|---:|---:|---:|",
    ]
    for b in names:
        path = root / "timelines" / f"case1_up{args.up}_{b}_3steps.nsys-rep"
        t = timeline.timeline(path, expected_updates=2)
        r = next(r for r in records if r["trace"] and r["backend"] == b)
        assert r["steps"] == 3
        if b == "sage2":
            assert t["stages"]["timeline::stock_qk_quant"]["count"] == 100
            assert "timeline::stock_qk_quant_per_thread" not in t["stages"]
        stage = "lowp_input_a2a" if b == "lowp" else "timeline::bf16_input_a2a"
        lines.append(
            f"| [{names[b]}](timelines/{path.name}) | {r['pipeline']:.2f} / {r['denoise']:.4f} | {t['loop_ms']:.3f} | {t['stages'][stage]['gpu_ms']:.3f} |"
        )
    lines += [
        "",
        "## Complete attention preprocessing",
        "",
        "From the first local_stats (Lowp) or input pack (Sage2) GPU operation to the Sage kernel start. Excludes QKV projection, attention and output A2A.",
        "",
        "| Backend | GPU0 100 windows ms | Mean per call us | Range of totals across 8 GPUs ms |",
        "|---|---:|---:|---:|",
    ]
    modules = {}
    for b in ["sage2", "lowp"]:
        t = pre_attention.analyze(args.up, b, steps=3, expected_calls=100)
        gpu0 = next(d for d in t["devices"] if d["device"] == 0)
        spans = [d["gpu_span_ms"] for d in t["devices"]]
        lines.append(
            f"| {names[b]} | {gpu0['gpu_span_ms']:.3f} | {gpu0['gpu_span_ms'] * 10:.3f} | {min(spans):.3f}–{max(spans):.3f} |"
        )
        modules[b] = attention_module.inspect(args.up, b, steps=3)
        assert (
            any("QuantInt8Kernel" in k["name"] for k in modules[b]["kernels"])
            if b == "sage2"
            else True
        )
    lines += [
        "",
        "## One attention module by function",
        "",
        "GPU0 / denoising_step_1 / transformer.blocks.25.attn. Each kernel belongs to one row; zero means no independent kernel. Shared statistics are separate. Kernel sums differ from windows containing gaps and overlap.",
        "",
        "| Function | Sage2 CUDA μs | Lowp μs |",
        "|---|---:|---:|",
    ]
    values = {b: attention_module.functional_metrics(m) for b, m in modules.items()}
    for i, label in enumerate(attention_module.FUNCTIONAL_LABELS):
        lines.append(
            f"| {label} | {values['sage2'][i]:.3f} | {values['lowp'][i]:.3f} |"
        )
    lines += [
        f"| Full preprocessing window | {modules['sage2']['pre_span_us']:.3f} | {modules['lowp']['pre_span_us']:.3f} |"
    ]
    for b, module in modules.items():
        lines += [
            "",
            f"### {names[b]} Raw kernels",
            "",
            "| NVTX | Kernel | μs |",
            "|---|---|---:|",
        ]
        for k in module["kernels"]:
            if k["phase"] == "Preprocessing":
                lines.append(
                    f"| `{k['stage']}` | <code>{escape(k['name'])}</code> | {k['us']:.3f} |"
                )
    lines += [
        "",
        "## Environment",
        "",
        "```json",
        (work / "environment.json").read_text(),
        "```",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
