"""Check routing on SM120-only installs without requiring that GPU here."""

import os
import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class InstrumentationTest(unittest.TestCase):
    def test_relative_model_path_survives_subprocess_cwd(self):
        script = Path(__file__).resolve().parent / "run.py"
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for backend in ["bf16", "sage2", "lowp"]:
                for mode in ["video", "trace"]:
                    work = root / "work" / f"up8_{backend}_{mode}"
                    work.mkdir(parents=True)
                    (work / "done.json").write_text("[]")

            def probe(command, **kwargs):
                self.assertEqual(command[3], str(root / "models/MiniMax-H3"))
                self.assertEqual(kwargs["cwd"], script.parent)
                Path(command[-1]).write_text("{}")

            previous = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch.object(
                        sys,
                        "argv",
                        [str(script), "--output", "./results", "--scratch", "./work"],
                    ),
                    patch("subprocess.run", side_effect=probe) as child,
                ):
                    result = runpy.run_path(str(script), run_name="__main__")
                child.assert_called_once()
                self.assertEqual(result["ROOT"], root / "results")
                self.assertEqual(result["SCRATCH"], root / "work")
            finally:
                os.chdir(previous)

    def test_architecture_specific_extension(self):
        for arch in ["sm89", "sm90"]:
            with self.subTest(extension=arch):
                ranges = []
                nvtx = SimpleNamespace(range_push=ranges.append, range_pop=lambda: None)
                torch = SimpleNamespace(
                    Tensor=type("Tensor", (), {}), cuda=SimpleNamespace(nvtx=nvtx)
                )
                core = ModuleType("sageattention.core")
                for name in [
                    "sageattn_qk_int8_pv_fp8_cuda",
                    "sageattn_qk_int8_pv_fp8_cuda_sm90",
                ]:
                    setattr(core, name, lambda *args, **kwargs: kwargs["qk_quant_gran"])
                for name in [
                    "per_warp_int8_cuda",
                    "per_thread_int8_triton",
                    "per_channel_fp8",
                ]:
                    setattr(core, name, lambda *args: None)
                entry = (
                    "qk_int8_sv_f8_accum_"
                    + ("f16" if arch == "sm89" else "f32")
                    + "_fuse_v_scale_attn_inst_buf"
                )
                extension = SimpleNamespace(**{entry: lambda: "called"})
                setattr(
                    core,
                    arch + "_compile",
                    SimpleNamespace(**{"_qattn_" + arch: extension}),
                )
                setattr(core, arch.upper() + "_ENABLED", True)
                sage = ModuleType("sageattention")
                sage.core = core
                layers = ModuleType("sglang.multimodal_gen.runtime.layers")
                layers.usp = SimpleNamespace(
                    _usp_input_all_to_all_packed_qkv=lambda: None,
                    _usp_output_all_to_all=lambda: None,
                )
                flash = ModuleType(
                    "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn"
                )
                flash.FlashAttentionImpl = type(
                    "FlashAttentionImpl", (), {"forward": lambda: None}
                )
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "torch": torch,
                            "sageattention": sage,
                            "sageattention.core": core,
                            "sglang.multimodal_gen.runtime.layers": layers,
                            flash.__name__: flash,
                        },
                    ),
                    patch.dict(
                        os.environ, {"H3_TIMELINE_NVTX": "1", "H3_SAGE2_QK_CUDA": "1"}
                    ),
                ):
                    runpy.run_path(str(Path(__file__).parent / "nvtx/sitecustomize.py"))
                    for name in [
                        "sageattn_qk_int8_pv_fp8_cuda",
                        "sageattn_qk_int8_pv_fp8_cuda_sm90",
                    ]:
                        self.assertEqual(
                            getattr(core, name)(
                                None, None, None, qk_quant_gran="per_thread"
                            ),
                            "per_warp",
                        )
                    self.assertEqual(getattr(extension, entry)(), "called")
                    self.assertTrue(
                        any("timeline::sage2_kernel" in label for label in ranges)
                    )


if __name__ == "__main__":
    unittest.main()
