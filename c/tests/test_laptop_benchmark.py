import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.laptop_benchmark import smi_sample


class LaptopBenchmarkTest(unittest.TestCase):
    def test_smi_sample(self):
        sample = smi_sample(
            "2026/07/23 12:00:00.000, 0, RTX, P0, 10.01, 67, 405, 405, "
            "1045, 0, 0, Not Active, Not Active, Not Active"
        )
        self.assertEqual(sample["power_w"], 10.01)
        self.assertEqual(sample["vram_used_mib"], 1045)

    def test_records_a_command_without_a_model_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text('{"fixture": true}\n')
            out = root / "run"
            tool = Path(__file__).resolve().parent.parent / "tools" / "laptop_benchmark.py"
            run = subprocess.run([
                sys.executable, str(tool), "--out", str(out), "--implementation", "c-cuda",
                "--model", "local-fixture", "--model-manifest", str(manifest), "--quant", "int4",
                "--prompt-ids", "fixture-prompt", "--batch", "1", "--context", "4",
                "--kv-layout", "mla", "--active-experts", "6", "--total-experts", "64",
                "--cache-state", "warm", "--phase", "decode", "--power-state", "ac-performance",
                "--no-telemetry", "--", sys.executable, "-c", "print('raw sample')"],
                text=True, capture_output=True, check=True)
            self.assertIn("raw sample", run.stdout)
            record = json.loads((out / "run.json").read_text())
            self.assertEqual(record["returncode"], 0)
            self.assertEqual(record["workload"]["phase"], "decode")
            self.assertEqual(record["workload"]["kv_layout"], "mla")
            self.assertEqual(record["workload"]["active_experts"], 6)
            self.assertIn("raw sample", (out / "samples.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
