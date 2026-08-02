import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("v4_flash_0731_prefix_oracle.py")
SPEC = importlib.util.spec_from_file_location("flash_prefix_oracle", SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class FlashPrefixOracleTest(unittest.TestCase):
    def test_prefix_manifest_and_header_parser(self):
        manifest = ORACLE.static_manifest()
        self.assertFalse(manifest["full43_oracle"])
        self.assertEqual([item["compression_ratio"] for item in manifest["layers"]], [0, 0, 4, 128])
        self.assertEqual([item["router"] for item in manifest["layers"]], ["hash", "hash", "hash", "learned"])
        self.assertEqual(manifest["shard_file_stat_total_bytes"], 166_886_535_336)
        self.assertEqual(manifest["index_tensor_payload_total_bytes"], 166_878_536_440)

        header = json.dumps({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00\x01\x00")
            tensors, payload_bytes, header_bytes = ORACLE.read_safetensors_header(path)
        self.assertEqual(tensors["x"]["shape"], [2])
        self.assertEqual((payload_bytes, header_bytes), (4, len(header)))

    def test_full43_plan_is_explicit_and_prefix_default_is_unchanged(self):
        self.assertEqual(ORACLE.execution_plan(), ORACLE.PREFIX)
        plan = ORACLE.execution_plan(42)
        self.assertEqual((len(plan), plan[-1]["layer"], plan[-1]["shard"]), (43, 42, 44))
        self.assertEqual([item["router"] for item in plan[:4]], ["hash", "hash", "hash", "learned"])
        manifest = ORACLE.run_manifest(42, True)
        self.assertEqual(manifest["schema"], ORACLE.FULL43_SCHEMA)
        self.assertTrue(manifest["full43_oracle"])
        self.assertTrue(manifest["final_head"])
        with self.assertRaisesRegex(ValueError, "requires --end-layer 42"):
            ORACLE.run_manifest(3, True)

    def test_cutpoint_manifest_uses_raw_little_endian_logical_arrays(self):
        data = b"\x00\x00\x80\x3f"
        with tempfile.TemporaryDirectory() as directory:
            cutpoint_dir = Path(directory) / "cutpoints"
            writer = ORACLE.CutpointWriter(cutpoint_dir, 17)
            writer.write_raw("one_f32", data, "F32", [1], "unit-test scalar")
            manifest = writer.finish(["one_f32"])
            on_disk = json.loads((cutpoint_dir / "manifest.json").read_text())
            self.assertEqual((cutpoint_dir / "one_f32.bin").read_bytes(), data)

        self.assertEqual(manifest["schema"], ORACLE.CUTPOINT_SCHEMA)
        self.assertEqual(on_disk["tensor_order"], ["one_f32"])
        self.assertEqual(on_disk["tensors"]["one_f32"]["shape"], [1])
        self.assertEqual(on_disk["tensors"]["one_f32"]["dtype"], "F32")
        self.assertEqual(on_disk["tensors"]["one_f32"]["byte_order"], "little")
        self.assertEqual(on_disk["tensors"]["one_f32"]["sha256"], hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
