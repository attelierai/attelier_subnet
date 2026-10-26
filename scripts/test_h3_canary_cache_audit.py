import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).with_name("h3_canary_cache_audit.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTest(unittest.TestCase):
    def test_sanitizes_cached_sampler_and_seed(self):
        history = {
            "canary-1": {
                "prompt": [
                    1,
                    "x",
                    {
                        "1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 7}},
                        "2": {"class_type": "KSampler", "inputs": {"positive": "private"}},
                    },
                ],
                "status": {
                    "status_str": "success",
                    "messages": [
                        ["execution_start", {"timestamp": 100}],
                        ["execution_cached", {"nodes": ["2"]}],
                        ["execution_success", {"timestamp": 200}],
                    ],
                },
            }
        }
        run = audit.sanitize(history, 50, 150)[0]
        self.assertEqual(run["seeds"], [7])
        self.assertTrue(run["samplerReused"])
        self.assertFalse(run["samplerExecuted"])
        self.assertNotIn("private", str(run))

    def test_cached_sampler_selector_is_not_denoising_reuse(self):
        history = {
            "canary-2": {
                "prompt": [
                    1,
                    "x",
                    {
                        "8": {"class_type": "KSamplerSelect", "inputs": {}},
                        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
                    },
                ],
                "status": {
                    "status_str": "success",
                    "messages": [
                        ["execution_start", {"timestamp": 100}],
                        ["execution_cached", {"nodes": ["8"]}],
                        ["execution_success", {"timestamp": 200}],
                    ],
                },
            }
        }
        run = audit.sanitize(history, 50, 150)[0]
        self.assertFalse(run["samplerReused"])
        self.assertTrue(run["samplerExecuted"])
        self.assertEqual(run["samplers"], [{"id": "10", "class": "SamplerCustomAdvanced"}])


if __name__ == "__main__":
    unittest.main()
