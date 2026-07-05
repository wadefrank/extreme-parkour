import tempfile
import unittest
from pathlib import Path

import numpy as np

from deploy.s100.scripts.calibration_data import CalibrationRecorder, TENSOR_SPECS


class CalibrationRecorderTest(unittest.TestCase):
    def test_writes_aligned_samples(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            recorder = CalibrationRecorder(
                output_dir,
                max_samples=3,
                warmup_updates=1,
            )
            values = {
                "depth_image": np.zeros((2, 58, 87), dtype=np.float32),
                "proprio": np.zeros((2, 53), dtype=np.float32),
                "h_in": np.zeros((2, 1, 512), dtype=np.float32),
                "actor_obs": np.zeros((2, 753), dtype=np.float32),
                "depth_latent": np.zeros((2, 32), dtype=np.float32),
            }
            self.assertEqual(recorder.record(**values), 0)
            self.assertEqual(recorder.record(**values), 2)
            self.assertEqual(recorder.record(**values), 3)
            self.assertTrue(recorder.full)

            for relative_dir, expected_shape in TENSOR_SPECS.values():
                paths = sorted((output_dir / relative_dir).glob("*.npy"))
                self.assertEqual(len(paths), 3)
                self.assertEqual(np.load(paths[0]).shape, expected_shape)
            self.assertTrue((output_dir / "manifest.json").is_file())

    def test_refuses_to_mix_existing_samples(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            recorder = CalibrationRecorder(output_dir, max_samples=1)
            recorder.close()
            sample_dir = output_dir / TENSOR_SPECS["depth_image"][0]
            np.save(sample_dir / "000000.npy", np.zeros((58, 87), dtype=np.float32))
            with self.assertRaises(FileExistsError):
                CalibrationRecorder(output_dir)


if __name__ == "__main__":
    unittest.main()
