import os
import unittest
import numpy as np
import tifffile
from microstitch.registration import RegistrationEngine


class TestMISTRegistrationDiagnostic(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = os.path.abspath("datasets/Phase_Image_Tiles/Phase_Image_Tiles")
        self.engine = RegistrationEngine(min_response=-1.0)

    def _load_tile(self, row: int, col: int) -> np.ndarray:
        filename = f"img_Phase_r{row:03d}_c{col:03d}.tif"
        filepath = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(filepath):
            self.skipTest(f"MIST dataset file missing: {filepath}")
        return tifffile.imread(filepath)

    def test_mist_neighboring_tiles_diagnostic(self):
        pairs = [
            ((1, 1), (1, 2)),
            ((1, 2), (1, 3)),
            ((1, 3), (1, 4)),
        ]

        print("\n=== MIST NEIGHBORING TILES REGISTRATION DIAGNOSTIC ===")
        for (r1, c1), (r2, c2) in pairs:
            tile1 = self._load_tile(r1, c1)
            tile2 = self._load_tile(r2, c2)

            res = self.engine.register(tile1, tile2)
            print(
                f"Pair r{r1:03d}_c{c1:03d} -> r{r2:03d}_c{c2:03d} | "
                f"dx: {res.dx:8.2f} px | dy: {res.dy:8.2f} px | "
                f"response: {res.response:10.6f} | valid: {res.valid}"
            )
            # Ensure output result is non-null and finite
            self.assertTrue(np.isfinite(res.dx))
            self.assertTrue(np.isfinite(res.dy))
            self.assertTrue(np.isfinite(res.response))


if __name__ == "__main__":
    unittest.main()
