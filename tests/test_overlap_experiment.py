import os
import unittest
import numpy as np
import tifffile
from microstitch.registration import RegistrationEngine


class TestOverlapExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = os.path.abspath("datasets/Phase_Image_Tiles/Phase_Image_Tiles")
        self.sample_tiles = [
            "img_Phase_r001_c001.tif",
            "img_Phase_r003_c005.tif",
            "img_Phase_r005_c010.tif",
            "img_Phase_r008_c005.tif",
            "img_Phase_r010_c010.tif",
        ]
        self.engine_gated = RegistrationEngine(
            min_response=0.05,
            min_spatial_score=0.50,
            use_laplacian=True,
            unwrap_aliasing=True,
        )

    def test_controlled_overlap_experiment(self):
        crop_w, crop_h = 800, 600
        start_x, start_y = 100, 100

        overlap_levels = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

        print("\n" + "=" * 120)
        print(" CONTROLLED OVERLAP REGISTRATION EXPERIMENT WITH QUALITY VALIDATION GATE")
        print("=" * 120)
        print(f"{'Overlap %':<10} | {'Known (dx,dy)':<16} | {'Est (dx,dy)':<18} | {'Error (px)':<10} | {'Resp':<10} | {'Spatial Score':<14} | {'Quality':<10} | Valid Ratio")
        print("-" * 120)

        for overlap_frac in overlap_levels:
            shift_x = int(round(crop_w * (1.0 - overlap_frac)))
            shift_y = int(round(shift_x * 0.25))

            expected_dx = -shift_x
            expected_dy = -shift_y

            errors = []
            responses = []
            spatial_scores = []
            quality_scores = []
            valid_statuses = []

            for tile_name in self.sample_tiles:
                tile_path = os.path.join(self.dataset_dir, tile_name)
                if not os.path.exists(tile_path):
                    self.skipTest(f"Missing MIST tile: {tile_path}")

                full_img = tifffile.imread(tile_path)

                ref_crop = full_img[start_y : start_y + crop_h, start_x : start_x + crop_w]
                curr_crop = full_img[
                    start_y + shift_y : start_y + shift_y + crop_h,
                    start_x + shift_x : start_x + shift_x + crop_w,
                ]

                res = self.engine_gated.register(ref_crop, curr_crop)

                err = float(np.hypot(res.dx - expected_dx, res.dy - expected_dy))
                errors.append(err)
                responses.append(res.response)
                spatial_scores.append(res.spatial_score)
                quality_scores.append(res.quality_score)
                valid_statuses.append(res.valid)

            mean_err = float(np.mean(errors))
            mean_resp = float(np.mean(responses))
            mean_spatial = float(np.mean(spatial_scores))
            mean_quality = float(np.mean(quality_scores))
            valid_count = sum(valid_statuses)
            total_count = len(valid_statuses)

            print(
                f"{int(overlap_frac*100):>3d}%       | "
                f"({expected_dx:4d}, {expected_dy:4d})      | "
                f"({res.dx:6.2f}, {res.dy:6.2f})   | "
                f"{mean_err:8.3f}   | "
                f"{mean_resp:8.4f}   | "
                f"{mean_spatial:12.6f}   | "
                f"{mean_quality:8.4f}   | "
                f"{valid_count}/{total_count} Valid"
            )

        print("=" * 120)


if __name__ == "__main__":
    unittest.main()
