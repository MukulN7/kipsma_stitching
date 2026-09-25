from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class RegistrationResult:
    """Dataclass holding the result of a 2D registration operation.

    Attributes:
        dx (float): Horizontal displacement in pixels from ref_frame to curr_frame.
        dy (float): Vertical displacement in pixels from ref_frame to curr_frame.
        response (float): Phase correlation peak response strength from cv2.phaseCorrelate.
        valid (bool): True if registration passed all quality and threshold gates.
        spatial_score (float): Normalized spatial cross-correlation score of the overlap region.
        quality_score (float): Unified quality metric in range [0.0, 1.0].
    """

    dx: float
    dy: float
    response: float
    valid: bool
    spatial_score: float = 0.0
    quality_score: float = 0.0


# Overlap fractions tried by the horizontal fallback (right and left hypotheses).
# Each fraction f means: strip width = int(round(f * frame_width)).
_HORIZONTAL_OVERLAP_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


class RegistrationEngine:
    """Primary 2D Translation Registration Engine based on Phase Correlation with Quality Validation Gate.

    Applies 2D Hann windowing and Laplacian edge-whitening preprocessing before
    computing sub-pixel translation shifts (dx, dy) and correlation response strength
    via cv2.phaseCorrelate. Automatically resolves FFT periodic shift aliasing and evaluates
    registration quality across spatial overlap boundaries.

    When full-frame phase correlation fails the quality gate, a horizontal-strip
    overlap fallback is attempted automatically. It tests a small set of plausible
    overlap fractions for rightward and leftward hypotheses, picks the best-scoring
    match, and converts the local strip shift back into global dx/dy coordinates.

    Coordinate Convention:
        The returned (dx, dy) represents the displacement vector of features from ref_frame to curr_frame:
        - Positive dx: a feature in curr_frame is located to the RIGHT (+x) relative to its position in ref_frame.
        - Positive dy: a feature in curr_frame is located DOWN (+y) relative to its position in ref_frame.
        - To align/warp curr_frame into ref_frame coordinates, apply spatial shift (-dx, -dy).
    """

    def __init__(
        self,
        min_response: float = 0.05,
        min_spatial_score: float = 0.50,
        max_shift: Optional[float] = None,
        use_laplacian: bool = True,
        unwrap_aliasing: bool = True,
        min_fallback_response: Optional[float] = None,
        use_sift: bool = False,
    ):
        """
        Args:
            min_response (float): Minimum correlation response required for valid=True. Default: 0.05.
            min_spatial_score (float): Minimum spatial cross-correlation required for valid=True. Default: 0.50.
            max_shift (Optional[float]): Maximum allowed shift distance in pixels. Default: None (unlimited).
            use_laplacian (bool): Whether to apply Laplacian edge whitening during preprocessing.
            unwrap_aliasing (bool): Whether to unwrap FFT periodic shift ambiguity (> W/2 or H/2).
            min_fallback_response (Optional[float]): Minimum phase correlation response required for strip fallback. Defaults to min_response.
            use_sift (bool): Whether to use SIFT+RANSAC translation as primary registration before phase correlation fallback. Default: True.
        """
        self.min_response = min_response
        self.min_spatial_score = min_spatial_score
        self.max_shift = max_shift
        self.use_laplacian = use_laplacian
        self.unwrap_aliasing = unwrap_aliasing
        self.min_fallback_response = min_fallback_response if min_fallback_response is not None else min_response
        self.use_sift = use_sift

    def register(self, ref_frame: np.ndarray, curr_frame: np.ndarray) -> RegistrationResult:
        """Registers curr_frame relative to ref_frame and evaluates registration quality.

        Uses SIFT feature matching with RANSAC translation as the primary registration
        method. If SIFT fails validation, falls back to full-frame phase correlation
        and horizontal strip overlap search.

        Args:
            ref_frame (np.ndarray): Reference image frame.
            curr_frame (np.ndarray): Current image frame to align.

        Returns:
            RegistrationResult: (dx, dy, response, valid, spatial_score, quality_score).
        """
        if ref_frame is None or curr_frame is None:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

        if ref_frame.shape != curr_frame.shape:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

        if ref_frame.size == 0 or curr_frame.size == 0:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

        # 1. Primary path: SIFT feature matching + RANSAC translation
        if self.use_sift:
            sift_result = self._register_sift(ref_frame, curr_frame)
            if sift_result.valid:
                return sift_result

        # 2. Fallback path: Phase correlation (full-frame -> strip fallback)
        return self._register_phase_correlation(ref_frame, curr_frame)

    def _register_sift(self, ref_frame: np.ndarray, curr_frame: np.ndarray) -> RegistrationResult:
        """Computes 2D translation via SIFT feature matching and RANSAC affine estimation."""
        try:
            gray_ref = self._to_gray_u8(ref_frame)
            gray_curr = self._to_gray_u8(curr_frame)

            sift = cv2.SIFT_create()
            kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
            kp_curr, des_curr = sift.detectAndCompute(gray_curr, None)

            if des_ref is None or des_curr is None or len(kp_ref) < 6 or len(kp_curr) < 6:
                return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des_ref, des_curr, k=2)

            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 6:
                return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

            if matrix is None:
                return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

            a, b = float(matrix[0, 0]), float(matrix[1, 0])
            scale = float(np.hypot(a, b))
            angle_deg = float(abs(np.degrees(np.arctan2(b, a))))
            dx = float(matrix[0, 2])
            dy = float(matrix[1, 2])
            shift_dist = float(np.hypot(dx, dy))

            inlier_count = int(np.sum(inliers)) if inliers is not None else 0
            inlier_ratio = float(inlier_count / len(good_matches))

            # Compute spatial overlap score on raw grayscale frames for metadata
            r_gray = gray_ref.astype(np.float32)
            c_gray = gray_curr.astype(np.float32)
            spatial_score = float(self._compute_spatial_overlap_score(r_gray, c_gray, dx, dy))
            quality_score = max(0.0, min(1.0, spatial_score))

            valid = bool(
                (abs(scale - 1.0) <= 0.05) and
                (angle_deg <= 3.0) and
                (inlier_count >= 5) and
                (inlier_ratio >= 0.45) and
                (inlier_count >= 10 or inlier_ratio >= 0.50) and
                (self.max_shift is None or shift_dist <= self.max_shift) and
                np.isfinite(dx) and np.isfinite(dy)
            )

            return RegistrationResult(
                dx, dy, float(inlier_ratio), valid,
                spatial_score=spatial_score, quality_score=quality_score
            )
        except Exception:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

    def _to_gray_u8(self, frame: np.ndarray) -> np.ndarray:
        """Converts frame to uint8 grayscale non-destructively for SIFT feature extraction."""
        if frame.ndim == 3:
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            gray = np.dot(frame[..., :3], weights)
        else:
            gray = frame

        if gray.dtype == np.uint8:
            return gray.copy()

        mn, mx = float(gray.min()), float(gray.max())
        rng = mx - mn if mx != mn else 1.0
        norm = ((gray.astype(np.float32) - mn) / rng * 255.0)
        return np.clip(norm, 0, 255).astype(np.uint8)

    def _register_phase_correlation(self, ref_frame: np.ndarray, curr_frame: np.ndarray) -> RegistrationResult:
        """Phase correlation registration pipeline (full-frame -> strip fallback)."""
        ref_prep = self._preprocess(ref_frame)
        curr_prep = self._preprocess(curr_frame)

        result = self._correlate_full_frame(ref_prep, curr_prep)

        # If full-frame fails the quality gate, try horizontal-strip fallback
        if not result.valid:
            fallback = self._try_horizontal_overlap(ref_frame, curr_frame)
            if fallback is not None and fallback.valid:
                return fallback
            # Neither path passed; return the full-frame result (captures response/scores)
            return result

        return result

    # ------------------------------------------------------------------ #
    # Internal: full-frame correlation path                               #
    # ------------------------------------------------------------------ #

    def _correlate_full_frame(self, ref_prep: np.ndarray, curr_prep: np.ndarray) -> RegistrationResult:
        """Runs phase correlation on the full preprocessed frames and applies the quality gate."""
        h, w = ref_prep.shape
        hann_window = cv2.createHanningWindow((w, h), cv2.CV_64F)

        (raw_dx, raw_dy), response = cv2.phaseCorrelate(
            ref_prep.astype(np.float64),
            curr_prep.astype(np.float64),
            window=hann_window,
        )

        if not (np.isfinite(raw_dx) and np.isfinite(raw_dy) and np.isfinite(response)):
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

        if self.unwrap_aliasing:
            dx, dy, spatial_score = self._unwrap_shift_with_score(ref_prep, curr_prep, float(raw_dx), float(raw_dy))
        else:
            dx, dy = float(raw_dx), float(raw_dy)
            spatial_score = self._compute_spatial_overlap_score(ref_prep, curr_prep, dx, dy)

        quality_score = max(0.0, min(1.0, float(spatial_score)))

        shift_dist = float(np.hypot(dx, dy))
        if self.max_shift is not None and shift_dist > self.max_shift:
            return RegistrationResult(
                float(dx), float(dy), float(response), False,
                spatial_score=float(spatial_score), quality_score=float(quality_score)
            )

        valid = bool(
            (response >= self.min_response) and
            (spatial_score >= self.min_spatial_score) and
            np.isfinite(dx) and np.isfinite(dy)
        )

        return RegistrationResult(
            float(dx), float(dy), float(response), valid,
            spatial_score=float(spatial_score), quality_score=float(quality_score)
        )

    # ------------------------------------------------------------------ #
    # Internal: horizontal-strip overlap fallback                         #
    # ------------------------------------------------------------------ #

    def _try_horizontal_overlap(
        self, ref_frame: np.ndarray, curr_frame: np.ndarray
    ) -> Optional[RegistrationResult]:
        """Tests a small set of horizontal overlap fractions for right and left hypotheses.

        For each overlap fraction f:
          - Right hypothesis (curr_frame recorded to the right of ref_frame):
              ref_strip  = rightmost (f*W) columns of ref_frame
              curr_strip = leftmost  (f*W) columns of curr_frame
              global dx  = W - strip_width + local_dx
              global dy  = local_dy
          - Left hypothesis (curr_frame recorded to the left of ref_frame):
              ref_strip  = leftmost  (f*W) columns of ref_frame
              curr_strip = rightmost (f*W) columns of curr_frame
              global dx  = -(W - strip_width) + local_dx   (i.e. negative)
              global dy  = local_dy

        Picks the hypothesis/fraction with the best spatial score that also
        passes the quality gate. Returns None if nothing passes.
        """
        h, w = ref_frame.shape[:2]
        if w < 4:
            return None

        best_result: Optional[RegistrationResult] = None
        best_score = -1.0

        for frac in _HORIZONTAL_OVERLAP_FRACTIONS:
            strip_w = max(2, int(round(frac * w)))
            if strip_w >= w:
                continue

            # ── Right hypothesis ──────────────────────────────────────────
            ref_strip  = ref_frame[:, w - strip_w:]   # right edge of ref
            curr_strip = curr_frame[:, :strip_w]       # left edge of curr

            r_result = self._correlate_strip(
                ref_strip, curr_strip,
                global_dx_offset=float(w - strip_w),
                global_dy_offset=0.0,
            )
            if r_result is not None and r_result.valid and r_result.spatial_score > best_score:
                best_score = r_result.spatial_score
                best_result = r_result

            # ── Left hypothesis ───────────────────────────────────────────
            ref_strip  = ref_frame[:, :strip_w]        # left edge of ref
            curr_strip = curr_frame[:, w - strip_w:]   # right edge of curr

            l_result = self._correlate_strip(
                ref_strip, curr_strip,
                global_dx_offset=-float(w - strip_w),
                global_dy_offset=0.0,
            )
            if l_result is not None and l_result.valid and l_result.spatial_score > best_score:
                best_score = l_result.spatial_score
                best_result = l_result

        return best_result

    def _correlate_strip(
        self,
        ref_strip: np.ndarray,
        curr_strip: np.ndarray,
        global_dx_offset: float,
        global_dy_offset: float,
    ) -> Optional[RegistrationResult]:
        """Runs phase correlation on a pair of equal-size strips and returns a RegistrationResult
        with dx/dy expressed in global (full-frame) coordinates.

        Returns None if the strips are degenerate or if the result fails the quality gate.
        """
        sh, sw = ref_strip.shape[:2]
        if sh < 2 or sw < 2:
            return None

        ref_prep = self._preprocess(ref_strip)
        curr_prep = self._preprocess(curr_strip)

        hann_window = cv2.createHanningWindow((sw, sh), cv2.CV_64F)

        try:
            (local_dx, local_dy), response = cv2.phaseCorrelate(
                ref_prep.astype(np.float64),
                curr_prep.astype(np.float64),
                window=hann_window,
            )
        except cv2.error:
            return None

        if not (np.isfinite(local_dx) and np.isfinite(local_dy) and np.isfinite(response)):
            return None

        # Convert local strip shift → global frame shift
        dx = global_dx_offset + float(local_dx)
        dy = global_dy_offset + float(local_dy)

        # Convert raw strips to 2D float32 for spatial overlap score evaluation
        if ref_strip.ndim == 3:
            r_gray = np.dot(ref_strip[..., :3], [0.114, 0.587, 0.299]).astype(np.float32)
            c_gray = np.dot(curr_strip[..., :3], [0.114, 0.587, 0.299]).astype(np.float32)
        else:
            r_gray = ref_strip.astype(np.float32)
            c_gray = curr_strip.astype(np.float32)

        strip_spatial = self._compute_spatial_overlap_score(
            r_gray, c_gray, float(local_dx), float(local_dy)
        )

        quality_score = max(0.0, min(1.0, float(strip_spatial)))

        if self.max_shift is not None and float(np.hypot(dx, dy)) > self.max_shift:
            return RegistrationResult(
                dx, dy, float(response), False,
                spatial_score=float(strip_spatial), quality_score=quality_score
            )

        valid = bool(
            (response >= self.min_fallback_response) and
            (strip_spatial >= self.min_spatial_score) and
            np.isfinite(dx) and np.isfinite(dy)
        )

        return RegistrationResult(
            dx, dy, float(response), valid,
            spatial_score=float(strip_spatial), quality_score=quality_score
        )

    def _unwrap_shift_with_score(self, img1: np.ndarray, img2: np.ndarray, raw_dx: float, raw_dy: float) -> Tuple[float, float, float]:
        """Resolves FFT periodic shift ambiguity and returns best candidate displacement along with spatial correlation score."""
        h, w = img1.shape

        # Generate candidate periodic shifts modulo image dimensions
        x_cand = [raw_dx]
        if raw_dx > 0:
            x_cand.append(raw_dx - w)
        elif raw_dx < 0:
            x_cand.append(raw_dx + w)

        y_cand = [raw_dy]
        if raw_dy > 0:
            y_cand.append(raw_dy - h)
        elif raw_dy < 0:
            y_cand.append(raw_dy + h)

        best_dx = raw_dx
        best_dy = raw_dy
        best_score = -1.0

        for cx in set(x_cand):
            for cy in set(y_cand):
                score = self._compute_spatial_overlap_score(img1, img2, cx, cy)
                if score > best_score:
                    best_score = score
                    best_dx = cx
                    best_dy = cy

        return best_dx, best_dy, best_score

    @staticmethod
    def _compute_spatial_overlap_score(img1: np.ndarray, img2: np.ndarray, shift_x: float, shift_y: float) -> float:
        """Computes normalized spatial overlap cross-correlation score for candidate (shift_x, shift_y)."""
        h, w = img1.shape
        sx = int(round(shift_x))
        sy = int(round(shift_y))

        x1_min = max(0, -sx)
        x1_max = min(w, w - sx)
        y1_min = max(0, -sy)
        y1_max = min(h, h - sy)

        if x1_max <= x1_min or y1_max <= y1_min:
            return -1.0

        x2_min = x1_min + sx
        x2_max = x1_max + sx
        y2_min = y1_min + sy
        y2_max = y1_max + sy

        overlap1 = img1[y1_min:y1_max, x1_min:x1_max]
        overlap2 = img2[y2_min:y2_max, x2_min:x2_max]

        norm1 = float(np.linalg.norm(overlap1))
        norm2 = float(np.linalg.norm(overlap2))

        if norm1 == 0.0 or norm2 == 0.0:
            return -1.0

        prod = float(np.sum(overlap1 * overlap2))
        return prod / (norm1 * norm2)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Converts frame to 2D float32 and optionally applies Laplacian whitening."""
        if frame.ndim == 3:
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            gray = np.dot(frame[..., :3], weights).astype(np.float32)
        else:
            gray = frame.astype(np.float32, copy=True)

        if not self.use_laplacian:
            return gray

        # Normalize to [0, 1] range before applying Laplacian filter
        min_val, max_val = float(gray.min()), float(gray.max())
        if max_val > min_val:
            norm = (gray - min_val) / (max_val - min_val)
        else:
            norm = np.zeros_like(gray)

        return cv2.Laplacian(norm, cv2.CV_32F, ksize=3)


class LightGlueRegistrationEngine:
    """2D Translation Registration Engine based on SuperPoint feature extraction and LightGlue feature matching.

    Uses SuperPoint deep features and LightGlue transformer matching to estimate 2D translation via RANSAC.
    Validates registration quality against response strength, spatial overlap score, and shift constraints.
    Exposes an identical registration interface to RegistrationEngine.
    """

    def __init__(
        self,
        min_spatial_score: float = 0.35,
        max_shift: Optional[float] = 150.0,
        max_num_keypoints: int = 1024,
        device: Optional[str] = None,
    ):
        """
        Args:
            min_spatial_score (float): Minimum spatial cross-correlation required for valid=True. Default: 0.35.
            max_shift (Optional[float]): Maximum allowed shift distance in pixels. Default: 150.0.
            max_num_keypoints (int): Maximum keypoints extracted per frame by SuperPoint. Default: 1024.
            device (Optional[str]): PyTorch device ('cuda' or 'cpu'). Defaults to CUDA if available, else CPU.
        """
        self.min_spatial_score = min_spatial_score
        self.max_shift = max_shift
        self.max_num_keypoints = max_num_keypoints
        self._device = device
        self._extractor = None
        self._matcher = None
        self._init_error = None

    @staticmethod
    def is_available() -> bool:
        """Returns True if PyTorch and LightGlue dependencies are installed."""
        try:
            import torch
            from lightglue import SuperPoint, LightGlue
            return True
        except ImportError:
            return False

    def _lazy_init(self) -> None:
        """Lazy-loads PyTorch, SuperPoint, and LightGlue models."""
        if self._extractor is not None and self._matcher is not None:
            return
        if self._init_error is not None:
            raise self._init_error

        try:
            import torch
            from lightglue import SuperPoint, LightGlue

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self._device)
            self._matcher = LightGlue(features="superpoint").eval().to(self._device)
        except Exception as exc:
            self._init_error = RuntimeError(
                "SuperPoint + LightGlue registration unavailable. "
                "Required dependencies missing or failed to initialize. "
                "Install with: pip install torch lightglue kornia\n"
                f"Original error: {exc}"
            )
            raise self._init_error

    def register(self, ref_frame: np.ndarray, curr_frame: np.ndarray) -> RegistrationResult:
        """Registers curr_frame relative to ref_frame using SuperPoint + LightGlue and RANSAC translation.

        Args:
            ref_frame (np.ndarray): Reference image frame.
            curr_frame (np.ndarray): Current image frame to align.

        Returns:
            RegistrationResult: (dx, dy, response, valid, spatial_score, quality_score).
        """
        if ref_frame is None or curr_frame is None:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

        if ref_frame.shape != curr_frame.shape or ref_frame.size == 0 or curr_frame.size == 0:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

        try:
            self._lazy_init()
        except RuntimeError:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

        try:
            import torch
            from lightglue.utils import rbd

            gray_ref = self._to_gray_u8(ref_frame)
            gray_curr = self._to_gray_u8(curr_frame)

            t_ref = torch.from_numpy(gray_ref).float().unsqueeze(0).unsqueeze(0).to(self._device) / 255.0
            t_curr = torch.from_numpy(gray_curr).float().unsqueeze(0).unsqueeze(0).to(self._device) / 255.0

            with torch.no_grad():
                feats_ref = self._extractor.extract(t_ref)
                feats_curr = self._extractor.extract(t_curr)
                matches_res = self._matcher({"image0": feats_ref, "image1": feats_curr})

                f_ref, f_curr, m_res = [rbd(x) for x in [feats_ref, feats_curr, matches_res]]
                matches = m_res["matches"]

            if matches is None or len(matches) < 6:
                return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

            kpts_ref = f_ref["keypoints"][matches[:, 0]].cpu().numpy()
            kpts_curr = f_curr["keypoints"][matches[:, 1]].cpu().numpy()

            src_pts = kpts_ref.reshape(-1, 1, 2).astype(np.float32)
            dst_pts = kpts_curr.reshape(-1, 1, 2).astype(np.float32)

            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

            if matrix is None:
                return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

            a, b = float(matrix[0, 0]), float(matrix[1, 0])
            scale = float(np.hypot(a, b))
            angle_deg = float(abs(np.degrees(np.arctan2(b, a))))
            dx = float(matrix[0, 2])
            dy = float(matrix[1, 2])
            shift_dist = float(np.hypot(dx, dy))

            inlier_count = int(np.sum(inliers)) if inliers is not None else 0
            inlier_ratio = float(inlier_count / len(matches)) if len(matches) > 0 else 0.0

            r_gray = gray_ref.astype(np.float32)
            c_gray = gray_curr.astype(np.float32)
            spatial_score = float(RegistrationEngine._compute_spatial_overlap_score(r_gray, c_gray, dx, dy))
            quality_score = max(0.0, min(1.0, spatial_score))

            valid = bool(
                (abs(scale - 1.0) <= 0.05) and
                (angle_deg <= 3.0) and
                (inlier_count >= 10) and
                (inlier_ratio >= 0.45) and
                (self.max_shift is None or shift_dist <= self.max_shift) and
                (spatial_score >= self.min_spatial_score) and
                np.isfinite(dx) and np.isfinite(dy)
            )

            return RegistrationResult(
                dx, dy, float(inlier_ratio), valid,
                spatial_score=spatial_score, quality_score=quality_score
            )
        except Exception:
            return RegistrationResult(0.0, 0.0, 0.0, False, spatial_score=0.0, quality_score=0.0)

    @staticmethod
    def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
        """Converts frame to uint8 grayscale non-destructively for feature extraction."""
        if frame.ndim == 3:
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            gray = np.dot(frame[..., :3], weights)
        else:
            gray = frame

        if gray.dtype == np.uint8:
            return gray.copy()

        mn, mx = float(gray.min()), float(gray.max())
        rng = mx - mn if mx != mn else 1.0
        norm = ((gray.astype(np.float32) - mn) / rng * 255.0)
        return np.clip(norm, 0, 255).astype(np.uint8)

