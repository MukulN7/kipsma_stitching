from typing import List, Optional
import numpy as np
from microstitch.mosaic import MosaicManager, MosaicTile


class MosaicCanvasRenderer:
    """Renders placed MosaicTile instances from MosaicManager into a single pixel canvas.

    Overlap Policy:
        Deterministic Overwrite — Later accepted tiles overwrite earlier tiles
        in overlapping pixel regions (`canvas[slice] = tile_frame`).

    Coordinate & Bounds Logic:
        1. Reads global bounds `(min_x, min_y, max_x, max_y)` across all placed tiles.
        2. Allocates canvas of size `(height, width)` where:
           `width = ceil(max_x - min_x)`
           `height = ceil(max_y - min_y)`
        3. Translates each tile's global position `(X, Y)` to canvas pixel indices:
           `col_start = int(round(global_x - min_x))`
           `row_start = int(round(global_y - min_y))`
        4. Translates negative global coordinates into positive non-zero canvas pixel indices.
        5. Preserves frame data type (`uint8` or `uint16`).
    """

    def render(
        self,
        manager: MosaicManager,
        frames: Optional[List[np.ndarray]] = None,
        blend: bool = True,
    ) -> np.ndarray:
        """Renders all placed tiles from MosaicManager into a consolidated pixel array canvas.

        Args:
            manager (MosaicManager): Manager instance holding placed tiles and global positions.
            frames (Optional[List[np.ndarray]]): Optional explicit list of frame arrays matching tiles.
            blend (bool): If True (default), applies 2D linear feather blending in overlap regions.
                          If False, applies deterministic overwrite of later tiles over earlier tiles.

        Returns:
            np.ndarray: Consolidated pixel canvas. Returns empty uint8 array if no tiles exist.
        """
        tiles = manager.get_tiles()
        if not tiles:
            return np.array([], dtype=np.uint8)

        min_x, min_y, max_x, max_y = manager.get_global_bounds()
        canvas_width = int(np.ceil(max_x - min_x))
        canvas_height = int(np.ceil(max_y - min_y))

        if canvas_width <= 0 or canvas_height <= 0:
            return np.array([], dtype=np.uint8)

        target_dtype = tiles[0].dtype

        # Resolve first frame array to check channels
        first_frame = tiles[0].frame if tiles[0].frame is not None else (frames[0] if (frames and len(frames) > 0) else None)
        if first_frame is None:
            return np.array([], dtype=target_dtype)

        is_color = (first_frame.ndim == 3)
        if is_color:
            canvas_shape = (canvas_height, canvas_width, first_frame.shape[2])
        else:
            canvas_shape = (canvas_height, canvas_width)

        if not blend:
            # Deterministic Overwrite mode
            canvas = np.zeros(canvas_shape, dtype=target_dtype)
            for i, tile in enumerate(tiles):
                tile_frame = tile.frame if tile.frame is not None else (frames[i] if (frames and i < len(frames)) else None)
                if tile_frame is None:
                    continue

                col_start = int(round(tile.global_x - min_x))
                row_start = int(round(tile.global_y - min_y))
                h, w = tile_frame.shape[:2]
                r1 = max(0, row_start); r2 = min(canvas_height, row_start + h)
                c1 = max(0, col_start); c2 = min(canvas_width, col_start + w)

                fr1 = r1 - row_start; fr2 = fr1 + (r2 - r1)
                fc1 = c1 - col_start; fc2 = fc1 + (c2 - c1)

                if is_color:
                    canvas[r1:r2, c1:c2, :] = tile_frame[fr1:fr2, fc1:fc2, :]
                else:
                    canvas[r1:r2, c1:c2] = tile_frame[fr1:fr2, fc1:fc2]

            return canvas

        # Weighted Accumulation Blending mode
        accum_canvas = np.zeros(canvas_shape, dtype=np.float32)
        weight_canvas = np.zeros(canvas_shape, dtype=np.float32)

        for i, tile in enumerate(tiles):
            tile_frame = tile.frame if tile.frame is not None else (frames[i] if (frames and i < len(frames)) else None)
            if tile_frame is None:
                continue

            col_start = int(round(tile.global_x - min_x))
            row_start = int(round(tile.global_y - min_y))
            h, w = tile_frame.shape[:2]

            r1 = max(0, row_start); r2 = min(canvas_height, row_start + h)
            c1 = max(0, col_start); c2 = min(canvas_width, col_start + w)

            fr1 = r1 - row_start; fr2 = fr1 + (r2 - r1)
            fc1 = c1 - col_start; fc2 = fc1 + (c2 - c1)

            weight_2d = self._generate_weight_mask(h, w)
            crop_weight = weight_2d[fr1:fr2, fc1:fc2]
            crop_frame = tile_frame[fr1:fr2, fc1:fc2].astype(np.float32)

            if is_color:
                w_broadcast = crop_weight[:, :, None]
                accum_canvas[r1:r2, c1:c2, :] += crop_frame * w_broadcast
                weight_canvas[r1:r2, c1:c2, :] += w_broadcast
            else:
                accum_canvas[r1:r2, c1:c2] += crop_frame * crop_weight
                weight_canvas[r1:r2, c1:c2] += crop_weight

        # Normalize accumulation canvas by weight canvas where weight > 0
        valid_mask = weight_canvas > 0
        blended = np.zeros(canvas_shape, dtype=np.float32)
        blended[valid_mask] = accum_canvas[valid_mask] / weight_canvas[valid_mask]

        # Clip and cast to target dtype (uint8 or uint16)
        if np.issubdtype(target_dtype, np.integer):
            info = np.iinfo(target_dtype)
            blended = np.clip(np.round(blended), info.min, info.max)

        return blended.astype(target_dtype)

    @staticmethod
    def _generate_weight_mask(h: int, w: int) -> np.ndarray:
        """Generates a 2D linear feathering weight mask for a tile of shape (h, w)."""
        dist_x = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
        dist_y = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1
        weight_2d = np.minimum(dist_x[None, :], dist_y[:, None]).astype(np.float32)
        return weight_2d
