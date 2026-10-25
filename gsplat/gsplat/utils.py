"""Python bindings for binning and sorting gaussians"""

from typing import Tuple
import os

from jaxtyping import Float, Int
from torch import Tensor
import torch

import gsplat.cuda as _C


# --- Intersect-buffer capacity manager (opt3: remove the per-forward D2H sync) --
#
# compute_cumulative_intersects() historically read cum_tiles_hit[-1].item()
# every forward to size the tile-intersect buffers. That is a hard device->host
# sync that blocks CPU run-ahead (and CUDA-graph capture). Instead we run the
# binning at a *fixed preallocated capacity*: for each resolution (tile_bounds)
# we take exact counts during a short warmup, then freeze a capacity of
# SAFETY x observed-max and stop syncing. A device-side assert in the CUDA
# binding aborts loudly if the capacity ever proves too small (never silent
# corruption). Set GSPLAT_ISECT_WARMUP=0 to disable (always exact/synced).
_ISECT_CAP_STATE = {}  # (tiles_x, tiles_y, gaussian_count) -> {"cap", "max_ni", "warm"}
_ISECT_CAP_WARMUP = int(os.environ.get("GSPLAT_ISECT_WARMUP", "8"))
_ISECT_CAP_SAFETY = float(os.environ.get("GSPLAT_ISECT_SAFETY", "3.0"))


def resolve_intersect_capacity(cum_tiles_hit: Tensor, num_points: int, tile_bounds) -> int:
    """Capacity for the intersect buffers this forward.

    Keyed by (resolution, Gaussian count): the intersect count depends on both
    the tile grid AND the number of Gaussians, and the latter changes when
    densification (--is_ad) grows/prunes points, so keying by resolution alone
    let a frozen capacity be exceeded after a densify. Each (resolution, count)
    combination warms up independently: during warmup it returns the exact count
    (one D2H sync, numerically identical to the pre-opt3 path since the padded
    tail is then empty) while tracking the max; afterwards it returns a frozen
    capacity (SAFETY x max) with no sync. SAFETY headroom absorbs the drift of
    Gaussian footprints within a fixed count/resolution. A device-side assert in
    the CUDA binding still aborts loudly if the headroom is ever insufficient.
    """
    if _ISECT_CAP_WARMUP <= 0:
        return int(cum_tiles_hit[-1].item()) if num_points > 0 else 0
    key = (int(tile_bounds[0]), int(tile_bounds[1]), int(num_points))
    st = _ISECT_CAP_STATE.get(key)
    if st is None:
        st = {"cap": None, "max_ni": 0, "warm": 0}
        _ISECT_CAP_STATE[key] = st
    if st["cap"] is not None:
        return st["cap"]  # steady state: no sync
    ni = int(cum_tiles_hit[-1].item()) if num_points > 0 else 0
    if ni > st["max_ni"]:
        st["max_ni"] = ni
    st["warm"] += 1
    if st["warm"] >= _ISECT_CAP_WARMUP:
        st["cap"] = max(1, int(st["max_ni"] * _ISECT_CAP_SAFETY))
    return ni


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
    )
    return (isect_ids, gaussian_ids)


def get_tile_bin_edges(
    num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"], tile_bounds: Tuple[int, int, int]
) -> Int[Tensor, "num_intersects 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _C.get_tile_bin_edges(num_intersects, isect_ids_sorted.contiguous(), tile_bounds)


def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"]
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, cov2d.contiguous())


def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted,tile_bounds)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
