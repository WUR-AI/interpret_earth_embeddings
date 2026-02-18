from geotessera import GeoTessera
import shapely
import argparse
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
import math
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio import MemoryFile
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
import time
from typing import Any


def get_point_utm_crs(lon: float, lat: float) -> str:
    """Determine local UTM crs code from given latitude and longitude.

    :param lon: longitude in WGS84
    :param lat: latitude in WGS84
    :return: UTM crs code
    """
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    utm_crs = f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}"
    return utm_crs


def point_reprojection(lon: float, lat: float, src_crs: str, dst_crs: str):
    """Reproject a point from one to another CRS systems.

    :param lon: longitude
    :param lat: latitude
    :param src_crs: source CRS
    :param dst_crs: destination CRS
    :return: (lon, lat) in reprojection coordinates
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(lon, lat)


def crs_to_pixel_coords(x, y, transform):
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return col, row


def create_bbox_with_radius(lon: float, lat: float, radius: float, utm_crs: str = None, return_wgs: bool = False) -> shapely.geometry.Polygon:
    """Creates a square bounding box of given radius (meters) around lon/lat.

    :param lon: Longitude (EPSG:4326)
    :param lat: Latitude (EPSG:4326)
    :param radius: Radius in meters
    :param utm_crs: Optional EPSG code for UTM CRS (e.g. "EPSG:32633")
    :param return_wgs: If True, returns WGS84 GeoJSON, else UTM Polygon
    """

    # Determine UTM CRS
    utm_crs = utm_crs or get_point_utm_crs(lon, lat)

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x, y = to_utm.transform(lon, lat)

    # Create bbox in UTM
    square_utm = box(x - radius, y - radius, x + radius, y + radius)

    if return_wgs:
        to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        square_wgs = transform(to_wgs.transform, square_utm)
        return square_wgs

    return square_utm


def reproject_dataset(src_raster: MemoryFile, dst_crs: str) -> MemoryFile:
    """Reprojects Memory file if it's not in dst_crs.

    :param src_raster: Raster file to reproject.
    :param dst_crs: CRS to reproject.
    """
    dst_crs = CRS.from_user_input(dst_crs)
    if src_raster.crs == dst_crs:
        return src_raster, None

    # Reprojection dim
    transform, width, height = calculate_default_transform(src_raster.crs, dst_crs, src_raster.width, src_raster.height, *src_raster.bounds)

    # Update metadata
    metadata = src_raster.meta.copy()
    metadata.update(crs=dst_crs, transform=transform, width=width, height=height, )

    memfile = MemoryFile()
    dst = memfile.open(**metadata)
    for i in range(1, src_raster.count + 1):
        reproject(source=rasterio.band(src_raster, i), destination=rasterio.band(dst, i), src_transform=src_raster.transform, src_crs=src_raster.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest, )
    return dst, memfile


def get_tessera_embeds(
    row: pd.Series,
    year: int,
    save_dir: str,
    tile_size: int,
    tessera_con: GeoTessera | None,
    padding: int = 10
    ) -> None:
    embed_tile_name = os.path.join(save_dir, f"{row.row_id}_tessera_y-{year}.tif")
    if os.path.exists(embed_tile_name):
        return

    # Local utm projection
    utm_crs = get_point_utm_crs(row.lon, row.lat)
    lon_utm, lat_utm = point_reprojection(row.lon, row.lat, "EPSG:4326", utm_crs)

    # Bounding box
    radius = math.ceil(tile_size / 2) + padding
    bbox = create_bbox_with_radius(row.lon, row.lat, radius=radius, utm_crs=utm_crs, return_wgs=True)

    # Request to tessera
    tiles_to_fetch = tessera_con.registry.load_blocks_for_region(bounds=bbox.bounds, year=int(year))

    # Mosaic returned tiles for the bbox
    tiles = []
    memfiles = []

    for _, _, _, embedding, crs, transform in tessera_con.fetch_embeddings(tiles_to_fetch):
        memfile = MemoryFile()
        memfiles.append(memfile)

        tile = memfile.open(driver="GTiff", height=embedding.shape[0], width=embedding.shape[1], count=embedding.shape[
            2], dtype=embedding.dtype, crs=crs, transform=transform, )

        for c in range(embedding.shape[2]):
            tile.write(embedding[:, :, c], c + 1)

        reproject_tile, reproject_memfile = reproject_dataset(tile, utm_crs)
        tiles.append(reproject_tile)
        if reproject_memfile:
            memfiles.append(reproject_memfile)

    mosaic, mosaic_transform = merge(tiles)
    mosaic = mosaic.transpose(1, 2, 0)

    for tile in tiles:
        tile.close()
    for mf in memfiles:
        mf.close()

    # Crop patch tile
    col, row = crs_to_pixel_coords(lon_utm, lat_utm, mosaic_transform)
    half = tile_size // 2
    row_min = row - half
    row_max = row + half
    col_min = col - half
    col_max = col + half
    crop = mosaic[row_min:row_max, col_min:col_max, :]

    # Save array
    os.makedirs(save_dir, exist_ok=True)

    crop_transform = mosaic_transform * Affine.translation(col_min, row_min)

    height, width, channels = crop.shape

    with rasterio.open(embed_tile_name, "w", driver="GTiff", height=height, width=width, count=channels, dtype=crop.dtype, crs=utm_crs, transform=crop_transform, ) as dst:
        for i in range(channels):
            dst.write(crop[:, :, i], i + 1)

    print(f"GeoTIFF saved as {embed_tile_name}")


def main(start, stop, root_dir, year=2024, tile_size=128, cache_root=None):
    csv_path = os.path.join(root_dir, 'data', 'dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv')
    df = pd.read_csv(csv_path)

    # Subset for selected samples
    df = df[(df['random_sample'] == 1) | (df['lc_stratified_sample'] == 1)]
    df.reset_index(drop=True, inplace=True)

    # Unique grid id - for spatial prox ordering
    df["grid_x"] = np.floor((df["lon"] + 180) / 20).astype(int)
    df["grid_y"] = np.floor((df["lat"] + 90) / 20).astype(int)
    df["grid_id"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)
    df = df.sort_values(["grid_y", "grid_x"]).reset_index(drop=True)

    # Pre-filter out IDs that already have a tile on disk
    save_dir = os.path.join(root_dir, 'data', f'tessera_{year}')
    os.makedirs(save_dir, exist_ok=True)
    existing_ids: set[int] = set()

    for fn in os.listdir(save_dir):
        if not fn.endswith(f"_tessera_y-{year}.tif"):
            continue
        # Filenames are of the form "{row_id}_tessera_y-{year}.tif"
        try:
            rid_str = fn.split("_", 1)[0]
            rid = int(rid_str)
        except ValueError:
            continue
        existing_ids.add(rid)

    if existing_ids:
        df = df[~df["id"].isin(existing_ids)]
        df.reset_index(drop=True, inplace=True)


    # Slice per SLURM array task
    total = len(df)
    task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_count_env = os.environ.get("SLURM_ARRAY_TASK_COUNT")

    if task_id_env is not None and task_count_env is not None:
        task_id = int(task_id_env)
        task_count = int(task_count_env)
        chunk = int(math.ceil(total / task_count)) if task_count > 0 else total
        start_idx = task_id * chunk
        stop_idx = min((task_id + 1) * chunk, total)
    else:
        # Fallback to explicit start/stop indices (e.g. when run locally)
        start_idx = start
        stop_idx = min(stop, total)

    df = df.iloc[start_idx:stop_idx]
    df.rename(columns={'id': 'row_id'}, inplace=True)

    print(f"Processing {len(df)} locations")

    # Tessera connection        
    cache_dir = os.makedirs(os.path.join(cache_root, 'tessera_cache'), exist_ok=True)
    gt = GeoTessera(cache_dir=cache_dir, embeddings_dir=cache_dir)

    fast_save_dir = os.path.join(cache_root, 'tessera_data', f'tessera_{year}')
    os.makedirs(fast_save_dir, exist_ok=True)

    # Download
    for row in df.itertuples():
        try:
            get_tessera_embeds(row, year, fast_save_dir, tile_size, tessera_con=gt)
        except Exception as e:
            try:
                get_tessera_embeds(row, year, fast_save_dir, tile_size, tessera_con=gt, padding=1000)
            except Exception as e:
                print(f"{row.row_id} did not get embedded: {e}")
                path = os.path.join('logs', f'tessera_skipped_{start}_{stop}.txt')
                with open(path, 'a') as f:
                    f.write(f"{row.row_id} because {e}\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main script with configurable parameters.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory path.")
    parser.add_argument("--cache_root", type=str, required=True, help="Directory to store embed cache (requires large storage limit).")
    parser.add_argument("--year", type=int, default=2024, help="Year (default: 2024).")
    parser.add_argument("--size", type=int, default=128, help="Image size (default: 128).")

    args = parser.parse_args()
    print(f"Starting download of tessera data for locations from index {args.start} to {args.stop}...")
    main(start=args.start, stop=args.stop, root_dir=args.root_dir, year=args.year, tile_size=args.size, cache_root=args.cache_root)