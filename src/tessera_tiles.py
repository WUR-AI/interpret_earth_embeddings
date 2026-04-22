import multiprocessing
import time
import socket
from geotessera import GeoTessera
from pyproj import Transformer
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

from src.utils.errors import PartialTileError, NoTileError

_process_gt = None


def _init_worker(cache_dir: str, use_local_registry: bool, registry_dir: str) -> None:
    """Pool initializer: runs once per worker process to set up GeoTessera."""
    from geotessera import GeoTessera
    global _process_gt
    socket.setdefaulttimeout(60)
    embeddings_dir = os.path.join(cache_dir, "raw_embeds")
    gt_kwargs = {"verify_hashes": False, "embeddings_dir": embeddings_dir, "dataset_version": 'v1'}
    if use_local_registry:
        _process_gt = GeoTessera(registry_dir=registry_dir, **gt_kwargs)
    else:
        _process_gt = GeoTessera(cache_dir=cache_dir, **gt_kwargs)


def _tile_worker_fetch(args: tuple) -> str:
    row, year, save_name, tile_size = args
    get_tessera_embeds(
        row=row, year=year, save_name=save_name, tile_size=tile_size, tessera_con=_process_gt,
    )
    return row["index"]


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

def get_tiles(lat_center, lon_center, half_size_m=75, year=2024):
    """Find all 0.1deg tiles (referenced at 0.05deg) that overlap AOI."""

    # Convert half-size from meters to degrees (approximate)
    half_lat_deg = half_size_m / 111320.0
    half_lon_deg = half_size_m / (111320.0 * np.cos(np.radians(lat_center)))

    # AOI bounds
    lat_min = lat_center - half_lat_deg
    lat_max = lat_center + half_lat_deg
    lon_min = lon_center - half_lon_deg
    lon_max = lon_center + half_lon_deg

    tile_size = 0.1

    # Find tile indices overlapping the AOI
    i_min = int(np.floor(lat_min / tile_size))
    i_max = int(np.floor(lat_max / tile_size))
    j_min = int(np.floor(lon_min / tile_size))
    j_max = int(np.floor(lon_max / tile_size))

    tiles = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            ref_lat = i * tile_size + 0.05  # tile reference (center)
            ref_lon = j * tile_size + 0.05
            tiles.append((year, round(ref_lon, 10), round(ref_lat, 10)))

    return tiles


def get_tessera_embeds(
        row: pd.Series,
        year: int,
        save_name: str,
        tile_size: int,
        tessera_con: GeoTessera | None,
        padding: int = 100
    ) -> None:

    # Skip if tile exists
    if os.path.exists(save_name):
        return

    # Local utm projection
    utm_crs = get_point_utm_crs(row.lon, row.lat)
    lon_utm, lat_utm = point_reprojection(row.lon, row.lat, "EPSG:4326", utm_crs)

    # Request to tessera
    radius = math.ceil(tile_size / 2) + padding
    tiles_to_fetch = get_tiles(lat_center=row.lat, lon_center=row.lon, half_size_m=radius * 10, year=int(year))

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

    if len(tiles) == 0:
        raise NoTileError(f"No tiles found for {row.row_id}") # if no tiles, add to skipped.txt

    mosaic, mosaic_transform = merge(tiles)
    mosaic = mosaic.transpose(1, 2, 0)

    for tile in tiles:
        tile.close()
    for mf in memfiles:
        mf.close()

    # Crop patch tile
    c, r = crs_to_pixel_coords(lon_utm, lat_utm, mosaic_transform)
    half = tile_size // 2
    row_min = r - half
    row_max = r + half
    col_min = c - half
    col_max = c + half

    if (row_min < 0 or row_max < 0 or col_min < 0 or col_max < 0):
        # retry with bigger padding
        if padding > 500:
            raise NoTileError(f"Padding {padding} > 500")
        get_tessera_embeds(row, year, save_name, tile_size, tessera_con, padding=padding+100)

    crop = mosaic[row_min:row_max, col_min:col_max, :]
    if not crop.shape == (128, 128, 128):
        if (crop.min() == 0.0 and crop.max() == 0.0):
            raise NoTileError(f"No tiles found for {row.row_id}")
        raise PartialTileError(f"Crop {row.row_id}, size is {crop.shape}")

    if (crop.min() == 0.0 and crop.max() == 0.0):
        raise NoTileError(f"Crop {row.row_id} has embeddings of 0.0s with tiles: {tiles_to_fetch}")

    # Save array
    crop_transform = mosaic_transform * Affine.translation(col_min, row_min)
    height, width, channels = crop.shape

    with rasterio.open(save_name, "w", driver="GTiff", height=height, width=width, count=channels, dtype=crop.dtype, crs=utm_crs, transform=crop_transform, ) as dst:
        for i in range(channels):
            dst.write(crop[:, :, i], i + 1)

    print(f"GeoTIFF saved as {save_name}")



def fetch_tessera_tiles(
        dataset,
        tile_size,
        year,
        data_dir,
        cache_dir,
        workers,
        retry_stuck,
) -> None:
    print(
        f"Fetching TESSERA tiles to data_dir={data_dir}\n"
        f"tile_size={tile_size}\n"
        f"year={year}\n"
    )

    # Target save_dir and save_csv_name
    target_dict = {
        "biomass": ("downstream_tasks/biomass_cleaned_centre.csv", 'biomass_index_tessera_y-year.tif'),
        'cropharvest': ("downstream_tasks/cropharvest_cleaned_global_threshold-200-sample.csv", 'crop_harvest_index_tessera_y-year.tif'),
        'global': ("dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv", "index_tessera_y-year.tif")
    }
    assert dataset in target_dict.keys(), KeyError

    # CSV with input coords
    csv_path = os.path.join(data_dir, target_dict[dataset][0])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Save dir
    save_dir = os.path.join(data_dir, f"tessera_{year}")
    os.makedirs(save_dir, exist_ok=True)

    # Cache dir
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, "cache", "tessera")
    os.makedirs(cache_dir, exist_ok=True)

    # Obtain coords
    input_df = pd.read_csv(csv_path)
    # Subset for selected samples
    if dataset == 'global':
        input_df = input_df[(input_df['random_sample'] == 1) | (input_df['lc_stratified_sample'] == 1)]
        input_df.reset_index(drop=True, inplace=True)

    n_total = len(input_df)
    print(f"Records: {n_total} total")

    # Filter out existing points
    existing_records = set()
    for fn in os.listdir(save_dir):
        if not fn.endswith(f"_tessera_y-{year}.tif"):
            continue
        try:
            rid_str = fn.split("_", 1)[0]
            rid = int(rid_str)
        except ValueError:
            continue
        existing_records.add(rid)

    if existing_records:
        input_df = input_df[~input_df["id"].isin(existing_records)]
        input_df.reset_index(drop=True, inplace=True)
        print(f"{len(existing_records)} already obtained")

    # Deal with stuck files
    stuck_file = os.path.join(save_dir, "tessera_tiles_stuck.txt")
    stuck_records = set()
    if os.path.exists(stuck_file):
        if retry_stuck:
            os.remove(stuck_file)
        else:
            with open(stuck_file, "r") as f:
                stuck_records = set(line.strip() for line in f.readlines())
            if stuck_records:
                print(f"Skipping {len(stuck_records)} previously-stuck record(s): {sorted(stuck_records)}")
    input_df = input_df[~input_df["index"].isin(stuck_records)]

    # Sort spatially
    # input_df["grid_x"] = np.floor((input_df["lon"] + 180) / 20).astype(int)
    # input_df["grid_y"] = np.floor((input_df["lat"] + 90) / 20).astype(int)
    # input_df["grid_id"] = input_df["grid_x"].astype(str) + "_" + input_df["grid_y"].astype(str)
    # input_df = input_df.sort_values(["grid_y", "grid_x"]).reset_index(drop=True)

    # Send to workers
    _use_local_registry = os.path.exists(os.path.join(cache_dir, "registry.parquet"))
    HEARTBEAT = 45  # seconds between "still fetching" log lines
    TILE_TIMEOUT = 180  # seconds per record before the worker process is killed

    _pool_initargs = (cache_dir, _use_local_registry, str(cache_dir))
    pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=_pool_initargs)

    done = 0
    name_template = target_dict[dataset][-1].replace('year', str(year))
    try:
        for _, row in input_df.iterrows():
            name_loc = int(row['index'].item())
            save_name = os.path.join(save_dir, name_template.replace('index', str(name_loc)))
            args = (row, year, save_name, tile_size)
            result = pool.apply_async(_tile_worker_fetch, (args,))
            start = time.monotonic()
            timed_out = False
            while True:
                try:
                    result.get(timeout=HEARTBEAT)
                    break  # completed successfully
                except multiprocessing.TimeoutError:
                    elapsed = int(time.monotonic() - start)
                    if elapsed >= TILE_TIMEOUT:
                        timed_out = True
                        break
                    print(f"  ... fetching {name_loc} ({elapsed}s)")
                except NoTileError:
                    print(f"  Skipped {name_loc}: no TESSERA data for this location/year")
                    break
                except PartialTileError:
                    print(f"  Skipped {name_loc}: tile too close to mosaic edge, not enough context")
                    break
                except Exception as exc:
                    print(f"  ERROR fetching {name_loc}: {exc}")
                    break

            if timed_out:
                pool.terminate()
                pool.join()
                pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=_pool_initargs)
                with open(stuck_file, "a") as fh:
                    fh.write(str(name_loc) + "\n")
                print(
                    f"  Stuck: {name_loc}  "
                    f"lon={row.lon:.4f} lat={row.lat:.4f} year={int(row.year)}"
                )

            done += 1
            if done % 100 == 0 or done == len(input_df):
                print(f"  {done}/{n_total}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        pool.terminate()
        pool.join()
        return

    pool.close()
    pool.join()

    print(f"Done. Tiles saved to: {save_dir}")
