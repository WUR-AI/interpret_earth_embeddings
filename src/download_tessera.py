from geotessera import GeoTessera
import shapely
import argparse
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
import math
import os
import pandas as pd
import rasterio
from rasterio import MemoryFile
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject


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


def get_tessera_embeds(row: pd.Series, year: int, save_dir: str, tile_size: int, tessera_con: GeoTessera | None, ) -> None:
    embed_tile_name = os.path.join(save_dir, f"{row.row_id}_tessera_y-{year}.tif")
    if os.path.exists(embed_tile_name):
        return

    # Local utm projection
    utm_crs = get_point_utm_crs(row.lon, row.lat)
    lon_utm, lat_utm = point_reprojection(row.lon, row.lat, "EPSG:4326", utm_crs)

    # Bounding box
    radius = math.ceil(tile_size / 2) + 10
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


def main(start, stop, root_dir, year=2024, tile_size=128):
    csv_path = os.path.join(root_dir, 'data', 'dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv')
    df = pd.read_csv(csv_path)

    # Subset for process
    df = df[(df['random_sample'] == 1) | (df['lc_stratified_sample'] == 1)]
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[start : min(stop, len(df)+1)]
    df.rename(columns={'id': 'row_id'}, inplace=True)

    save_dir = os.path.join(root_dir, 'data', f'tessera_{year}')
    os.makedirs(save_dir, exist_ok=True)

    # Tessera connection
    cache_dir = os.path.join('data', 'cache', "tessera")
    gt = GeoTessera(cache_dir=cache_dir)

    # Shuffle for multi-proces downloading
    for row in df.itertuples():
        try:
            get_tessera_embeds(row, year, save_dir, tile_size, tessera_con=gt)
        except Exception as e:
            print(f"{row.row_id} did not get embedded: {e}")
            path = os.path.join(root_dir, 'data', 'tessera_skipped.txt')
            with open(path, 'a' if os.path.exists(path) else 'w') as f:
                f.write(str(int(row.row_id))+ '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main script with configurable parameters.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory path.")
    parser.add_argument("--year", type=int, default=2024, help="Year (default: 2024).")
    parser.add_argument("--size", type=int, default=128, help="Image size (default: 128).")

    args = parser.parse_args()
    print(f"Starting download of tessera data for locations from index {args.start} to {args.stop}...")
    main(start=args.start, stop=args.stop, root_dir=args.root_dir, year=args.year, tile_size=args.size)