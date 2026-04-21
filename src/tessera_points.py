import multiprocessing
import os
import pandas as pd
import threading
import time
import socket

from src.utils.errors import NoDataError

_csv_lock = threading.Lock()
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

def get_tessera_point_embeds(lon, lat, name_loc, year, tessera_con, output_csv):
    points = [(lon, lat)]
    embeddings = tessera_con.sample_embeddings_at_points(points, year=year)
    if embeddings.min() == 0 and embeddings.max() == 0:
        raise NoDataError(f'Could not retrieve tessera embeddings for {name_loc} with coordinates: ({lat}, {lon})')
    embed_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(1, 129)]
    )
    embed_df["id"] = name_loc
    with _csv_lock:
        embed_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv))

def _point_worker_fetch(args: tuple) -> str:
    """Multiprocessing worker — reuses the per-process GeoTessera instance."""
    lon, lat, name_loc, year, output_csv = args
    get_tessera_point_embeds(
        lon=lon, lat=lat, name_loc=name_loc, year=year, tessera_con=_process_gt, output_csv=output_csv)
    return name_loc

def fetch_tessera_points(dataset, data_dir, year, cache_dir, workers, retry_stuck):
    print(f"Fetching {dataset} TESSERA point embeddings to data_dir={data_dir}\n"
          f"year={year}")

    # Target save_dir and save_csv_name
    target_dict = {
        "biomass":  ("downstream_tasks", "biomass_cleaned_centre.csv", "biomass_tessera_centre.csv"),
        'cropharvest': ("downstream_tasks", "cropharvest_cleaned_global_threshold-200-sample.csv","cropharvest_200_tessera_centre.csv"),
       'global': (f"tessera_{year}_centre", "dw_locations_2026-02-13-1659_year-2024_50m_spherical_100k_random_stratified.csv", "tessera_centre.csv")
    }
    assert dataset in target_dict.keys(), KeyError

    # CSV with input coords
    csv_path = os.path.join(data_dir, target_dict[dataset][0], target_dict[dataset][1])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Save dir
    save_dir = os.path.join(data_dir, target_dict[dataset][0])
    os.makedirs(save_dir, exist_ok=True)
    output_csv_path = os.path.join(data_dir, target_dict[dataset][0], target_dict[dataset][-1])

    # Cache dir
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, "cache", "tessera")
    os.makedirs(cache_dir, exist_ok=True)

    # Obtain coords
    input_df = pd.read_csv(csv_path)
    n_total = len(input_df)
    print(f"Records: {n_total} total")

    # Filter out existing points
    if os.path.exists(output_csv_path):
        existing_records = set(pd.read_csv(output_csv_path)['id'])
        input_df = input_df[~input_df["index"].isin(existing_records)]
        print(f"{len(existing_records)} already obtained")
    else:
        os.makedirs(os.path.join(data_dir, target_dict[dataset][0]), exist_ok=True)

    # Deal with stuck files
    stuck_file = os.path.join(save_dir, "tessera_points_stuck.txt")
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

    # Send to workers
    _use_local_registry = os.path.exists(os.path.join(cache_dir, "registry.parquet"))
    HEARTBEAT = 15  # seconds between "still fetching" log lines
    TILE_TIMEOUT = 180  # seconds per record before the worker process is killed

    _pool_initargs = (cache_dir, _use_local_registry, str(cache_dir))
    pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=_pool_initargs)

    done = 0

    try:
        for _, row in input_df.iterrows():
            name_loc = int(row['index'].item())
            args = (row.lon, row.lat, name_loc, year, output_csv_path)
            result = pool.apply_async(_point_worker_fetch, (args,))
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
                except NoDataError as exc:
                    print(NoDataError)
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
            if done % 20 == 0 or done == len(input_df):
                print(f"  {done}/{n_total}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        pool.terminate()
        pool.join()
        return

    pool.close()
    pool.join()

    print(f"Done. Points saved to: {output_csv_path}")
