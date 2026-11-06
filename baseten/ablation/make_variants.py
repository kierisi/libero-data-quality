"""Build the three single-corruption LIBERO-Object datasets from the two published copies, with NO video re-encoding.

  instruction_only = clean files, with task labels taken from the corrupted copy
  gripper_only     = clean files, with the action column (and its stats) taken from the corrupted copy
  camera_only      = corrupted files (whose videos carry the swapped cameras), with task labels and actions
                     (and action stats) put back from the clean copy

Why this works: in the corrupted copy, each corruption touched different episodes and different columns.
Instruction swaps only changed task labels, gripper flips only changed action dim 6, and camera swaps only
changed the videos. So each single-corruption dataset is one copy with some columns taken from the other.
Episode i is the same demonstration in both copies, and frame rows line up one to one (checked below).

How LeRobot 0.6.1 reads what we touch (verified against the v0.6.1 source):
  - data/*.parquet is loaded with the dataset's declared HF features, and the per-frame "task" string is
    meta.tasks.iloc[task_index], i.e. tasks.parquet is read BY POSITION. The two copies number their tasks
    differently (first-seen order), so task_index is remapped through the task text.
  - Normalization stats at train time come only from meta/stats.json. The per-episode "stats/*" columns in
    meta/episodes are dropped at load, so they are replaced only for consistency.
  - Columns are replaced at the pyarrow level, so the parquet schema (fixed-size float32 lists, the
    huggingface schema metadata, one row group per episode) is byte-for-byte the writer's own layout.

Also repairs info.json counts from the clean copy (the published corrupted info.json is stale: 453/66,832).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

NS = "kierisi"
REV = "v3.0"
ARMS = ("instruction_only", "gripper_only", "camera_only")


def log(*a):
    print("[variants]", *a, flush=True)


def parquet_files(root, sub):
    files = sorted((root / sub).glob("chunk-*/file-*.parquet"))
    assert files, f"no parquet files under {root / sub}"
    return files


def read_all(files):
    return pa.concat_tables([pq.read_table(f) for f in files]).combine_chunks()


def col_np(table, name):
    return table.column(name).to_numpy()


def list_col_np(table, name):
    """Flatten a (fixed-size) list column to a 2-D numpy array."""
    arr = table.column(name).combine_chunks()
    n = len(arr)
    return arr.flatten().to_numpy().reshape(n, -1)


def task_maps(root):
    t = pd.read_parquet(root / "meta" / "tasks.parquet")  # index = task text, column task_index
    idx = t["task_index"].to_numpy()
    assert (idx == np.arange(len(t))).all(), (
        f"{root}: tasks.parquet rows are not in task_index order. LeRobot reads tasks by row position."
    )
    texts = [str(s) for s in t.index]
    return texts, {s: i for i, s in enumerate(texts)}


def rewrite_in_place(files, replacements):
    """Rewrite each parquet file with some columns replaced.

    Keeps the original arrow schema (types, field order, huggingface metadata) and row-group layout.
    `replacements` maps column name -> full-length pa.Array/ChunkedArray covering all files in order.
    """
    off = 0
    for f in files:
        pf = pq.ParquetFile(f)
        schema = pf.schema_arrow
        tmp = f.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(str(tmp), schema=schema, compression="snappy", use_dictionary=True)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            n = t.num_rows
            for name, col in replacements.items():
                i = t.schema.get_field_index(name)
                assert i >= 0, f"column {name!r} not in {f}"
                field = t.schema.field(i)
                piece = col.slice(off, n)
                if isinstance(piece, pa.ChunkedArray):
                    piece = piece.combine_chunks()
                if piece.type != field.type:
                    piece = piece.cast(field.type)
                t = t.set_column(i, field, piece)
            writer.write_table(t)
            off += n
        writer.close()
        del pf
        tmp.replace(f)
    return off


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="where variants/ is written")
    p.add_argument("--src", default=None, help="where the two published copies are downloaded (default: OUT/src)")
    a = p.parse_args()
    out = Path(a.out)
    src_root = Path(a.src) if a.src else out / "src"
    src = {}
    for name in ("clean", "corrupted"):
        src[name] = Path(snapshot_download(f"{NS}/libero_object_{name}", repo_type="dataset", revision=REV,
                                           local_dir=src_root / name))
        log("downloaded", name, src[name])
    C, B = src["clean"], src["corrupted"]
    clean_info = json.load(open(C / "meta" / "info.json"))
    assert clean_info["codebase_version"] == "v3.0"

    # The published corrupted info.json is stale (453 / 66,832). Fix the local copy so every reader
    # (including verify_variants.py) sees the true counts. Data, episodes, and videos are complete.
    binfo_path = B / "meta" / "info.json"
    binfo = json.load(open(binfo_path))
    if (binfo["total_episodes"], binfo["total_frames"]) != (clean_info["total_episodes"], clean_info["total_frames"]):
        log(f"repairing corrupted info.json counts {binfo['total_episodes']}/{binfo['total_frames']} -> "
            f"{clean_info['total_episodes']}/{clean_info['total_frames']}")
        for k in ("total_episodes", "total_frames", "splits"):
            binfo[k] = clean_info[k]
        json.dump(binfo, open(binfo_path, "w"), indent=4)

    # ---- frame tables and row alignment checks
    cf, bf = parquet_files(C, "data"), parquet_files(B, "data")
    ctab, btab = read_all(cf), read_all(bf)
    assert ctab.schema.equals(btab.schema), "data parquet schemas differ between the two copies"
    assert ctab.num_rows == btab.num_rows == clean_info["total_frames"], (ctab.num_rows, btab.num_rows, clean_info["total_frames"])
    for col in ("index", "episode_index", "frame_index"):
        assert (col_np(ctab, col) == col_np(btab, col)).all(), f"row mismatch on {col}"
    assert np.allclose(col_np(ctab, "timestamp"), col_np(btab, "timestamp")), "timestamp mismatch"
    assert np.array_equal(list_col_np(ctab, "observation.state"), list_col_np(btab, "observation.state")), "state mismatch"
    c_act, b_act = list_col_np(ctab, "action"), list_col_np(btab, "action")
    assert np.array_equal(c_act[:, :6], b_act[:, :6]), "action dims 0..5 differ (only the gripper may differ)"
    n_flip = int((~np.isclose(c_act[:, 6], b_act[:, 6])).sum())
    log(f"frames aligned: {ctab.num_rows}, frames with a different gripper value: {n_flip}")

    # ---- task index remaps (each copy numbers tasks in first-seen order, so they differ)
    c_texts, c_t2i = task_maps(C)
    b_texts, b_t2i = task_maps(B)
    assert set(c_texts) == set(b_texts), "task sets differ"
    b_to_c = np.array([c_t2i[s] for s in b_texts], dtype=np.int64)  # corrupted index -> clean index
    c_to_b = np.array([b_t2i[s] for s in c_texts], dtype=np.int64)
    c_ti, b_ti = col_np(ctab, "task_index"), col_np(btab, "task_index")
    bad_task_in_clean_idx = pa.array(b_to_c[b_ti], type=pa.int64())
    clean_task_in_bad_idx = pa.array(c_to_b[c_ti], type=pa.int64())
    n_swap_frames = int((b_to_c[b_ti] != c_ti).sum())
    log(f"frames whose task label differs between copies: {n_swap_frames}")

    # ---- episode tables
    cef, bef = parquet_files(C, "meta/episodes"), parquet_files(B, "meta/episodes")
    ceps, beps = read_all(cef), read_all(bef)
    assert set(ceps.column_names) == set(beps.column_names), "episodes schemas differ"
    for col in ("episode_index", "length", "dataset_from_index", "dataset_to_index"):
        assert (col_np(ceps, col) == col_np(beps, col)).all(), f"episode mismatch on {col}"
    action_stat_cols = [c for c in ceps.column_names if c.startswith("stats/action/")]
    assert action_stat_cols, "no stats/action/* columns in meta/episodes"

    def write_variant(name, base, frame_cols, ep_cols, stats_action_from):
        base_tab, base_eps = (ctab, ceps) if base == C else (btab, beps)
        dst = out / "variants" / f"libero_object_{name}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(base, dst, ignore=shutil.ignore_patterns(".cache", ".git*", "answer_key", "*.lock"))
        n = rewrite_in_place(parquet_files(dst, "data"), frame_cols)
        assert n == ctab.num_rows, (n, ctab.num_rows)
        m = rewrite_in_place(parquet_files(dst, "meta/episodes"), ep_cols)
        assert m == ceps.num_rows, (m, ceps.num_rows)
        # global stats: action stats from the copy whose actions we kept
        stats = json.load(open(dst / "meta" / "stats.json"))
        stats["action"] = json.load(open(stats_action_from / "meta" / "stats.json"))["action"]
        json.dump(stats, open(dst / "meta" / "stats.json", "w"), indent=4)
        # counts from the clean copy (known good)
        info = json.load(open(dst / "meta" / "info.json"))
        for k in ("total_episodes", "total_frames", "splits"):
            info[k] = clean_info[k]
        json.dump(info, open(dst / "meta" / "info.json", "w"), indent=4)
        # quick self-check: schema unchanged, replaced columns present
        t = read_all(parquet_files(dst, "data"))
        assert t.schema.equals(base_tab.schema, check_metadata=True), f"{name}: data schema drifted"
        e = read_all(parquet_files(dst, "meta/episodes"))
        assert e.schema.equals(base_eps.schema, check_metadata=True), f"{name}: episodes schema drifted"
        untouched = [c for c in base_tab.column_names if c not in frame_cols]
        assert t.select(untouched).equals(base_tab.select(untouched)), f"{name}: an untouched data column changed"
        untouched = [c for c in base_eps.column_names if c not in ep_cols]
        assert e.select(untouched).equals(base_eps.select(untouched)), f"{name}: an untouched episodes column changed"
        log("wrote", dst)
        return t, e

    # instruction_only: clean base + corrupted task labels (expressed in clean task indices)
    t, e = write_variant(
        "instruction_only", C,
        frame_cols={"task_index": bad_task_in_clean_idx},
        ep_cols={"tasks": beps.column("tasks")},
        stats_action_from=C,
    )
    assert np.array_equal(col_np(t, "task_index"), b_to_c[b_ti])
    assert np.array_equal(list_col_np(t, "action"), c_act)
    # gripper_only: clean base + corrupted actions
    t, e = write_variant(
        "gripper_only", C,
        frame_cols={"action": btab.column("action")},
        ep_cols={c: beps.column(c) for c in action_stat_cols},
        stats_action_from=B,
    )
    assert np.array_equal(col_np(t, "task_index"), c_ti)
    assert np.array_equal(list_col_np(t, "action"), b_act)
    # camera_only: corrupted base (swapped videos) + clean task labels and actions
    t, e = write_variant(
        "camera_only", B,
        frame_cols={"task_index": clean_task_in_bad_idx, "action": ctab.column("action")},
        ep_cols={"tasks": ceps.column("tasks"), **{c: ceps.column(c) for c in action_stat_cols}},
        stats_action_from=C,
    )
    assert np.array_equal(col_np(t, "task_index"), c_to_b[c_ti])
    assert np.array_equal(list_col_np(t, "action"), c_act)

    shutil.copy(B / "answer_key" / "corruption_manifest.json", out / "variants" / "corruption_manifest.json")
    log("done")


if __name__ == "__main__":
    main()
