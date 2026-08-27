#!/usr/bin/env python3
"""Reassign smFISH spots to once-expanded nuclei linked across z-substacks.

This script replaces the separate assignment/export notebook steps used for
experiments 306, 307, 314, and 332.  It performs four operations consistently:

1. Discover every ``<sample>/565/results`` directory below configured roots.
2. Read the *original* 2-D nuclear label projections from ``results/labels``.
3. Link likely representations of the same nucleus in adjacent projections.
4. Expand each original label image exactly once, assign spots to linked cells,
   and write per-cell, per-animal, wide positive-cell, and QC CSV files.

The linking rule is deliberately conservative.  Labels in adjacent projections
are joined only when they are reciprocal best-overlap partners and satisfy at
least one configured overlap threshold.  Every link is exported for auditing.

Default experiment roots are the paths supplied for the current manuscript.
They can be replaced with repeatable ``--experiment NAME,GENE,ROOT`` arguments.

Typical use on the acquisition/analysis Mac::

    python reassign_smfish_3d.py \
      --output-root "/Volumes/Backup Plus/Experiment_results/reassignment_v2"

Run a synthetic end-to-end validation before processing real data::

    python reassign_smfish_3d.py --self-test

Required packages: numpy, pandas, scipy, scikit-image, tifffile.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

try:
    from skimage.segmentation import expand_labels as _skimage_expand_labels
except ImportError:  # portable fallback used by the validation environment
    _skimage_expand_labels = None

try:
    from tifffile import imread as _tifffile_imread
    from tifffile import imwrite as _tifffile_imwrite
except ImportError:  # portable fallback; the user's notebook env has tifffile
    _tifffile_imread = None
    _tifffile_imwrite = None


def imread(path: str | Path) -> np.ndarray:
    if _tifffile_imread is not None:
        return np.asarray(_tifffile_imread(path))
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install tifffile or Pillow to read label TIFF files") from exc
    return np.asarray(Image.open(path))


def imwrite(path: str | Path, array: np.ndarray, **kwargs: object) -> None:
    if _tifffile_imwrite is not None:
        _tifffile_imwrite(path, array, **kwargs)
        return
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install tifffile or Pillow to write TIFF files") from exc
    Image.fromarray(np.asarray(array)).save(path)


def expand_labels(labels: np.ndarray, distance: float) -> np.ndarray:
    """Use skimage when available; otherwise reproduce nearest-label expansion."""

    if distance <= 0:
        return labels.copy()
    if _skimage_expand_labels is not None:
        return _skimage_expand_labels(labels, distance=distance)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install scikit-image or scipy for label expansion") from exc
    distances, nearest = distance_transform_edt(
        labels == 0,
        return_distances=True,
        return_indices=True,
    )
    expanded = labels.copy()
    mask = (labels == 0) & (distances <= distance)
    nearest_labels = labels[tuple(nearest)]
    expanded[mask] = nearest_labels[mask]
    return expanded


SCRIPT_VERSION = "1.0.1"
LOGGER = logging.getLogger("smfish_reassignment")


DEFAULT_EXPERIMENTS = (
    (
        "306",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/306_analysis_results_alter",
    ),
    (
        "307",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/307_analysis_resutls/307_quantificatoin",
    ),
    (
        "314",
        "notum",
        "/Volumes/Backup Plus/Experiment_results/314_analysis_results",
    ),
    (
        "332",
        "notum",
        "/Volumes/Backup Plus/Experiment_results/332_notum_analysis",
    ),
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    gene: str
    root: Path


@dataclass(frozen=True)
class Parameters:
    projection_size: int
    expansion_distance: float
    min_smaller_overlap: float
    min_iou: float
    channel_folder: str
    save_assignment_maps: bool


@dataclass(frozen=True)
class SampleSpec:
    experiment: str
    gene: str
    experiment_root: Path
    condition: str
    animal: str
    sample_dir: Path
    results_dir: Path

    @property
    def sample_id(self) -> str:
        return f"{self.experiment}__{self.condition}__{self.animal}"


class UnionFind:
    """Small union-find implementation for (projection, local-label) nodes."""

    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}
        self.rank: dict[tuple[int, int], int] = {}

    def add(self, item: tuple[int, int]) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: tuple[int, int], second: tuple[int, int]) -> None:
        self.add(first)
        self.add(second)
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if rank_a == rank_b:
            self.rank[root_a] += 1


def natural_key(path_or_name: str | Path) -> list[object]:
    """Sort paths containing numbers in human/numeric order."""

    name = Path(path_or_name).name
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", name)]


def parse_experiment_argument(value: str) -> ExperimentSpec:
    """Parse NAME,GENE,ROOT while allowing commas inside ROOT after field two."""

    parts = value.split(",", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--experiment must be formatted as NAME,GENE,ROOT"
        )
    name, gene, root = (part.strip() for part in parts)
    return ExperimentSpec(name=name, gene=gene, root=Path(root))


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Pillow emits every TIFF tag at DEBUG level; keep pipeline diagnostics
    # readable when --verbose is enabled.
    logging.getLogger("PIL").setLevel(logging.WARNING)


def discover_samples(spec: ExperimentSpec, channel_folder: str) -> list[SampleSpec]:
    """Find only ``Experiment/<condition>/<animal>/<channel>/results`` samples.

    Experiment roots can also contain Control or other analysis folders with
    their own ``565/results`` paths.  Those are intentionally excluded.
    """

    if not spec.root.exists():
        LOGGER.warning("Experiment root does not exist; skipping: %s", spec.root)
        return []

    samples: list[SampleSpec] = []
    for results_dir in spec.root.rglob("results"):
        if not results_dir.is_dir() or results_dir.parent.name != channel_folder:
            continue
        sample_dir = results_dir.parent.parent
        condition_dir = sample_dir.parent
        experiment_dir = condition_dir.parent
        if experiment_dir.name.lower() != "experiment":
            LOGGER.debug(
                "Ignoring results directory outside an Experiment subtree: %s",
                results_dir,
            )
            continue
        samples.append(
            SampleSpec(
                experiment=spec.name,
                gene=spec.gene,
                experiment_root=spec.root,
                condition=condition_dir.name,
                animal=sample_dir.name,
                sample_dir=sample_dir,
                results_dir=results_dir,
            )
        )

    unique = {sample.results_dir.resolve(): sample for sample in samples}
    return sorted(
        unique.values(),
        key=lambda item: (
            natural_key(item.condition),
            natural_key(item.animal),
            str(item.results_dir),
        ),
    )


def locate_label_files(results_dir: Path) -> list[Path]:
    """Return original, unexpanded nuclear label images in projection order."""

    label_dir = results_dir / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Original label directory is missing: {label_dir}")

    preferred = sorted(label_dir.glob("Nucleus_Labels_*.tif*"), key=natural_key)
    if preferred:
        return preferred

    fallback = sorted(label_dir.glob("*.tif*"), key=natural_key)
    if not fallback:
        raise FileNotFoundError(f"No TIFF label files found in: {label_dir}")
    LOGGER.warning(
        "No Nucleus_Labels_*.tif files in %s; using all TIFF files", label_dir
    )
    return fallback


def locate_spot_file(results_dir: Path) -> Path:
    """Use background-filtered spots when present, matching the legacy pipeline."""

    candidates = (
        results_dir / "spots_post_decomposition_and_background_removed.npy",
        results_dir / "spots_post_decomposition.npy",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Neither spots_post_decomposition_and_background_removed.npy nor "
        f"spots_post_decomposition.npy exists in {results_dir}"
    )


def read_label(path: Path) -> np.ndarray:
    labels = np.asarray(imread(path))
    labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected one 2-D label image at {path}; got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.floor(labels))):
            raise ValueError(f"Label image contains non-integer values: {path}")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"Label image contains negative labels: {path}")
    return labels


def validate_label_sequence(label_files: Sequence[Path]) -> tuple[int, int]:
    if not label_files:
        raise ValueError("No label projections were provided")
    expected_shape: tuple[int, int] | None = None
    for path in label_files:
        labels = read_label(path)
        if expected_shape is None:
            expected_shape = labels.shape
        elif labels.shape != expected_shape:
            raise ValueError(
                f"Label image shape mismatch: {path} has {labels.shape}, "
                f"expected {expected_shape}"
            )
    assert expected_shape is not None
    return expected_shape


def label_areas(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)


def reciprocal_best_links(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    projection_a: int,
    min_smaller_overlap: float,
    min_iou: float,
) -> list[dict[str, float | int]]:
    """Find conservative one-to-one links between adjacent label projections.

    Candidate pairs must overlap in x-y.  Each accepted pair must be the
    highest-intersection partner for both labels and must meet either the
    intersection/smaller-area threshold or the IoU threshold.
    """

    if labels_a.shape != labels_b.shape:
        raise ValueError("Adjacent label projections have different shapes")

    overlap_mask = (labels_a > 0) & (labels_b > 0)
    if not np.any(overlap_mask):
        return []

    a_values = labels_a[overlap_mask].astype(np.int64, copy=False)
    b_values = labels_b[overlap_mask].astype(np.int64, copy=False)
    base = int(labels_b.max()) + 1
    encoded = a_values * base + b_values
    pair_codes, intersections = np.unique(encoded, return_counts=True)
    a_ids = pair_codes // base
    b_ids = pair_codes % base

    areas_a = label_areas(labels_a)
    areas_b = label_areas(labels_b)

    smaller_fraction = intersections / np.minimum(areas_a[a_ids], areas_b[b_ids])
    iou = intersections / (
        areas_a[a_ids] + areas_b[b_ids] - intersections
    )

    # Determine reciprocal best partners using intersection size.  A tie is
    # resolved deterministically by the smaller partner label ID.
    best_for_a: dict[int, tuple[int, int]] = {}
    best_for_b: dict[int, tuple[int, int]] = {}
    for index, (a_id, b_id, intersection) in enumerate(
        zip(a_ids, b_ids, intersections, strict=True)
    ):
        a_id_int = int(a_id)
        b_id_int = int(b_id)
        intersection_int = int(intersection)
        current_a = best_for_a.get(a_id_int)
        if current_a is None or (intersection_int, -b_id_int) > (
            current_a[1],
            -current_a[0],
        ):
            best_for_a[a_id_int] = (b_id_int, intersection_int)
        current_b = best_for_b.get(b_id_int)
        if current_b is None or (intersection_int, -a_id_int) > (
            current_b[1],
            -current_b[0],
        ):
            best_for_b[b_id_int] = (a_id_int, intersection_int)

    links: list[dict[str, float | int]] = []
    for index, (a_id, b_id, intersection, overlap_fraction, pair_iou) in enumerate(
        zip(
            a_ids,
            b_ids,
            intersections,
            smaller_fraction,
            iou,
            strict=True,
        )
    ):
        a_id_int = int(a_id)
        b_id_int = int(b_id)
        reciprocal = (
            best_for_a[a_id_int][0] == b_id_int
            and best_for_b[b_id_int][0] == a_id_int
        )
        threshold_pass = (
            float(overlap_fraction) >= min_smaller_overlap
            or float(pair_iou) >= min_iou
        )
        if not reciprocal or not threshold_pass:
            continue
        links.append(
            {
                "projection_a": projection_a,
                "label_a": a_id_int,
                "projection_b": projection_a + 1,
                "label_b": b_id_int,
                "intersection_pixels": int(intersection),
                "smaller_overlap_fraction": float(overlap_fraction),
                "iou": float(pair_iou),
            }
        )
    return links


def build_linked_cells(
    label_files: Sequence[Path],
    min_smaller_overlap: float,
    min_iou: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, int], int]]:
    """Create global cell IDs and exportable node/link tables."""

    union_find = UnionFind()
    node_rows: list[dict[str, int]] = []
    link_rows: list[dict[str, float | int]] = []

    previous: np.ndarray | None = None
    for projection, path in enumerate(label_files):
        labels = read_label(path)
        areas = label_areas(labels)
        for local_label in range(1, len(areas)):
            if areas[local_label] == 0:
                continue
            node = (projection, local_label)
            union_find.add(node)
            node_rows.append(
                {
                    "projection": projection,
                    "local_label": local_label,
                    "nuclear_area_pixels": int(areas[local_label]),
                }
            )

        if previous is not None:
            links = reciprocal_best_links(
                previous,
                labels,
                projection_a=projection - 1,
                min_smaller_overlap=min_smaller_overlap,
                min_iou=min_iou,
            )
            link_rows.extend(links)
            for link in links:
                union_find.union(
                    (int(link["projection_a"]), int(link["label_a"])),
                    (int(link["projection_b"]), int(link["label_b"])),
                )
        previous = labels

    roots = sorted({union_find.find(node) for node in union_find.parent})
    root_to_cell = {root: index + 1 for index, root in enumerate(roots)}
    node_to_cell = {
        node: root_to_cell[union_find.find(node)] for node in union_find.parent
    }

    node_df = pd.DataFrame(node_rows)
    node_df["global_cell_id"] = [
        node_to_cell[(int(row.projection), int(row.local_label))]
        for row in node_df.itertuples(index=False)
    ]
    link_df = pd.DataFrame(
        link_rows,
        columns=[
            "projection_a",
            "label_a",
            "projection_b",
            "label_b",
            "intersection_pixels",
            "smaller_overlap_fraction",
            "iou",
        ],
    )
    return node_df, link_df, node_to_cell


def load_spots(path: Path) -> np.ndarray:
    spots = np.asarray(np.load(path, allow_pickle=False))
    if spots.ndim != 2 or spots.shape[1] < 3:
        raise ValueError(
            f"Expected spot coordinates with shape (n, >=3) at {path}; "
            f"got {spots.shape}"
        )
    coordinates = spots[:, [0, -2, -1]].astype(np.float64, copy=False)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"Spot coordinates contain NaN/inf values: {path}")
    return coordinates


def parse_condition(condition: str) -> tuple[float | None, str]:
    normalized = condition.lower().replace("-", "_").replace(" ", "_")
    time_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)", normalized)
    time_hours = float(time_match.group(1)) if time_match else None

    if "held" in normalized or "hold_open" in normalized or "holdopen" in normalized:
        injury = "held_open"
    elif "amputation" in normalized or "amput" in normalized:
        injury = "amputation"
    elif "incision" in normalized or "incis" in normalized:
        injury = "incision"
    elif "dorsal" in normalized and "window" in normalized:
        injury = "dorsal_window"
    else:
        injury = "unparsed"
    return time_hours, injury


def assign_spots_to_linked_cells(
    sample: SampleSpec,
    label_files: Sequence[Path],
    node_df: pd.DataFrame,
    node_to_cell: dict[tuple[int, int], int],
    projection_size: int,
    expansion_distance: float,
    save_assignment_maps: bool,
    sample_output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Expand original labels once and count spots by linked global cell."""

    spot_file = locate_spot_file(sample.results_dir)
    spots = load_spots(spot_file)
    projection_indices = np.floor_divide(spots[:, 0].astype(np.int64), projection_size)
    y_coordinates = np.rint(spots[:, 1]).astype(np.int64)
    x_coordinates = np.rint(spots[:, 2]).astype(np.int64)

    global_cell_count = int(node_df["global_cell_id"].max()) if not node_df.empty else 0
    counts = np.zeros(global_cell_count + 1, dtype=np.int64)
    background_spots = 0
    invalid_spots = 0
    assigned_spots = 0

    map_dir = sample_output_dir / "assignment_maps"
    if save_assignment_maps:
        map_dir.mkdir(parents=True, exist_ok=True)

    for projection, label_path in enumerate(label_files):
        original_labels = read_label(label_path)
        expanded = expand_labels(original_labels, distance=expansion_distance)

        local_to_global = np.zeros(int(original_labels.max()) + 1, dtype=np.int64)
        projection_nodes = node_df[node_df["projection"] == projection]
        for row in projection_nodes.itertuples(index=False):
            local_to_global[int(row.local_label)] = int(row.global_cell_id)

        spot_mask = projection_indices == projection
        ys = y_coordinates[spot_mask]
        xs = x_coordinates[spot_mask]
        valid = (
            (ys >= 0)
            & (ys < expanded.shape[0])
            & (xs >= 0)
            & (xs < expanded.shape[1])
        )
        invalid_spots += int(np.sum(~valid))
        ys = ys[valid]
        xs = xs[valid]

        local_labels = expanded[ys, xs].astype(np.int64, copy=False)
        background_spots += int(np.sum(local_labels == 0))
        foreground = local_labels > 0
        if np.any(foreground):
            global_ids = local_to_global[local_labels[foreground]]
            if np.any(global_ids == 0):
                raise RuntimeError(
                    f"An expanded label lacked a global mapping in {sample.sample_id}, "
                    f"projection {projection}"
                )
            counts += np.bincount(global_ids, minlength=len(counts))
            assigned_spots += int(len(global_ids))

    out_of_projection_spots = int(
        np.sum((projection_indices < 0) | (projection_indices >= len(label_files)))
    )
    invalid_spots += out_of_projection_spots

    # Write maps only after all projections have been assigned, so every
    # projection displays the final summed count for cells linked across z.
    if save_assignment_maps:
        for projection, label_path in enumerate(label_files):
            original_labels = read_label(label_path)
            expanded = expand_labels(original_labels, distance=expansion_distance)
            local_to_global = np.zeros(int(original_labels.max()) + 1, dtype=np.int64)
            projection_nodes = node_df[node_df["projection"] == projection]
            for row in projection_nodes.itertuples(index=False):
                local_to_global[int(row.local_label)] = int(row.global_cell_id)
            expanded_global = local_to_global[expanded]
            # Store transcript counts, not cell IDs. uint32 prevents the uint8
            # overflow present in the legacy assignment-map implementation.
            count_map = counts[expanded_global].astype(np.uint32, copy=False)
            imwrite(
                map_dir / f"assignment_count_map_{projection:03d}.tif",
                count_map,
                photometric="minisblack",
            )

    component_summary = (
        node_df.groupby("global_cell_id", as_index=False)
        .agg(
            first_projection=("projection", "min"),
            last_projection=("projection", "max"),
            n_projections=("projection", "nunique"),
            summed_nuclear_area_pixels=("nuclear_area_pixels", "sum"),
        )
        .sort_values("global_cell_id")
    )
    component_summary["transcript_count"] = counts[1:]
    component_summary["responding"] = component_summary["transcript_count"] > 0
    component_summary.insert(0, "sample_id", sample.sample_id)
    component_summary.insert(0, "animal", sample.animal)
    component_summary.insert(0, "condition", sample.condition)
    component_summary.insert(0, "gene", sample.gene)
    component_summary.insert(0, "experiment", sample.experiment)
    time_hours, injury = parse_condition(sample.condition)
    component_summary.insert(3, "time_hours", time_hours)
    component_summary.insert(4, "injury", injury)

    total_spots = int(len(spots))
    accounted = assigned_spots + background_spots + invalid_spots
    qc = {
        "experiment": sample.experiment,
        "gene": sample.gene,
        "condition": sample.condition,
        "time_hours": time_hours,
        "injury": injury,
        "animal": sample.animal,
        "sample_id": sample.sample_id,
        "sample_path": str(sample.sample_dir),
        "spot_file": str(spot_file),
        "label_projections": len(label_files),
        "total_spots": total_spots,
        "assigned_spots": assigned_spots,
        "background_spots": background_spots,
        "invalid_spots": invalid_spots,
        "accounted_spots": accounted,
        "spot_accounting_matches": accounted == total_spots,
        "total_linked_cells": int(len(component_summary)),
        "responding_cells": int(component_summary["responding"].sum()),
        "responding_fraction": float(component_summary["responding"].mean())
        if len(component_summary)
        else math.nan,
        "maximum_transcript_count": int(component_summary["transcript_count"].max())
        if len(component_summary)
        else 0,
        "cells_spanning_multiple_projections": int(
            np.sum(component_summary["n_projections"] > 1)
        ),
        "cells_spanning_more_than_three_projections": int(
            np.sum(component_summary["n_projections"] > 3)
        ),
    }
    return component_summary, qc


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_sample_outputs(
    sample_output_dir: Path,
    cell_df: pd.DataFrame,
    node_df: pd.DataFrame,
    link_df: pd.DataFrame,
    qc: dict[str, object],
) -> None:
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(cell_df, sample_output_dir / "cell_counts.csv")
    atomic_write_csv(node_df, sample_output_dir / "projection_label_to_cell.csv")
    atomic_write_csv(link_df, sample_output_dir / "adjacent_projection_links.csv")
    with (sample_output_dir / "sample_qc.json").open("w", encoding="utf-8") as stream:
        json.dump(qc, stream, indent=2, allow_nan=True)


def animal_summary_from_cells(cell_df: pd.DataFrame) -> dict[str, object]:
    first = cell_df.iloc[0]
    positive = cell_df.loc[cell_df["transcript_count"] > 0, "transcript_count"].to_numpy()
    return {
        "experiment": first["experiment"],
        "gene": first["gene"],
        "condition": first["condition"],
        "time_hours": first["time_hours"],
        "injury": first["injury"],
        "animal": first["animal"],
        "sample_id": first["sample_id"],
        "total_cells": int(len(cell_df)),
        "responding_cells": int(len(positive)),
        "responding_fraction": float(len(positive) / len(cell_df)) if len(cell_df) else math.nan,
        "total_assigned_transcripts": int(np.sum(positive)) if len(positive) else 0,
        "mean_count_all_cells": float(cell_df["transcript_count"].mean()),
        "mean_log1p_all_cells": float(np.log1p(cell_df["transcript_count"]).mean()),
        "mean_count_responding_cells": float(np.mean(positive)) if len(positive) else math.nan,
        "median_count_responding_cells": float(np.median(positive)) if len(positive) else math.nan,
        "mean_log1p_responding_cells": float(np.log1p(positive).mean())
        if len(positive)
        else math.nan,
        "fraction_responding_cells_ge_5": float(np.mean(positive >= 5))
        if len(positive)
        else math.nan,
        "maximum_count": int(np.max(positive)) if len(positive) else 0,
    }


def process_sample(
    sample: SampleSpec,
    parameters: Parameters,
    output_root: Path,
    overwrite: bool,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    sample_output_dir = output_root / sample.experiment / sample.condition / sample.animal
    completion_marker = sample_output_dir / "COMPLETE.json"
    if completion_marker.exists() and not overwrite:
        LOGGER.info("Skipping completed sample: %s", sample.sample_id)
        cell_df = pd.read_csv(sample_output_dir / "cell_counts.csv")
        node_df = pd.read_csv(sample_output_dir / "projection_label_to_cell.csv")
        link_df = pd.read_csv(sample_output_dir / "adjacent_projection_links.csv")
        with (sample_output_dir / "sample_qc.json").open(encoding="utf-8") as stream:
            qc = json.load(stream)
        return cell_df, qc, node_df, link_df

    if sample_output_dir.exists() and overwrite:
        shutil.rmtree(sample_output_dir)
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Processing %s", sample.sample_id)
    label_files = locate_label_files(sample.results_dir)
    validate_label_sequence(label_files)
    node_df, link_df, node_to_cell = build_linked_cells(
        label_files,
        min_smaller_overlap=parameters.min_smaller_overlap,
        min_iou=parameters.min_iou,
    )
    cell_df, qc = assign_spots_to_linked_cells(
        sample,
        label_files,
        node_df,
        node_to_cell,
        projection_size=parameters.projection_size,
        expansion_distance=parameters.expansion_distance,
        save_assignment_maps=parameters.save_assignment_maps,
        sample_output_dir=sample_output_dir,
    )
    write_sample_outputs(sample_output_dir, cell_df, node_df, link_df, qc)
    with completion_marker.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "script_version": SCRIPT_VERSION,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": asdict(parameters),
            },
            stream,
            indent=2,
        )
    return cell_df, qc, node_df, link_df


def aggregate_outputs(
    cell_frames: Sequence[pd.DataFrame],
    qc_rows: Sequence[dict[str, object]],
    output_root: Path,
) -> None:
    if not cell_frames:
        raise RuntimeError("No samples were successfully processed")

    all_cells = pd.concat(cell_frames, ignore_index=True)
    atomic_write_csv(all_cells, output_root / "all_cells_long.csv")

    positive_series: dict[str, pd.Series] = {}
    animal_rows: list[dict[str, object]] = []
    for sample_id, sample_cells in all_cells.groupby("sample_id", sort=True):
        positive = sample_cells.loc[
            sample_cells["transcript_count"] > 0, "transcript_count"
        ].reset_index(drop=True)
        positive_series[str(sample_id)] = positive
        animal_rows.append(animal_summary_from_cells(sample_cells))

    positive_wide = pd.DataFrame(positive_series)
    atomic_write_csv(positive_wide, output_root / "positive_cells_wide.csv")
    atomic_write_csv(pd.DataFrame(animal_rows), output_root / "animal_summary.csv")
    atomic_write_csv(pd.DataFrame(qc_rows), output_root / "sample_qc.csv")


def run_pipeline(
    experiments: Sequence[ExperimentSpec],
    parameters: Parameters,
    output_root: Path,
    overwrite: bool,
    fail_fast: bool,
    sample_regex: str | None,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "run_configuration.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "script_version": SCRIPT_VERSION,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiments": [
                    {"name": item.name, "gene": item.gene, "root": str(item.root)}
                    for item in experiments
                ],
                "parameters": asdict(parameters),
            },
            stream,
            indent=2,
        )

    pattern = re.compile(sample_regex) if sample_regex else None
    all_samples: list[SampleSpec] = []
    for experiment in experiments:
        discovered = discover_samples(experiment, parameters.channel_folder)
        LOGGER.info("Discovered %d samples for experiment %s", len(discovered), experiment.name)
        all_samples.extend(discovered)
    if pattern:
        all_samples = [sample for sample in all_samples if pattern.search(sample.sample_id)]
        LOGGER.info("%d samples remain after --sample-regex", len(all_samples))
    if not all_samples:
        LOGGER.error("No samples discovered. Check roots and --channel-folder.")
        return 2

    cell_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for sample in all_samples:
        try:
            cell_df, qc, _, _ = process_sample(
                sample,
                parameters=parameters,
                output_root=output_root,
                overwrite=overwrite,
            )
            cell_frames.append(cell_df)
            qc_rows.append(qc)
        except Exception as exc:  # keep the batch running unless requested otherwise
            LOGGER.exception("Failed sample %s", sample.sample_id)
            errors.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_path": str(sample.sample_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if fail_fast:
                break

    if errors:
        atomic_write_csv(pd.DataFrame(errors), output_root / "processing_errors.csv")
    if cell_frames:
        aggregate_outputs(cell_frames, qc_rows, output_root)
    LOGGER.info(
        "Finished: %d successful samples, %d failed samples. Output: %s",
        len(cell_frames),
        len(errors),
        output_root,
    )
    return 1 if errors else 0


def create_synthetic_experiment(base: Path) -> ExperimentSpec:
    """Create one minimal sample with a known linked-cell assignment."""

    root = base / "synthetic_experiment"
    results = root / "Experiment" / "6h_Amputation" / "Image1" / "565" / "results"
    labels_dir = results / "labels"
    labels_dir.mkdir(parents=True)

    # This decoy path must not be discovered: only the Experiment subtree is
    # valid input, even when another folder contains a 565/results directory.
    (root / "Control" / "6h_Amputation" / "Image1" / "565" / "results").mkdir(
        parents=True
    )

    label_0 = np.zeros((40, 50), dtype=np.uint16)
    label_0[10:18, 10:18] = 1
    label_0[25:32, 30:38] = 2
    label_1 = np.zeros_like(label_0)
    label_1[11:19, 11:19] = 1  # same nucleus as projection 0, label 1
    label_1[5:11, 35:42] = 2   # new nucleus
    label_2 = np.zeros_like(label_0)
    label_2[6:12, 36:43] = 1   # same nucleus as projection 1, label 2

    for index, labels in enumerate((label_0, label_1, label_2)):
        imwrite(labels_dir / f"Nucleus_Labels_{index:03d}.tif", labels)

    # z,y,x; projection size 10. The first two spots should join the same
    # global cell. One spot is near/assigned after a single expansion, one is
    # background, and one is in the second linked cell.
    spots = np.array(
        [
            [2, 14, 14],
            [12, 15, 15],
            [8, 20, 20],
            [15, 0, 0],
            [25, 8, 39],
        ],
        dtype=np.int64,
    )
    np.save(results / "spots_post_decomposition_and_background_removed.npy", spots)
    return ExperimentSpec(name="test", gene="test_gene", root=root)


def run_self_test() -> int:
    configure_logging(verbose=True)
    with tempfile.TemporaryDirectory(prefix="smfish_reassignment_test_") as temporary:
        base = Path(temporary)
        experiment = create_synthetic_experiment(base)
        output = base / "output"
        parameters = Parameters(
            projection_size=10,
            expansion_distance=3,
            min_smaller_overlap=0.30,
            min_iou=0.10,
            channel_folder="565",
            save_assignment_maps=True,
        )
        status = run_pipeline(
            [experiment],
            parameters=parameters,
            output_root=output,
            overwrite=True,
            fail_fast=True,
            sample_regex=None,
        )
        if status != 0:
            raise AssertionError("Synthetic pipeline returned a nonzero status")
        cells = pd.read_csv(output / "all_cells_long.csv")
        qc = pd.read_csv(output / "sample_qc.csv").iloc[0]
        positive_counts = sorted(cells.loc[cells["transcript_count"] > 0, "transcript_count"])
        if positive_counts != [1, 2]:
            raise AssertionError(f"Unexpected synthetic positive counts: {positive_counts}")
        if int(qc["total_spots"]) != 5:
            raise AssertionError("Synthetic total spot count is incorrect")
        if not bool(qc["spot_accounting_matches"]):
            raise AssertionError("Synthetic spot accounting failed")
        print("SELF-TEST PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment_argument,
        help=(
            "Override defaults with NAME,GENE,ROOT. Repeat for multiple roots. "
            "If omitted, experiments 306/307/314/332 are used."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/Backup Plus/Experiment_results/smfish_reassignment_v2"
        ),
    )
    parser.add_argument("--projection-size", type=int, default=10)
    parser.add_argument("--expansion-distance", type=float, default=20.0)
    parser.add_argument("--min-smaller-overlap", type=float, default=0.35)
    parser.add_argument("--min-iou", type=float, default=0.15)
    parser.add_argument("--channel-folder", default="565")
    parser.add_argument(
        "--save-assignment-maps",
        action="store_true",
        help="Write uint32 per-projection transcript-count maps (large output).",
    )
    parser.add_argument(
        "--sample-regex",
        help="Process only sample IDs matching this regular expression.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()

    if args.projection_size <= 0:
        parser.error("--projection-size must be positive")
    if args.expansion_distance < 0:
        parser.error("--expansion-distance cannot be negative")
    for value, name in (
        (args.min_smaller_overlap, "--min-smaller-overlap"),
        (args.min_iou, "--min-iou"),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1")

    configure_logging(args.verbose)
    experiments = args.experiment or [
        ExperimentSpec(name=name, gene=gene, root=Path(root))
        for name, gene, root in DEFAULT_EXPERIMENTS
    ]
    parameters = Parameters(
        projection_size=args.projection_size,
        expansion_distance=args.expansion_distance,
        min_smaller_overlap=args.min_smaller_overlap,
        min_iou=args.min_iou,
        channel_folder=args.channel_folder,
        save_assignment_maps=args.save_assignment_maps,
    )
    return run_pipeline(
        experiments=experiments,
        parameters=parameters,
        output_root=args.output_root,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        sample_regex=args.sample_regex,
    )


if __name__ == "__main__":
    raise SystemExit(main())
