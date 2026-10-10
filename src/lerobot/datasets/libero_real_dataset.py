"""Small real-row LIBERO dataset smoke helpers."""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LIBERO_IMAGE_KEY = "observation.images.image"
LIBERO_WRIST_IMAGE_KEY = "observation.images.wrist_image"
LIBERO_STATE_KEY = "observation.state"
LIBERO_ACTION_KEY = "action"
LIBERO_REQUIRED_FEATURES = {
    LIBERO_IMAGE_KEY: [256, 256, 3],
    LIBERO_WRIST_IMAGE_KEY: [256, 256, 3],
    LIBERO_STATE_KEY: [8],
    LIBERO_ACTION_KEY: [7],
}


@dataclass(frozen=True)
class RealLiberoSample:
    dataset_root: Path
    parquet_path: Path
    num_rows: int
    rows: list[dict[str, Any]]
    image_format: str
    wrist_image_format: str
    state_shape: tuple[int, int]
    action_shape: tuple[int, int]
    parquet_paths: tuple[Path, ...] = ()


def load_libero_info(dataset_root: str | Path) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LIBERO info.json not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_libero_task_index_to_text(dataset_root: str | Path) -> dict[int, str]:
    dataset_root = Path(dataset_root)
    task_path = dataset_root / "meta" / "tasks.parquet"
    if not task_path.is_file():
        raise FileNotFoundError(f"LIBERO tasks.parquet not found: {task_path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NotImplementedError("Reading LIBERO task text mapping requires optional pyarrow.") from exc

    table = pq.read_table(task_path)
    payload = table.to_pydict()
    return {int(index): str(text) for index, text in zip(payload["task_index"], payload["task"], strict=True)}


def validate_libero_info_schema(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features", {})
    observed = {}
    for key, expected_shape in LIBERO_REQUIRED_FEATURES.items():
        if key not in features:
            raise ValueError(f"LIBERO info.json missing feature: {key}")
        shape = list(features[key].get("shape", []))
        if shape != expected_shape:
            raise ValueError(f"LIBERO feature {key} shape mismatch: expected {expected_shape}, got {shape}")
        observed[key] = {"shape": shape, "dtype": features[key].get("dtype")}
    return {
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "features": observed,
    }


def select_smallest_libero_parquet(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    data_dir = dataset_root / "data" / "chunk-000"
    candidates = sorted(
        (path for path in data_dir.glob("*.parquet") if path.is_file()),
        key=lambda path: (path.stat().st_size, path.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return candidates[0]


def list_libero_parquet_files(dataset_root: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root)
    data_dir = dataset_root / "data"
    candidates = sorted(path for path in data_dir.glob("chunk-*/*.parquet") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return candidates


def count_libero_parquet_rows(parquet_path: str | Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NotImplementedError("Counting real LIBERO parquet rows requires optional pyarrow.") from exc

    return int(pq.ParquetFile(Path(parquet_path)).metadata.num_rows)


def count_libero_task_rows(dataset_root: str | Path, task_index: int) -> int:
    """Count task anchors without decoding image payloads."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NotImplementedError("Counting LIBERO task rows requires optional pyarrow.") from exc

    count = 0
    for parquet_path in list_libero_parquet_files(dataset_root):
        task_column = pq.read_table(parquet_path, columns=["task_index"])["task_index"].to_numpy()
        count += int(np.count_nonzero(task_column == int(task_index)))
    return count


def _pa_table_to_rows(table: Any) -> list[dict[str, Any]]:
    return [{name: table[name][index].as_py() for name in table.column_names} for index in range(table.num_rows)]


def read_libero_parquet_rows(parquet_path: str | Path, *, num_rows: int = 16) -> list[dict[str, Any]]:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NotImplementedError("Reading real LIBERO parquet rows requires optional pyarrow.") from exc

    parquet_path = Path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)
    table = parquet_file.read().slice(0, num_rows)
    return _pa_table_to_rows(table)


def iter_real_libero_training_row_batches(
    dataset_root: str | Path,
    *,
    num_rows: int,
    rows_per_batch: int = 256,
    decode_main_image: bool = True,
    decode_wrist_image: bool = True,
    task_index_filter: int | None = None,
    extra_columns: tuple[str, ...] = (),
):
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")
    if rows_per_batch <= 0:
        raise ValueError("rows_per_batch must be positive.")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NotImplementedError("Reading real LIBERO parquet row batches requires optional pyarrow.") from exc

    task_index_to_text = load_libero_task_index_to_text(dataset_root)
    columns = [LIBERO_STATE_KEY, LIBERO_ACTION_KEY, "task_index", "episode_index", "frame_index"]
    for column in extra_columns:
        if column not in columns:
            columns.append(column)
    if decode_main_image:
        columns.append(LIBERO_IMAGE_KEY)
    if decode_wrist_image:
        columns.append(LIBERO_WRIST_IMAGE_KEY)
    yielded = 0
    for parquet_path in list_libero_parquet_files(dataset_root):
        if yielded >= num_rows:
            break
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(batch_size=rows_per_batch, columns=columns):
            remaining = num_rows - yielded
            if remaining <= 0:
                break
            if task_index_filter is None:
                limited_batch = record_batch.slice(0, min(remaining, record_batch.num_rows))
            else:
                limited_batch = record_batch
            raw_rows = _pa_table_to_rows(limited_batch)
            if task_index_filter is not None:
                raw_rows = [row for row in raw_rows if int(row.get("task_index", -1)) == int(task_index_filter)]
                if not raw_rows:
                    continue
                raw_rows = raw_rows[:remaining]
            training_rows, row_summary = real_libero_rows_to_training_rows(
                raw_rows,
                decode_main_image=decode_main_image,
                decode_wrist_image=decode_wrist_image,
                task_index_to_text=task_index_to_text,
                passthrough_fields=extra_columns,
            )
            yielded += len(training_rows)
            yield parquet_path, training_rows, row_summary


def _paeth_predictor_byte(left: int, up: int, up_left: int) -> int:
    p = left + up - up_left
    pa = abs(p - left)
    pb = abs(p - up)
    pc = abs(p - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def _decode_png_rgb_opencv(image_bytes: bytes) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None
    if decoded.ndim == 2:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    elif decoded.shape[-1] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    elif decoded.shape[-1] == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    else:
        return None
    return np.asarray(decoded, dtype=np.uint8)


def _decode_png_rgb_pillow(image_bytes: bytes) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _decode_png_rgb_pure_python(image_bytes: bytes) -> np.ndarray:
    """Decode the simple 8-bit RGB/RGBA PNG rows used by LeRobot image parquet columns."""

    signature = b"\x89PNG\r\n\x1a\n"
    if not image_bytes.startswith(signature):
        raise ValueError("Expected PNG bytes.")
    offset = len(signature)
    width = height = bit_depth = color_type = interlace = None
    compressed_chunks: list[bytes] = []
    while offset < len(image_bytes):
        length = struct.unpack(">I", image_bytes[offset : offset + 4])[0]
        chunk_type = image_bytes[offset + 4 : offset + 8]
        data = image_bytes[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(
                ">IIBBBBB", data
            )
        elif chunk_type == b"IDAT":
            compressed_chunks.append(data)
        elif chunk_type == b"IEND":
            break
    if None in (width, height, bit_depth, color_type, interlace):
        raise ValueError("PNG is missing IHDR.")
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(
            f"Unsupported PNG encoding: bit_depth={bit_depth}, color_type={color_type}, interlace={interlace}"
        )
    channels = 3 if color_type == 2 else 4
    row_bytes = int(width) * channels
    raw = zlib.decompress(b"".join(compressed_chunks))
    expected = int(height) * (row_bytes + 1)
    if len(raw) != expected:
        raise ValueError(f"PNG payload size mismatch: expected {expected}, got {len(raw)}")

    recon = np.zeros((int(height), row_bytes), dtype=np.uint8)
    source = np.frombuffer(raw, dtype=np.uint8)
    stride = row_bytes + 1
    for y in range(int(height)):
        filter_type = int(source[y * stride])
        scan = source[y * stride + 1 : (y + 1) * stride]
        prior = recon[y - 1] if y > 0 else np.zeros((row_bytes,), dtype=np.uint8)
        if filter_type == 0:
            recon[y] = scan
            continue
        for x in range(row_bytes):
            left = int(recon[y, x - channels]) if x >= channels else 0
            up = int(prior[x])
            up_left = int(prior[x - channels]) if x >= channels else 0
            value = int(scan[x])
            if filter_type == 1:
                value += left
            elif filter_type == 2:
                value += up
            elif filter_type == 3:
                value += (left + up) // 2
            elif filter_type == 4:
                value += _paeth_predictor_byte(left, up, up_left)
            else:
                raise ValueError(f"Unsupported PNG filter type: {filter_type}")
            recon[y, x] = value & 0xFF
    image = recon.reshape((int(height), int(width), channels))
    return image[:, :, :3].copy()


def decode_png_rgb(image_bytes: bytes) -> np.ndarray:
    decoded = _decode_png_rgb_opencv(image_bytes)
    if decoded is None:
        decoded = _decode_png_rgb_pillow(image_bytes)
    if decoded is None:
        decoded = _decode_png_rgb_pure_python(image_bytes)
    if decoded.shape != (256, 256, 3):
        raise ValueError(f"Decoded PNG shape mismatch: {decoded.shape}")
    return np.ascontiguousarray(decoded, dtype=np.uint8)


def _decode_image_cell(cell: Any) -> tuple[np.ndarray, str]:
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return decode_png_rgb(cell["bytes"]), "dict(bytes=png,path=string)"
        if cell.get("path"):
            raise NotImplementedError("Path-only LIBERO image cells are not supported by this smoke helper.")
    if isinstance(cell, (bytes, bytearray)):
        return decode_png_rgb(bytes(cell)), "png_bytes"
    array = np.asarray(cell)
    if array.shape == (256, 256, 3):
        return array.astype(np.uint8), "array"
    raise ValueError(f"Unsupported LIBERO image cell format: {type(cell).__name__}")


def real_libero_rows_to_training_rows(
    rows: list[dict[str, Any]],
    *,
    decode_main_image: bool = True,
    decode_wrist_image: bool = True,
    task_index_to_text: dict[int, str] | None = None,
    passthrough_fields: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_rows: list[dict[str, Any]] = []
    image_format = wrist_format = ""
    for index, row in enumerate(rows):
        if decode_main_image:
            image, image_format = _decode_image_cell(row[LIBERO_IMAGE_KEY])
        else:
            image = None
            image_format = "skipped_for_training"
        if decode_wrist_image:
            wrist_image, wrist_format = _decode_image_cell(row[LIBERO_WRIST_IMAGE_KEY])
        else:
            wrist_image = None
            wrist_format = "skipped_for_training"
        state = np.asarray(row[LIBERO_STATE_KEY], dtype=np.float32)
        action = np.asarray(row[LIBERO_ACTION_KEY], dtype=np.float32)
        if decode_main_image and image is not None and image.shape != (256, 256, 3):
            raise ValueError(f"image row {index} shape mismatch: {image.shape}")
        if decode_wrist_image and wrist_image is not None and wrist_image.shape != (256, 256, 3):
            raise ValueError(f"wrist_image row {index} shape mismatch: {wrist_image.shape}")
        if state.shape != (8,):
            raise ValueError(f"state row {index} shape mismatch: {state.shape}")
        if action.shape != (7,):
            raise ValueError(f"action row {index} shape mismatch: {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"state/action row {index} contains non-finite values.")
        target = np.float32(action[-1])
        task_index = int(row.get("task_index", 0))
        task_text = (task_index_to_text or {}).get(task_index)
        if task_text is None:
            task_text = str(row.get("task", "")).strip()
        if not task_text:
            raise ValueError(f"LIBERO row {index} with task_index={task_index} has no task text.")
        if task_text.startswith("libero task_index"):
            raise ValueError(f"LIBERO row {index} uses fake task text: {task_text}")
        training_row = {
                "observation.state": state,
                "action": action,
                "value_targets": target,
                "loss_weights": np.float32(1.0),
                "task_index": np.asarray(task_index, dtype=np.int64),
                "episode_index": np.asarray(row.get("episode_index", index), dtype=np.int64),
                "frame_index": np.asarray(row.get("frame_index", index), dtype=np.int64),
                "task": task_text,
            }
        if decode_main_image:
            training_row[LIBERO_IMAGE_KEY] = image
        if decode_wrist_image:
            training_row["observation.images.wrist_image"] = wrist_image
        for field_name in passthrough_fields:
            if field_name not in row:
                raise KeyError(f"Requested passthrough field '{field_name}' is missing from row {index}.")
            training_row[field_name] = row[field_name]
        training_rows.append(training_row)
    states = np.stack([row["observation.state"] for row in training_rows])
    actions = np.stack([row["action"] for row in training_rows])
    summary = {
        "image_format": image_format,
        "wrist_image_format": wrist_format,
        "image_shape": list(training_rows[0][LIBERO_IMAGE_KEY].shape) if decode_main_image else None,
        "main_image_decoded": bool(decode_main_image),
        "wrist_image_shape": (
            list(training_rows[0]["observation.images.wrist_image"].shape)
            if decode_wrist_image
            else None
        ),
        "wrist_image_decoded": bool(decode_wrist_image),
        "state_shape": list(states.shape),
        "action_shape": list(actions.shape),
        "state_finite": bool(np.isfinite(states).all()),
        "action_finite": bool(np.isfinite(actions).all()),
        "fake_task_text_used": False,
        "task_index_to_text_mapping_available": bool(task_index_to_text),
        "first_task_texts": [str(row["task"]) for row in training_rows[: min(4, len(training_rows))]],
    }
    return training_rows, summary


def load_real_libero_sample(dataset_root: str | Path, *, num_rows: int = 16) -> tuple[RealLiberoSample, dict[str, Any]]:
    dataset_root = Path(dataset_root)
    info_summary = validate_libero_info_schema(load_libero_info(dataset_root))
    smallest_parquet_path = select_smallest_libero_parquet(dataset_root)
    smallest_rows = count_libero_parquet_rows(smallest_parquet_path)
    parquet_paths = [smallest_parquet_path]
    if num_rows <= smallest_rows:
        raw_rows = read_libero_parquet_rows(smallest_parquet_path, num_rows=num_rows)
    else:
        raw_rows = []
        parquet_paths = []
        remaining = num_rows
        for parquet_path in list_libero_parquet_files(dataset_root):
            if remaining <= 0:
                break
            rows = read_libero_parquet_rows(parquet_path, num_rows=remaining)
            if not rows:
                continue
            raw_rows.extend(rows)
            parquet_paths.append(parquet_path)
            remaining -= len(rows)
    if not raw_rows:
        raise ValueError(f"No LIBERO rows could be read from {dataset_root}")
    training_rows, row_summary = real_libero_rows_to_training_rows(
        raw_rows,
        task_index_to_text=load_libero_task_index_to_text(dataset_root),
    )
    sample = RealLiberoSample(
        dataset_root=dataset_root,
        parquet_path=parquet_paths[0],
        num_rows=len(training_rows),
        rows=training_rows,
        image_format=row_summary["image_format"],
        wrist_image_format=row_summary["wrist_image_format"],
        state_shape=tuple(row_summary["state_shape"]),
        action_shape=tuple(row_summary["action_shape"]),
        parquet_paths=tuple(parquet_paths),
    )
    return sample, {
        "info": info_summary,
        "rows": row_summary,
        "source_parquet_files": [str(path) for path in parquet_paths],
        "source_parquet_file_count": len(parquet_paths),
    }


def export_real_libero_pistar06_features(
    dataset_root: str | Path,
    output_path: str | Path,
    metadata_output_path: str | Path,
    *,
    num_rows: int = 16,
    image_feature_dim: int = 8,
    language_feature_dim: int = 8,
    seed: int = 55,
) -> dict[str, Any]:
    """Export deterministic Pistar06 fixtures when that optional stack exists.

    The real LIBERO reader above has no Pistar06/JAX dependency.  Keeping this
    compatibility entry point lazy allows old scripts to import the module in a
    minimal PyTorch environment while still producing the same clear error when
    the optional feature-export modules are unavailable.
    """
    try:
        from lerobot.values.pistar06.feature_artifact import compute_file_sha256
        from lerobot.values.pistar06.feature_export import (
            Pistar06FeatureExportConfig,
            export_rows_to_precomputed_feature_batch,
        )
        from lerobot.values.pistar06.feature_provider import (
            FeatureProviderMetadata,
            FixtureFeatureProvider,
            save_feature_provider_metadata,
        )
        from lerobot.values.pistar06.training import save_precomputed_feature_fixture
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pistar06 feature export is optional; install/enable the local Pistar06 modules "
            "before calling export_real_libero_pistar06_features."
        ) from exc

    sample, summary = load_real_libero_sample(dataset_root, num_rows=num_rows)
    output_path = Path(output_path)
    metadata_output_path = Path(metadata_output_path)
    provider = FixtureFeatureProvider(
        image_feature_dim=image_feature_dim,
        language_feature_dim=language_feature_dim,
        seed=seed,
        notes="Deterministic feature encoder from real LIBERO rows; heavy SmolVLM2 forward is disabled.",
    )
    config = Pistar06FeatureExportConfig(
        image_feature_dim=image_feature_dim,
        language_feature_dim=language_feature_dim,
        seed=seed,
        write_row_identity_hashes=True,
        require_row_identity_hashes=True,
    )
    batch = export_rows_to_precomputed_feature_batch(sample.rows, config, provider=provider)
    feature_path = save_precomputed_feature_fixture(output_path, batch)
    feature_sha256 = compute_file_sha256(feature_path)
    metadata = FeatureProviderMetadata(
        **{
            **provider.metadata.to_dict(),
            "feature_sha256": feature_sha256,
            "extra": {
                "source_dataset": str(sample.dataset_root),
                "source_parquet_file": str(sample.parquet_path),
                "source_rows_real": True,
                "source_row_count": int(sample.num_rows),
                "feature_encoder": "deterministic_fixture_from_real_rows",
                "smolvlm2_heavy_forward": False,
                "image_format": sample.image_format,
                "wrist_image_format": sample.wrist_image_format,
            },
        }
    )
    save_feature_provider_metadata(metadata, metadata_output_path)
    return {
        "features_path": str(feature_path),
        "metadata_path": str(metadata_output_path),
        "feature_sha256": feature_sha256,
        "sample": {
            "dataset_root": str(sample.dataset_root),
            "parquet_path": str(sample.parquet_path),
            "num_rows": sample.num_rows,
            "image_format": sample.image_format,
            "wrist_image_format": sample.wrist_image_format,
            "state_shape": list(sample.state_shape),
            "action_shape": list(sample.action_shape),
        },
        "summary": summary,
    }


__all__ = [
    "LIBERO_ACTION_KEY",
    "LIBERO_IMAGE_KEY",
    "LIBERO_REQUIRED_FEATURES",
    "LIBERO_STATE_KEY",
    "LIBERO_WRIST_IMAGE_KEY",
    "RealLiberoSample",
    "decode_png_rgb",
    "export_real_libero_pistar06_features",
    "iter_real_libero_training_row_batches",
    "count_libero_task_rows",
    "list_libero_parquet_files",
    "load_real_libero_sample",
    "load_libero_task_index_to_text",
    "read_libero_parquet_rows",
    "real_libero_rows_to_training_rows",
    "select_smallest_libero_parquet",
    "validate_libero_info_schema",
]
