"""
FastAPI ETL: zip of Parquet ``*.nakama-0`` files -> merge, transform, upload rows to Supabase.

Parquet schema (sample file):
  user_id, match_id: string
  map_id: string (e.g. ``Lockdown``) — may also appear as bytes in other exports
  x, y, z: float
  ts: timestamp[ms]
  event: binary UTF-8 (e.g. ``Position``, ``Loot``) -> decoded to text

Output rows: base telemetry, ``log_date`` (from the upload form), ``event``, then ``is_bot``, ``pixel_x``, ``pixel_y`` -> table ``match_logs``.

Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# --- Supabase (MVP: hardcoded; replace with env/secret manager in production) ---
SUPABASE_URL = "https://wiaiaomyccparnaluctd.supabase.co"
SUPABASE_KEY = "sb_publishable_iYy80sIBU2grdC4mSzYp1A_G6LZEUTy"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="LILA Games - ETL Backend Engine")

# CORS: ``allow_origins=["*"]`` is not enough when the browser uses credentialed requests
# (``credentials: 'include'`` / cookies) — the spec forbids ``Access-Control-Allow-Origin: *`` then.
# Lovable previews/production often use https://<id>.lovable.dev (see Lovable docs).
# Add your Netlify/Vercel/custom domain here if you deploy the frontend elsewhere.
_CORS_LOCAL_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    "https://lovable.dev",
]
# Subdomains of Lovable + optional other hosts (adjust if your preview URL differs).
_CORS_ORIGIN_REGEX = r"https://([a-zA-Z0-9-]+\.)*lovable\.dev$|https://([a-zA-Z0-9-]+\.)*lovable\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_LOCAL_ORIGINS,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_credentials=True,
    max_age=86400,
)

SUPABASE_BATCH_SIZE = 1000


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Non-HTTP errors -> JSON 500 + logs. HTTPException is delegated (correct status + detail shape)."""
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    tb = traceback.format_exc()
    logger.error("Unhandled error on %s: %s\n%s", request.url.path, exc, tb)
    traceback.print_exc(file=sys.stderr)
    msg = str(exc).replace("\r", " ").replace("\n", " ")[:2000]
    return JSONResponse(
        status_code=500,
        content={"detail": msg, "error_type": type(exc).__name__},
    )

# Minimap: 1024×1024 image; world (x,z) -> pixels (origin + scale per map).
MAP_CONFIGS: dict[str, dict[str, float]] = {
    "AmbroseValley": {"scale": 900.0, "origin_x": -370.0, "origin_z": -473.0},
    "GrandRift": {"scale": 581.0, "origin_x": -290.0, "origin_z": -290.0},
    "Lockdown": {"scale": 1000.0, "origin_x": -500.0, "origin_z": -500.0},
}
IMAGE_SIZE = 1024

# Matches typical export: base telemetry first, derived columns last.
OUTPUT_COLUMN_ORDER = (
    "user_id",
    "match_id",
    "map_id",
    "x",
    "y",
    "z",
    "ts",
    "log_date",
    "event",
    "is_bot",
    "pixel_x",
    "pixel_y",
)


def _normalize_map_id(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, (bytes, bytearray, memoryview)):
        return bytes(val).decode("utf-8", errors="replace").strip()
    return str(val).strip()


def _decode_event(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, (bytes, bytearray, memoryview)):
        return bytes(val).decode("utf-8", errors="replace")
    return str(val)


def _add_minimap_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized (x,z) -> pixel_x, pixel_y; unknown map or bad coords -> 0."""
    n = len(df)
    pixel_x = np.zeros(n, dtype=np.int64)
    pixel_y = np.zeros(n, dtype=np.int64)

    if "map_id" not in df.columns or "x" not in df.columns or "z" not in df.columns:
        df = df.copy()
        df["pixel_x"] = pixel_x
        df["pixel_y"] = pixel_y
        return df

    mid_norm = df["map_id"].map(_normalize_map_id)
    x_all = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64)
    z_all = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=np.float64)

    for map_key, cfg in MAP_CONFIGS.items():
        m = (mid_norm == map_key).to_numpy()
        if not m.any():
            continue
        ox, oz, sc = cfg["origin_x"], cfg["origin_z"], cfg["scale"]
        u = (x_all[m] - ox) / sc
        v = (z_all[m] - oz) / sc
        ok = np.isfinite(u) & np.isfinite(v)
        px = np.zeros_like(u, dtype=np.int64)
        py = np.zeros_like(v, dtype=np.int64)
        px[ok] = (u[ok] * IMAGE_SIZE).astype(np.int64)
        py[ok] = ((1.0 - v[ok]) * IMAGE_SIZE).astype(np.int64)
        pixel_x[m] = px
        pixel_y[m] = py

    out = df.copy()
    out["pixel_x"] = pixel_x
    out["pixel_y"] = pixel_y
    return out


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in OUTPUT_COLUMN_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    return df[present + rest]


def _json_safe_value(val: Any) -> Any:
    """Make a single cell JSON/PostgREST friendly (numpy/pandas/datetime/NA)."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if pd.isna(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if val is pd.NA:
        return None
    return val


def _sanitize_records_for_supabase(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize numpy/pandas/datetime values from ``to_dict(orient='records')`` for PostgREST JSON."""
    return [{k: _json_safe_value(v) for k, v in row.items()} for row in records]


@app.get("/upload-daily-logs/")
async def upload_daily_logs_info() -> dict[str, str]:
    return {
        "message": "POST multipart: field 'file' (zip of .nakama-0 Parquet) and 'log_date' (string, e.g. 2025-02-11).",
        "try_it": "Open /docs → POST /upload-daily-logs/",
    }


def _is_nakama_parquet_filename(name: str) -> bool:
    """Match ``*.nakama-0`` case-insensitively (Windows zips often vary casing)."""
    return name.lower().endswith(".nakama-0")


def _safe_zip_entry(arcname: str) -> bool:
    """Reject zip-slip and absolute paths (security + portability)."""
    n = arcname.replace("\\", "/").lstrip("/")
    if not n or n.startswith(".."):
        return False
    return ".." not in n.split("/")


@app.post("/upload-daily-logs/")
async def process_daily_logs(
    file: UploadFile = File(...),
    log_date: str = Form(
        ...,
        description="Calendar date for this log batch (same value on every row uploaded to Supabase).",
    ),
) -> dict[str, Any]:
    log_date = log_date.strip()
    if not log_date:
        raise HTTPException(status_code=400, detail="log_date must be a non-empty string.")

    upload_name = file.filename or "upload.zip"
    logger.info("Upload started: filename=%r log_date=%r", upload_name, log_date)

    try:
        frames: list[pd.DataFrame] = []
        parquet_paths: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "upload.zip")

            with open(zip_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    f.write(chunk)

            if os.path.getsize(zip_path) == 0:
                raise HTTPException(status_code=400, detail="Empty file upload.")

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        name = info.filename
                        if not _safe_zip_entry(name):
                            logger.warning("Skip unsafe zip entry: %r", name)
                            continue
                        base = os.path.basename(name.replace("\\", "/"))
                        if not _is_nakama_parquet_filename(base):
                            continue
                        try:
                            raw = zf.read(name)
                            df = pq.read_table(io.BytesIO(raw)).to_pandas()
                            frames.append(df)
                            parquet_paths.append(name.replace("\\", "/"))
                        except Exception as exc:
                            logger.warning("Skip unreadable entry %s: %s", name, exc)
            except zipfile.BadZipFile as exc:
                logger.warning("Bad zip: %s", exc)
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not a valid zip archive.",
                ) from exc

        if not frames:
            logger.warning("No .nakama-0 Parquet files could be read from zip")
            raise HTTPException(
                status_code=400,
                detail="No valid Parquet data found (.nakama-0 files) inside the zip.",
            )

        sample = parquet_paths[:5]
        more = len(parquet_paths) - len(sample)
        logger.info(
            "Merging %s Parquet file(s); first entries %s%s",
            len(frames),
            sample,
            f" ... (+{more} more)" if more > 0 else "",
        )

        master_df = pd.concat(frames, ignore_index=True, sort=False)
        master_df["log_date"] = log_date

        if "event" in master_df.columns:
            master_df["event"] = master_df["event"].map(_decode_event)

        if "user_id" in master_df.columns:
            has_human_uuid = master_df["user_id"].astype(str).str.contains(
                "-", regex=False, na=False
            )
            master_df["is_bot"] = np.logical_not(np.asarray(has_human_uuid, dtype=bool))

        master_df = _add_minimap_pixels(master_df)
        master_df = _reorder_columns(master_df)

        records = master_df.to_dict(orient="records")
        records = _sanitize_records_for_supabase(records)

        def _insert_batches() -> None:
            for i in range(0, len(records), SUPABASE_BATCH_SIZE):
                chunk = records[i : i + SUPABASE_BATCH_SIZE]
                supabase.table("match_logs").insert(chunk).execute()

        try:
            await asyncio.to_thread(_insert_batches)
        except Exception as db_exc:
            logger.exception("Supabase insert failed: %s", db_exc)
            raise HTTPException(
                status_code=502,
                detail=f"Database upload failed: {db_exc!s}",
            ) from db_exc

        logger.info(
            "Supabase upload OK: total_rows=%s log_date=%r",
            len(records),
            log_date,
        )

        return {
            "message": "Data successfully processed and uploaded to Supabase",
            "total_rows_inserted": len(records),
        }
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Upload processing failed:\n%s", tb)
        traceback.print_exc(file=sys.stderr)
        safe = str(exc).replace("\r", " ").replace("\n", " ")[:2000]
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {safe}",
        ) from exc


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness + quick import sanity (call this before debugging uploads)."""
    return {
        "status": "ok",
        "pandas": pd.__version__,
        "pyarrow": getattr(pa, "__version__", "unknown"),
    }
