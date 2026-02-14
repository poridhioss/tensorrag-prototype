from __future__ import annotations

import io
import json
from typing import Any

import boto3
import joblib
import pandas as pd

from app.config import settings


class S3StorageService:
    """Drop-in replacement for StorageService that uses S3/MinIO."""

    def __init__(self, user_id: str = "default") -> None:
        client_kwargs: dict[str, Any] = {
            "aws_access_key_id": settings.S3_ACCESS_KEY,
            "aws_secret_access_key": settings.S3_SECRET_KEY,
            "region_name": settings.S3_REGION,
        }
        if settings.S3_ENDPOINT:
            client_kwargs["endpoint_url"] = settings.S3_ENDPOINT

        self.s3 = boto3.client("s3", **client_kwargs)
        self.bucket = settings.S3_BUCKET
        self.user_id = user_id

    def _key(self, pipeline_id: str, node_id: str, key: str, ext: str) -> str:
        return f"{self.user_id}/{pipeline_id}/{node_id}/{key}.{ext}"

    def _ref(self, s3_key: str) -> str:
        return f"s3://{self.bucket}/{s3_key}"

    # --- DataFrame (Parquet) ---

    def save_dataframe(
        self, pipeline_id: str, node_id: str, key: str, df: pd.DataFrame
    ) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        s3_key = self._key(pipeline_id, node_id, key, "parquet")
        self.s3.upload_fileobj(buf, self.bucket, s3_key)
        return self._ref(s3_key)

    def load_dataframe(self, ref: str) -> pd.DataFrame:
        s3_key = self._parse_ref(ref)
        buf = io.BytesIO()
        self.s3.download_fileobj(self.bucket, s3_key, buf)
        buf.seek(0)
        return pd.read_parquet(buf)

    # --- Model (Joblib) ---

    def save_model(
        self, pipeline_id: str, node_id: str, key: str, model: Any
    ) -> str:
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        s3_key = self._key(pipeline_id, node_id, key, "joblib")
        self.s3.upload_fileobj(buf, self.bucket, s3_key)
        return self._ref(s3_key)

    def load_model(self, ref: str) -> Any:
        s3_key = self._parse_ref(ref)
        buf = io.BytesIO()
        self.s3.download_fileobj(self.bucket, s3_key, buf)
        buf.seek(0)
        return joblib.load(buf)

    # --- JSON ---

    def save_json(
        self, pipeline_id: str, node_id: str, key: str, data: dict
    ) -> str:
        body = json.dumps(data, default=str).encode()
        s3_key = self._key(pipeline_id, node_id, key, "json")
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=body)
        return self._ref(s3_key)

    def load_json(self, ref: str) -> dict:
        s3_key = self._parse_ref(ref)
        resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return json.loads(resp["Body"].read())

    # --- Binary (PNG, etc.) ---

    def save_bytes(
        self, pipeline_id: str, node_id: str, key: str, data: bytes, ext: str
    ) -> str:
        s3_key = self._key(pipeline_id, node_id, key, ext)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=data)
        return self._ref(s3_key)

    def load_bytes(self, ref: str) -> bytes:
        s3_key = self._parse_ref(ref)
        resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return resp["Body"].read()

    # --- Cleanup ---

    def cleanup_pipeline(self, pipeline_id: str) -> None:
        prefix = f"{self.user_id}/{pipeline_id}/"
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = page.get("Contents", [])
            if objects:
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )

    # --- Helpers ---

    def _parse_ref(self, ref: str) -> str:
        """Extract S3 key from 's3://bucket/key' ref."""
        prefix = f"s3://{self.bucket}/"
        if ref.startswith(prefix):
            return ref[len(prefix):]
        return ref

    def get_bytes_streaming(self, ref: str) -> tuple[io.BytesIO, str]:
        """Get object as streaming body + suffix for artifact serving."""
        s3_key = self._parse_ref(ref)
        resp = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        buf = io.BytesIO(resp["Body"].read())
        suffix = "." + s3_key.rsplit(".", 1)[-1] if "." in s3_key else ""
        return buf, suffix
