#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

VALID_CATEGORIES = {"underwear", "light", "dark"}


class GarmentDB:
    def __init__(self, db_path: str | Path = "garments.db"):
        self.db_path = str(db_path)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS garments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL
                        CHECK(category IN ('underwear', 'light', 'dark')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    garment_id INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (garment_id)
                        REFERENCES garments(id)
                        ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _normalize_vector(embedding: Iterable[float]) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise ValueError("임베딩 벡터가 비어 있습니다.")
        if not np.all(np.isfinite(vec)):
            raise ValueError("임베딩에 NaN 또는 inf 값이 포함되어 있습니다.")
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("모든 값이 0인 임베딩은 사용할 수 없습니다.")
        return vec / norm

    @staticmethod
    def _vector_to_blob(vector: np.ndarray) -> bytes:
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def _blob_to_vector(blob: bytes, dimension: int) -> np.ndarray:
        vec = np.frombuffer(blob, dtype=np.float32)
        if vec.size != dimension:
            raise ValueError("DB 임베딩 크기가 손상되었습니다.")
        return vec

    def add_garment(self, name: str, category: str) -> int:
        name = name.strip()
        category = category.strip().lower()
        if not name:
            raise ValueError("옷 이름은 비어 있을 수 없습니다.")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category는 {sorted(VALID_CATEGORIES)} 중 하나여야 합니다.")

        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO garments (name, category) VALUES (?, ?)",
                (name, category),
            )
            return int(cur.lastrowid)

    def delete_garment(self, garment_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM garments WHERE id = ?", (garment_id,))
            return cur.rowcount > 0

    def list_garments(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT g.id, g.name, g.category, g.created_at,
                       COUNT(e.id) AS embedding_count
                FROM garments g
                LEFT JOIN embeddings e ON e.garment_id = g.id
                GROUP BY g.id
                ORDER BY g.id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def add_embedding(self, garment_id: int, embedding: Iterable[float]) -> int:
        vec = self._normalize_vector(embedding)
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM garments WHERE id = ?", (garment_id,)
            ).fetchone()
            if not exists:
                raise ValueError(f"garment_id={garment_id}인 옷이 없습니다.")
            cur = conn.execute(
                "INSERT INTO embeddings (garment_id, vector, dimension) VALUES (?, ?, ?)",
                (garment_id, self._vector_to_blob(vec), int(vec.size)),
            )
            return int(cur.lastrowid)

    def delete_embedding(self, embedding_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM embeddings WHERE id = ?", (embedding_id,))
            return cur.rowcount > 0

    def find_similar(
        self,
        query_embedding: Iterable[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> list[dict]:
        query = self._normalize_vector(query_embedding)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id AS embedding_id, e.vector, e.dimension,
                       g.id AS garment_id, g.name, g.category
                FROM embeddings e
                JOIN garments g ON g.id = e.garment_id
                """
            ).fetchall()

        best_by_garment = {}
        for row in rows:
            dim = int(row["dimension"])
            if dim != query.size:
                continue
            candidate = self._blob_to_vector(row["vector"], dim)
            sim = float(np.dot(query, candidate))
            gid = int(row["garment_id"])
            current = best_by_garment.get(gid)
            if current is None or sim > current["similarity"]:
                best_by_garment[gid] = {
                    "garment_id": gid,
                    "name": row["name"],
                    "category": row["category"],
                    "similarity": sim,
                    "matched_embedding_id": int(row["embedding_id"]),
                }

        results = sorted(
            best_by_garment.values(), key=lambda x: x["similarity"], reverse=True
        )
        if threshold is not None:
            results = [r for r in results if r["similarity"] >= threshold]
        return results[:top_k]

    def identify(self, query_embedding: Iterable[float], threshold: float = 0.85):
        results = self.find_similar(query_embedding, top_k=1, threshold=threshold)
        return results[0] if results else None


def load_vector_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).reshape(-1)
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32).reshape(-1)
    raise ValueError("벡터 파일은 .npy 또는 .json 형식만 지원합니다.")


def main():
    parser = argparse.ArgumentParser(description="의류 임베딩 DB")
    parser.add_argument("--db", default="garments.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add-garment")
    p.add_argument("--name", required=True)
    p.add_argument("--category", required=True, choices=sorted(VALID_CATEGORIES))

    sub.add_parser("list")

    p = sub.add_parser("delete-garment")
    p.add_argument("--id", type=int, required=True)

    p = sub.add_parser("add-embedding")
    p.add_argument("--garment-id", type=int, required=True)
    p.add_argument("--vector", required=True)

    p = sub.add_parser("search")
    p.add_argument("--vector", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=None)

    p = sub.add_parser("identify")
    p.add_argument("--vector", required=True)
    p.add_argument("--threshold", type=float, default=0.85)

    args = parser.parse_args()
    db = GarmentDB(args.db)

    if args.cmd == "add-garment":
        print(db.add_garment(args.name, args.category))
    elif args.cmd == "list":
        print(json.dumps(db.list_garments(), ensure_ascii=False, indent=2))
    elif args.cmd == "delete-garment":
        print("삭제 완료" if db.delete_garment(args.id) else "해당 옷 없음")
    elif args.cmd == "add-embedding":
        vec = load_vector_file(args.vector)
        print(db.add_embedding(args.garment_id, vec))
    elif args.cmd == "search":
        vec = load_vector_file(args.vector)
        print(json.dumps(db.find_similar(vec, args.top_k, args.threshold), ensure_ascii=False, indent=2))
    elif args.cmd == "identify":
        vec = load_vector_file(args.vector)
        result = db.identify(vec, args.threshold)
        print("미등록 의류" if result is None else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
