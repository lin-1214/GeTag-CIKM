#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert MovieLens 1M (movies.dat, ratings.dat) to:
each row = one user's full sequence (sorted by time)
each cell = "[user_id, datetime.datetime(...), rating, 'movie title']"

Output: ml1m_events_by_user.csv
"""

import os
import csv
import sys
import json
from datetime import datetime
from collections import defaultdict

MOVIES = "movies.dat"
RATINGS = "ratings.dat"
OUTPUT = "ml1m_events_by_user.csv"

def load_movies(path: str) -> dict:
    """
    movies.dat format (ML-1M):
      MovieID::Title::Genres
    """
    mid2title = {}
    with open(path, "r", encoding="latin-1") as f:  # ML-1M 是 latin-1
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 2:
                continue
            try:
                mid = int(parts[0])
            except ValueError:
                continue
            title = parts[1].strip()
            mid2title[mid] = title
    return mid2title

def iter_ratings(path: str):
    """
    ratings.dat format (ML-1M):
      UserID::MovieID::Rating::Timestamp
    """
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 4:
                continue
            try:
                uid = int(parts[0])
                mid = int(parts[1])
                rating = int(float(parts[2]))  # ML-1M 是整數 1..5，但保險轉一下
                ts = int(parts[3])
            except ValueError:
                continue
            yield uid, mid, rating, ts

def format_event_cell(user_id: int, dt: datetime, rating: int, title: str) -> str:
    """
    產生你要的 cell 文字：
    [user_id, datetime.datetime(Y, M, D, h, m, s, 0), rating, 'title']
    """
    # 為了和你們樣式一致，microsecond 固定寫 0
    return f"[{user_id}, datetime.datetime({dt.year}, {dt.month}, {dt.day}, {dt.hour}, {dt.minute}, {dt.second}, 0), {rating}, '{title}']"

def main():
    here = os.path.abspath(os.path.dirname(__file__))
    movies_path = os.path.join(here, MOVIES)
    ratings_path = os.path.join(here, RATINGS)
    output_path = os.path.join(here, OUTPUT)
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    json_dir = os.path.join(project_root, "json")
    dataset_cache_dirs = [
        os.path.join(project_root, "dataset", "movielens", "ml-1m"),
        os.path.join(project_root, "src", "dataset", "movielens", "ml-1m"),
    ]

    if not os.path.exists(movies_path):
        print(f"ERROR: {movies_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(ratings_path):
        print(f"ERROR: {ratings_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Loading movies mapping ...")
    mid2title = load_movies(movies_path)

    os.makedirs(json_dir, exist_ok=True)
    movie_id_to_name = {str(mid): title for mid, title in mid2title.items()}
    movie_name_to_id = {}
    for mid, title in mid2title.items():
        normalized = title.strip()
        movie_name_to_id.setdefault(normalized, mid)

    with open(os.path.join(json_dir, "movie_id_to_name.json"), "w", encoding="utf-8") as f_out:
        json.dump(movie_id_to_name, f_out, ensure_ascii=False, indent=2)
    with open(os.path.join(json_dir, "movie_name_to_id.json"), "w", encoding="utf-8") as f_out:
        json.dump(movie_name_to_id, f_out, ensure_ascii=False, indent=2)
    print(f"✓ Wrote movie_id_to_name.json and movie_name_to_id.json to {json_dir}")

    print("Reading ratings and grouping by user ...")
    by_user = defaultdict(list)  # uid -> list[(ts, rating, title, movieId)]
    missing_title = 0

    for uid, mid, rating, ts in iter_ratings(ratings_path):
        title = mid2title.get(mid)
        if title is None:
            # 少數情況下可能找不到對應 title（理論上 ML-1M 不會）
            missing_title += 1
            title = f"MovieID_{mid}"
        by_user[uid].append((ts, rating, title, mid))

    if missing_title:
        print(f"WARNING: {missing_title} ratings with missing title mapping, filled with 'MovieID_<id>'")

    print("Sorting each user's events by timestamp ...")
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x[0])  # 按 ts 升序

    print("Preparing retrieval-compatible cache files ...")
    all_movie_ids = sorted({mid for events in by_user.values() for _, _, _, mid in events})
    user_ids = sorted(by_user.keys())
    n_users = len(user_ids)
    train_end = int(n_users * 0.7)
    val_end = int(n_users * 0.9)
    splits = {
        "train": user_ids[:train_end],
        "val": user_ids[train_end:val_end],
        "test": user_ids[val_end:],
    }

    split_rows = {}
    split_sessions = {}
    for split_name, user_subset in splits.items():
        rows = []
        sessions = []
        for uid in user_subset:
            movie_seq = [mid for _, _, _, mid in by_user[uid]]
            sessions.append((uid, json.dumps(movie_seq)))
            for ts, rating, title, mid in by_user[uid]:
                rows.append([uid, mid, rating, ts])
        split_rows[split_name] = rows
        split_sessions[split_name] = sessions

    for cache_dir in dataset_cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "all_items.json"), "w", encoding="utf-8") as f_out:
            json.dump(all_movie_ids, f_out, ensure_ascii=False, indent=2)

        for split_name in splits.keys():
            rows = split_rows[split_name]
            generic_path = os.path.join(cache_dir, f"{split_name}.csv")
            with open(generic_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["userId", "movieId", "rating", "timestamp"])
                writer.writerows(rows)

            interactions_path = os.path.join(cache_dir, f"{split_name}_loli.csv")
            with open(interactions_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["userId", "movieId", "rating", "timestamp"])
                writer.writerows(rows)

            session_path = os.path.join(cache_dir, f"session_{split_name}.csv")
            with open(session_path, "w", newline="", encoding="utf-8") as f_session:
                writer = csv.writer(f_session)
                writer.writerow(["sessionId", "loaded_pids"])
                writer.writerows(split_sessions[split_name])

    print(f"Writing CSV to {output_path} ...")
    # 每列長度不一，用 csv.writer 逐列寫出（每格是一個字串）
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for uid, events in by_user.items():
            row_cells = []
            for ts, rating, title, _ in events:
                dt = datetime.utcfromtimestamp(ts)  # 若要台灣時區可改成 fromtimestamp(ts) 後再 +8h
                cell = format_event_cell(uid, dt, rating, title)
                row_cells.append(cell)
            writer.writerow(row_cells)

    print("Done.")

if __name__ == "__main__":
    main()