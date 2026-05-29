import os
import json
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from audio_processor import (
    get_youtube_metadata,
    extract_heatmap_peaks,
    download_youtube_audio,
    extract_audio_features,
    vectorized_sliding_window_filter,
    find_best_match_window_kadane,
    auto_discover_matches,
    subsequence_dtw
)

app = FastAPI(title="Who Sampled Engine API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "db.json")

os.makedirs(DATA_DIR, exist_ok=True)

# Helper to load DB
def load_db():
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, "w") as f:
            json.dump({"tracks": {}}, f)
        return {"tracks": {}}
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"tracks": {}}

# Helper to save DB
def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def intervals_overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
    # Returns True if there is any overlap between [start1, end1] and [start2, end2]
    return max(start1, start2) < min(end1, end2)

def filter_non_overlapping(peaks: list, max_k: int = 5, check_query: bool = False) -> list:
    accepted = []
    for p in peaks:
        overlap = False
        for a in accepted:
            # Check host overlap
            if intervals_overlap(p["start_time"], p["end_time"], a["start_time"], a["end_time"]):
                overlap = True
                break
            # Check query overlap if requested
            if check_query and "query_start_time" in p and "query_start_time" in a:
                if intervals_overlap(p["query_start_time"], p["query_end_time"], a["query_start_time"], a["query_end_time"]):
                    overlap = True
                    break
        if not overlap:
            accepted.append(p)
            if len(accepted) >= max_k:
                break
    return accepted

class AnalyzeURLRequest(BaseModel):
    url: str

class MatchRequest(BaseModel):
    host_id: str
    query_id: str
    threshold: float = 0.45
    sub_segments: int = 2
    crop_start: Optional[float] = None
    crop_end: Optional[float] = None
    auto_discover: bool = False   # M×N grid search mode — no crop needed

@app.get("/api/tracks")
def get_tracks():
    db = load_db()
    # Return track list without heavy features path
    tracks_list = []
    for tid, info in db.get("tracks", {}).items():
        tracks_list.append({
            "id": tid,
            "title": info.get("title", "Unknown"),
            "url": info.get("url", ""),
            "duration": info.get("duration", 0.0),
            "heatmap": info.get("heatmap", []),
            "has_features": True
        })
    return tracks_list

@app.post("/api/tracks/analyze-url")
def analyze_url(req: AnalyzeURLRequest):
    """
    Given a YouTube URL, extract metadata, download audio, compute chroma features,
    and save to the database.
    """
    url = req.url
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
    print(f"Fetching metadata for {url}...")
    metadata = get_youtube_metadata(url)
    if not metadata:
        raise HTTPException(status_code=400, detail="Could not extract YouTube metadata")
        
    title = metadata.get("title", f"YouTube Track {uuid.uuid4().hex[:6]}")
    track_id = str(uuid.uuid4())
    
    # Extract engagement heatmap peaks
    peaks = extract_heatmap_peaks(metadata, top_n=5, min_duration=4.0)
    
    try:
        # Download and transcode audio
        wav_path = download_youtube_audio(url, DATA_DIR, track_id)
        
        # Extract features
        features = extract_audio_features(wav_path)
        
        # Clean up the large audio file to save disk space, keeping only the computed features.
        # But wait! If we want to play the audio in the frontend, we might want to keep the WAV or mp3.
        # Let's keep it in the data directory so we can serve/play it!
        # Actually, let's keep it, but we can also serve it as static files.
        
        # Save features to a separate file
        features_filename = f"features_{track_id}.json"
        features_path = os.path.join(DATA_DIR, features_filename)
        with open(features_path, "w") as f:
            json.dump(features, f)
            
        # Update Database
        db = load_db()
        db["tracks"][track_id] = {
            "id": track_id,
            "title": title,
            "url": url,
            "duration": features["duration"],
            "heatmap": peaks,
            "features_file": features_filename,
            "audio_file": f"{track_id}.wav"
        }
        save_db(db)
        
        return {
            "status": "success",
            "track_id": track_id,
            "title": title,
            "duration": features["duration"],
            "heatmap": peaks
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process track: {str(e)}")

@app.post("/api/tracks/upload")
def upload_track(file: UploadFile = File(...), title: str = Form(...)):
    """
    Handle local audio file uploads.
    """
    track_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    temp_path = os.path.join(DATA_DIR, f"{track_id}_temp.{ext}")
    
    # Save uploaded file
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(file.file.read())
            
        # Convert to standard mono WAV using ffmpeg if it is not already
        wav_path = os.path.join(DATA_DIR, f"{track_id}.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_path,
            "-ar", "22050",
            "-ac", "1",
            wav_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_path)  # Remove temp
        
        # Extract features
        features = extract_audio_features(wav_path)
        
        # Save features
        features_filename = f"features_{track_id}.json"
        features_path = os.path.join(DATA_DIR, features_filename)
        with open(features_path, "w") as f:
            json.dump(features, f)
            
        db = load_db()
        db["tracks"][track_id] = {
            "id": track_id,
            "title": title,
            "url": "Uploaded File",
            "duration": features["duration"],
            "heatmap": [],  # No heatmap for uploaded files
            "features_file": features_filename,
            "audio_file": f"{track_id}.wav"
        }
        save_db(db)
        
        return {
            "status": "success",
            "track_id": track_id,
            "title": title,
            "duration": features["duration"]
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/match")
def match_tracks(req: MatchRequest):
    """
    Perform local sequence alignment matching between host and query track.
    1. Loads features.
    2. Runs vectorized O(1) sliding window filter.
    3. Triggers Subsequence DTW on candidate matching segments.
    """
    db = load_db()
    tracks = db.get("tracks", {})
    
    if req.host_id not in tracks or req.query_id not in tracks:
        raise HTTPException(status_code=400, detail="Host or Query track not found in database")
        
    host_info = tracks[req.host_id]
    query_info = tracks[req.query_id]
    
    # Load features
    host_feat_path = os.path.join(DATA_DIR, host_info["features_file"])
    query_feat_path = os.path.join(DATA_DIR, query_info["features_file"])
    
    try:
        with open(host_feat_path, "r") as f:
            host_feat = json.load(f)
        with open(query_feat_path, "r") as f:
            query_feat = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load track features: {str(e)}")
        
    # Convert lists back to numpy arrays
    host_chroma = np.array(host_feat["chroma"])
    query_chroma = np.array(query_feat["chroma"])

    host_times = np.array(host_feat["times"])
    query_times = np.array(query_feat["times"])
    # AUTO-DISCOVERY MODE (M×N grid search)
    # Triggered when req.auto_discover=True (no crop selected).
    # Divides both tracks into segments, finds the top-K best-matching pairs
    # via cosine-similarity matrix + NMS, then expands each pair with Kadane.
    # Returns immediately — no sliding-window filter or DTW needed.
    # ─────────────────────────────────────────────────────────────────────────
    if req.auto_discover:
        discovered = auto_discover_matches(
            host_chroma=host_chroma,
            host_times=host_times,
            query_chroma=query_chroma,
            query_times=query_times,
            M=20, N=20, top_k=15
        )
        if not discovered:
            raise HTTPException(status_code=400, detail="Auto-discovery found no matching segments.")

        # Map to candidate peak dictionaries
        peaks = [
            {
                "start_time":       m["host_start_time"],
                "end_time":         m["host_end_time"],
                "query_start_time": m["query_start_time"],
                "query_end_time":   m["query_end_time"],
                "similarity":       m["similarity"],
            }
            for m in discovered
        ]
        
        # Filter non-overlapping pairs, checking both host and query intervals
        filtered_peaks = filter_non_overlapping(peaks, max_k=5, check_query=True)
        if not filtered_peaks:
            filtered_peaks = peaks[:5]  # fallback if all pruned
            
        best = filtered_peaks[0]
        secondary = [
            {
                "frame":            0,
                "start_time":       p["start_time"],
                "end_time":         p["end_time"],
                "query_start_time": p["query_start_time"],
                "query_end_time":   p["query_end_time"],
                "similarity":       p["similarity"],
            }
            for p in filtered_peaks
        ]
        total_cells = 20 * 20
        return {
            "best_match": {
                "start_time":       best["start_time"],
                "end_time":         best["end_time"],
                "score":            1.0 - best["similarity"],
                "confidence":       best["similarity"],
                "kadane_score":     best["similarity"],
                "query_start_time": best["query_start_time"],
                "query_end_time":   best["query_end_time"],
            },
            "pruning_stats": {
                "total_windows":  total_cells,
                "pruned_windows": total_cells - len(discovered),
                "pruning_rate":   1.0 - len(discovered) / total_cells,
            },
            "similarity_curve": {"times": [], "similarities": []},
            "warping_path":     [],
            "cost_matrix":      {"matrix": [], "host_times": [], "query_times": []},
            "secondary_matches": secondary,
            "is_auto_mode": True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STANDARD MODE — crop-based pipeline (existing Kadane + DTW flow)
    # ─────────────────────────────────────────────────────────────────────────
    # Slice query chroma and times if crop times are specified
    if req.crop_start is not None and req.crop_end is not None:
        crop_indices = np.where((query_times >= req.crop_start) & (query_times <= req.crop_end))[0]
        if len(crop_indices) > 0:
            query_chroma = query_chroma[:, crop_indices]
            query_times = query_times[crop_indices]
    
    # 1. Run O(1) Sliding-Window Filter
    sim_curve, candidate_frames = vectorized_sliding_window_filter(
        host_chroma=host_chroma,
        query_chroma=query_chroma,
        threshold=req.threshold,
        K=req.sub_segments
    )
    
    # Calculate what percentage of windows were pruned
    total_windows = len(sim_curve)
    pruned_count = total_windows - len(candidate_frames)
    pruning_rate = (pruned_count / total_windows) if total_windows > 0 else 1.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. BOUNDARY DETECTION: Kadane's max-subarray on the similarity curve
    # ─────────────────────────────────────────────────────────────────────────
    # The similarity curve sim_curve[j] is the average chroma cosine similarity
    # between the query and host window starting at frame j.
    #
    # We shift every value by the baseline so:
    #   sim > baseline  →  positive contribution (good match frame, keep expanding)
    #   sim < baseline  →  negative contribution (weak frame, costs us to include)
    #
    # Kadane's algorithm then finds the window [win_start, win_end] with the
    # maximum total NET score.  This naturally solves the user's problem:
    #
    #   8-sec perfect match:  8  * (0.80 - 0.45) = 2.80  total score
    #   14-sec match (3 bad): 11 * (0.80 - 0.45)
    #                       +  3 * (0.30 - 0.45) = 3.85 - 0.45 = 3.40  ← WINS
    #
    # The 3 weak seconds are "paid for" by the 11 strong seconds around them.
    query_N = query_chroma.shape[1]   # query length in frames

    # Adaptive Kadane baseline: frames must beat the 65th-percentile similarity
    # for this specific track pair to earn a positive contribution.
    # A fixed threshold (e.g. 0.45) doesn't work for pop music where even
    # unrelated sections score 0.6+.  Using a percentile makes the baseline
    # relative to THIS comparison, so Kadane only extends into frames that are
    # genuinely above-average matches for this pair of songs.
    # We also enforce req.threshold as a hard floor (don't expand into frames
    # the user considers too weak).
    kadane_baseline = max(req.threshold, float(np.percentile(sim_curve, 65)))

    win_start, win_end, kadane_score = find_best_match_window_kadane(
        sim_curve, baseline=kadane_baseline
    )

    # Each similarity window j covers host frames [j, j + query_N - 1].
    # The full detected host region therefore spans [win_start, win_end + query_N - 1].
    host_start_idx = win_start
    host_end_idx   = min(len(host_times) - 1, win_end + query_N - 1)

    start_time = float(host_times[host_start_idx])
    end_time   = float(host_times[host_end_idx])

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ALIGNMENT: Subsequence DTW within the Kadane window only
    # ─────────────────────────────────────────────────────────────────────────
    # Running DTW on the full host is O(N * L). By restricting to the Kadane
    # window it becomes O(N * W) where W << L, which is much faster.
    # The warping path and cost matrix are used only for visualization.
    host_window_chroma = host_chroma[:, host_start_idx : host_end_idx + 1]
    host_window_times  = host_times[host_start_idx : host_end_idx + 1]

    if host_window_chroma.shape[1] >= query_N:
        alignment = subsequence_dtw(query_chroma, host_window_chroma)
        # Offset warping path frame indices back to absolute host frame space
        time_path = []
        for coord in alignment["warping_path"]:
            i, j_rel = coord[0], coord[1]
            j_abs = host_start_idx + j_rel
            if j_abs < len(host_times):
                time_path.append({
                    "query_frame": i,
                    "query_time":  float(query_times[i]),
                    "host_frame":  j_abs,
                    "host_time":   float(host_times[j_abs])
                })
        dtw_score = alignment["score"]

        # Sub-sampled cost matrix for the frontend heatmap
        cost_matrix_np = np.array(alignment["cost_matrix"])
        _, W_cols = cost_matrix_np.shape
        col_step = max(1, W_cols // 300)
        sub_cost_matrix = cost_matrix_np[:, ::col_step].tolist()
        sub_host_times  = host_window_times[::col_step].tolist()
    else:
        # Window too small for DTW (edge case) — skip alignment visualization
        time_path       = []
        dtw_score       = 1.0
        sub_cost_matrix = []
        sub_host_times  = []

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SECONDARY MATCHES from similarity-curve local peaks
    # ─────────────────────────────────────────────────────────────────────────
    raw_peaks = []
    for idx in candidate_frames:
        left  = max(0, idx - query_N // 2)
        right = min(total_windows, idx + query_N // 2)
        if sim_curve[idx] == np.max(sim_curve[left:right]):
            raw_peaks.append({
                "frame":      int(idx),
                "start_time": float(host_times[idx]),
                "end_time":   float(host_times[min(len(host_times)-1, idx + query_N - 1)]),
                "similarity": float(sim_curve[idx])
            })
            
    # Sort all raw peaks by similarity descending
    raw_peaks = sorted(raw_peaks, key=lambda x: x["similarity"], reverse=True)
    
    # Filter non-overlapping pairs, checking host intervals
    local_peaks = filter_non_overlapping(raw_peaks, max_k=5, check_query=False)

    confidence = max(0.0, min(1.0, 1.0 - dtw_score))

    return {
        "best_match": {
            "start_time": start_time,
            "end_time":   end_time,
            "score":      dtw_score,
            "confidence": confidence,
            "kadane_score": float(kadane_score),
        },
        "pruning_stats": {
            "total_windows": total_windows,
            "pruned_windows": pruned_count,
            "pruning_rate":  float(pruning_rate)
        },
        "similarity_curve": {
            "times":        host_times[:total_windows].tolist(),
            "similarities": sim_curve.tolist()
        },
        "warping_path": time_path,
        "cost_matrix": {
            "matrix":      sub_cost_matrix,
            "host_times":  sub_host_times,
            "query_times": query_times.tolist()
        },
        "secondary_matches": local_peaks[:5]
    }

# Endpoint to serve the audio files as static files with HTTP Range support
from fastapi.responses import FileResponse, StreamingResponse

def get_range_header(range_header: str, file_size: int):
    start_str, end_str = range_header.replace("bytes=", "").split("-")
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else file_size - 1
    return start, min(end, file_size - 1)

def file_iterator(file_path: str, start: int, end: int, chunk_size: int = 1024 * 1024):
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data

@app.get("/api/tracks/{track_id}/audio")
def get_track_audio(track_id: str, request: Request):
    db = load_db()
    tracks = db.get("tracks", {})
    if track_id not in tracks:
        raise HTTPException(status_code=404, detail="Track not found")
    audio_file = tracks[track_id]["audio_file"]
    audio_path = os.path.join(DATA_DIR, audio_file)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    file_size = os.path.getsize(audio_path)
    range_header = request.headers.get("range")
    
    if range_header:
        try:
            start, end = get_range_header(range_header, file_size)
            content_length = end - start + 1
            return StreamingResponse(
                file_iterator(audio_path, start, end),
                status_code=206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Content-Length": str(content_length),
                    "Accept-Ranges": "bytes",
                },
                media_type="audio/wav"
            )
        except Exception as e:
            print(f"Error serving range request: {e}")
            
    # Fallback to full download if no range header
    return FileResponse(audio_path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
