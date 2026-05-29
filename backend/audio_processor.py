import os
import json
import subprocess
import numpy as np
import librosa
import soundfile as sf

def get_youtube_metadata(url: str):
    """
    Fetch YouTube metadata (including the engagement heatmap) using yt-dlp.
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--skip-download",
        url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
        return metadata
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return {}

def extract_heatmap_peaks(metadata: dict, top_n: int = 5, min_duration: float = 4.0):
    """
    Extract the highest-engagement peaks from the YouTube heatmap.
    Returns a list of dicts with start_time, end_time, and relative intensity value.
    """
    heatmap = metadata.get("heatmap")
    if not heatmap:
        # Fallback: if no heatmap, return empty or we could estimate from transients.
        return []
    
    # Heatmap is a list of dicts: [{'start_time': 0.0, 'end_time': 3.5, 'value': 0.1}, ...]
    # Sort by engagement value descending
    sorted_heatmap = sorted(heatmap, key=lambda x: x.get("value", 0), reverse=True)
    
    peaks = []
    for item in sorted_heatmap:
        start = item.get("start_time", 0.0)
        end = item.get("end_time", 0.0)
        val = item.get("value", 0.0)
        
        # Avoid selecting overlapping segments
        overlap = False
        for p in peaks:
            if not (end <= p["start_time"] or start >= p["end_time"]):
                overlap = True
                break
        
        if not overlap:
            duration = end - start
            if duration < min_duration:
                # Pad to at least min_duration if possible
                mid = (start + end) / 2.0
                start = max(0.0, mid - min_duration / 2.0)
                end = start + min_duration
            
            peaks.append({
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "value": round(val, 4)
            })
            if len(peaks) >= top_n:
                break
                
    # Sort chronological for display
    return sorted(peaks, key=lambda x: x["start_time"])

def download_youtube_audio(url: str, output_dir: str, track_id: str):
    """
    Download audio from YouTube as a mono WAV at 22050 Hz.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{track_id}.wav")
    
    # yt-dlp command to extract mono WAV audio at 22050 Hz
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg: -ar 22050 -ac 1",
        "-o", os.path.join(output_dir, f"{track_id}.%(ext)s"),
        url
    ]
    
    print(f"Downloading YouTube audio to {output_path}...")
    subprocess.run(cmd, check=True)
    
    # Sometimes yt-dlp might keep the .wav extension or transcode differently
    expected_path = os.path.join(output_dir, f"{track_id}.wav")
    if not os.path.exists(expected_path) and os.path.exists(expected_path + ".wav"):
        os.rename(expected_path + ".wav", expected_path)
        
    if not os.path.exists(expected_path):
        # Look for any wav file starting with track_id
        files = [f for f in os.listdir(output_dir) if f.startswith(track_id) and f.endswith(".wav")]
        if files:
            os.rename(os.path.join(output_dir, files[0]), expected_path)
        else:
            raise FileNotFoundError(f"Could not locate downloaded audio file for {track_id}")
            
    return expected_path

def extract_audio_features(audio_path: str, sr: int = 22050, hop_length: int = 1024):
    """
    Load mono audio and extract Chroma CENS features and energy envelopes.
    """
    print(f"Extracting features from {audio_path}...")
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Compute Chroma CENS (energy-normalized statistics, excellent for matching)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    
    # Normalize chroma columns to unit length for cosine similarity
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    
    # Compute energy envelope (RMS) for energy-based filtering
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    
    return {
        "chroma": chroma_norm.tolist(),  # List of lists for JSON serialization
        "rms": rms.tolist(),
        "times": times.tolist(),
        "duration": duration,
        "sr": sr,
        "hop_length": hop_length
    }

def vectorized_sliding_window_filter(host_chroma: np.ndarray, query_chroma: np.ndarray, threshold: float = 0.45, K: int = 2):
    """
    Run an O(1) sliding window heuristic to drop unlikely match positions.
    Splits the query into K sub-segments and computes cumulative sum averages.
    
    Returns:
        similarities: np.ndarray of shape (L - N + 1,)
        candidate_frames: np.ndarray of shape (NumCandidates,) containing sliding window indices that pass.
    """
    # Dimensions: 12 x L for host, 12 x N for query
    L = host_chroma.shape[1]
    N = query_chroma.shape[1]
    
    if L < N:
        return np.array([]), np.array([])
    
    # 1. Prepare query sub-segment unit vectors
    M = N // K  # Frames per sub-segment
    query_sub_vectors = []
    for k in range(K):
        q_sub = query_chroma[:, k * M : (k + 1) * M]
        q_avg = np.mean(q_sub, axis=1)
        q_unit = q_avg / (np.linalg.norm(q_avg) + 1e-8)
        query_sub_vectors.append(q_unit)
        
    # 2. Precompute host cumulative sums along time axis
    # S has shape (12, L + 1)
    S = np.zeros((12, L + 1))
    S[:, 1:] = np.cumsum(host_chroma, axis=1)
    
    # 3. Vectorized sliding window calculation
    # We will slide a window of size N. The window starts at frame j in [0, L - N].
    # The k-th sub-segment in the window starts at j + k*M and ends at j + (k+1)*M - 1.
    num_windows = L - N + 1
    sub_similarities = []
    
    for k in range(K):
        # Start and end indexes for k-th sub-segment for all windows j
        # For window j, sub-segment spans [j + k*M, j + (k+1)*M - 1]
        start_indices = np.arange(num_windows) + k * M
        end_indices = start_indices + M
        
        # Sum is S[:, end_indices] - S[:, start_indices]
        sums = S[:, end_indices] - S[:, start_indices]  # Shape (12, num_windows)
        
        # Normalize sum vectors
        norms = np.linalg.norm(sums, axis=0, keepdims=True) + 1e-8
        avg_units = sums / norms  # Shape (12, num_windows)
        
        # Cosine similarity with query sub-segment k
        q_unit = query_sub_vectors[k]  # Shape (12,)
        sim_k = np.dot(avg_units.T, q_unit)  # Shape (num_windows,)
        sub_similarities.append(sim_k)
        
    # Average similarity across all sub-segments
    similarities = np.mean(sub_similarities, axis=0)  # Shape (num_windows,)
    
    # Get indices where similarity exceeds threshold
    candidate_frames = np.where(similarities >= threshold)[0]
    
    return similarities, candidate_frames


def auto_discover_matches(
    host_chroma: np.ndarray,
    host_times: np.ndarray,
    query_chroma: np.ndarray,
    query_times: np.ndarray,
    M: int = 20,
    N: int = 20,
    top_k: int = 5,
) -> list:
    """
    Coarse-to-fine M x N anchor grid search — used when no crop is selected.

    Algorithm
    ---------
    1.  Divide the host into M equal segments and the query into N segments.
        Compute a unit-normalised mean chroma vector (12-D) per segment.
    2.  Build the N x M cosine-similarity matrix with a single matmul:
            sim_matrix = query_segs @ host_segs.T
    3.  Greedy NMS: iterate pairs high→low; accept pair (qi, hi) only when no
        already-accepted pair occupies a host segment within ±1 of hi.
    4.  For each accepted anchor, extract the query segment and a padded host
        window, run vectorized_sliding_window_filter + Kadane to expand to the
        natural musical boundary.
    5.  Return top_k results sorted best-first, each carrying both host AND
        query timestamps (so the frontend can highlight both waveforms).
    """
    L = host_chroma.shape[1]
    Q = query_chroma.shape[1]

    # 1. Segment mean chroma vectors
    def _seg_means(chroma, n_segs):
        seg_len = max(1, chroma.shape[1] // n_segs)
        segs = []
        for i in range(n_segs):
            s = i * seg_len
            e = min(chroma.shape[1], (i + 1) * seg_len)
            v = chroma[:, s:e].mean(axis=1)
            v = v / (np.linalg.norm(v) + 1e-8)
            segs.append((v, s, e))
        return segs

    h_segs = _seg_means(host_chroma, M)
    q_segs = _seg_means(query_chroma, N)

    # 2. N x M similarity matrix
    H_mat = np.stack([s[0] for s in h_segs])   # (M, 12)
    Q_mat = np.stack([s[0] for s in q_segs])   # (N, 12)
    sim_matrix = Q_mat @ H_mat.T               # (N, M)

    # 3. Greedy NMS over pairs
    used_host = set()
    anchors   = []
    for flat_idx in np.argsort(sim_matrix.ravel())[::-1]:
        qi, hi = divmod(int(flat_idx), M)
        if any(abs(hi - uh) <= 1 for uh in used_host):
            continue
        used_host.add(hi)
        anchors.append((qi, hi, float(sim_matrix[qi, hi])))
        if len(anchors) >= top_k:
            break

    # 4. Expand each anchor via Kadane on a local similarity curve
    host_seg_len = max(1, L // M)
    results = []

    for qi, hi, coarse_sim in anchors:
        q_s, q_e = q_segs[qi][1], q_segs[qi][2]
        q_slice  = query_chroma[:, q_s:q_e]
        if q_slice.shape[1] < 4:
            continue

        h_pad   = host_seg_len
        h_win_s = max(0, h_segs[hi][1] - h_pad)
        h_win_e = min(L, h_segs[hi][2] + h_pad)
        h_slice = host_chroma[:, h_win_s:h_win_e]

        if h_slice.shape[1] < q_slice.shape[1]:
            expanded_hs = h_segs[hi][1]
            expanded_he = min(L - 1, h_segs[hi][2])
        else:
            sim_local, _ = vectorized_sliding_window_filter(
                h_slice, q_slice, threshold=0.0, K=2
            )
            if len(sim_local) == 0:
                expanded_hs = h_segs[hi][1]
                expanded_he = min(L - 1, h_segs[hi][2])
            else:
                kb = max(0.40, float(np.percentile(sim_local, 60)))
                ws, we, _ = find_best_match_window_kadane(sim_local, baseline=kb)
                q_N = q_slice.shape[1]
                expanded_hs = h_win_s + ws
                expanded_he = min(L - 1, h_win_s + we + q_N - 1)

        hs_t = float(host_times[min(expanded_hs, len(host_times) - 1)])
        he_t = float(host_times[min(expanded_he, len(host_times) - 1)])
        qs_t = float(query_times[min(q_s, len(query_times) - 1)])
        qe_t = float(query_times[min(q_e - 1, len(query_times) - 1)])

        results.append({
            "host_start_time":  hs_t,
            "host_end_time":    he_t,
            "query_start_time": qs_t,
            "query_end_time":   qe_t,
            "similarity":       coarse_sim,
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def find_best_match_window_kadane(
    similarities: np.ndarray,
    baseline: float = 0.45,
) -> tuple:
    """
    Find the contiguous host window with maximum total 'net similarity' using
    Kadane's maximum-subarray algorithm.

    Why this beats normalised DTW for the user's case
    --------------------------------------------------
    Normalised DTW (cost / path_length) measures AVERAGE quality. It always
    prefers a short perfect match over a longer match that includes a few weak
    frames — even if the longer match has higher total musical evidence.

    Kadane measures TOTAL net score: each similarity value is shifted by a
    baseline so frames below the baseline contribute negatively.  This creates
    the right tradeoff automatically:

        8-sec  perfect match:  8  * (0.80 - 0.45) =  2.8  total score
        14-sec decent  match: 11  * (0.80 - 0.45)   <- 11 good seconds
                            +  3  * (0.30 - 0.45)   <- 3 weak seconds
                            = 3.85 - 0.45 = 3.40   > 2.8  ← 14-sec WINS

    The 3 weak seconds are "paid for" by the 11 strong seconds around them,
    which is exactly the desired behaviour.

    Parameters
    ----------
    similarities : np.ndarray of shape (num_windows,)
        Sliding-window similarity scores from vectorized_sliding_window_filter.
        similarities[j] = average chroma cosine similarity between the query
        and host frames [j, j + query_N - 1].
    baseline : float
        Similarity threshold.  Frames above baseline earn positive score;
        frames below baseline subtract from the running total.
        Typical value: the same threshold used in the O(1) filter (0.45).

    Returns
    -------
    (best_start, best_end, max_sum) : (int, int, float)
        best_start, best_end are INCLUSIVE indices into `similarities`.
        max_sum is the total net score of the winning window.
    """
    shifted = similarities - baseline

    max_sum      = float("-inf")
    current_sum  = 0.0
    best_start   = 0
    best_end     = 0
    current_start = 0

    for j, s in enumerate(shifted):
        current_sum += s
        if current_sum > max_sum:
            max_sum    = current_sum
            best_start = current_start
            best_end   = j
        if current_sum < 0.0:
            current_sum   = 0.0
            current_start = j + 1

    return best_start, best_end, max_sum


def subsequence_dtw(query_chroma: np.ndarray, host_chroma: np.ndarray, min_span_frames: int = 0):
    """
    Subsequence DTW: aligns query_chroma (12 x N) against host_chroma (12 x L).
    The query can start anywhere in the host for free (subsequence semantics).

    Key design decisions
    --------------------
    1. Path-length normalisation: D[N-1,j] / P[N-1,j] gives the average per-step
       cost regardless of where in the host the alignment ends.
    2. Host-span constraint (min_span_frames): only consider ending positions j
       where the path's host span (j - j_origin + 1) >= min_span_frames.  Without
       this, a tiny 4-frame "perfect" match always beats a 15-second region that
       is good-but-not-perfect, because the accumulated cost of the short match is
       smaller in absolute terms.

    Parameters
    ----------
    min_span_frames : int
        Minimum number of HOST frames the winning path must span.
        Set to 0 to disable (pure normalised-cost selection).

    Returns
    -------
    dict with cost_matrix, warping_path, start_frame, end_frame, score
    """
    N = query_chroma.shape[1]
    L = host_chroma.shape[1]

    # 1. Local distance matrix: cosine distance (columns already unit-normalized)
    #    Shape: (N, L)  — local_dist[i,j] = 1 - dot(query[:,i], host[:,j])
    local_dist = 1.0 - np.dot(query_chroma.T, host_chroma)
    local_dist = np.clip(local_dist, 0.0, 2.0)

    # 2. Accumulated cost matrix D and path-length matrix P
    #    D[i,j] = accumulated cost of best warping path from (0, j_start) to (i, j)
    #    P[i,j] = number of steps along that best path (used to normalize D for fair comparison)
    D = np.full((N, L), np.inf)
    P = np.zeros((N, L), dtype=np.int32)

    # Parent pointer matrix for backtracking:
    #  0 = diagonal (i-1, j-1)  |  1 = insertion (i-1, j)  |  2 = deletion (i, j-1)
    parent = np.zeros((N, L), dtype=np.int8)

    # First row: subsequence DTW — the query can begin at any host position for free
    D[0, :] = local_dist[0, :]
    P[0, :] = 1
    # j_origin[i,j] = the host column where the optimal path reaching (i,j) began.
    # Tracking this lets us compute the host SPAN of each path: j - j_origin[N-1,j] + 1
    # and enforce a minimum-span constraint when selecting j_end.
    j_origin = np.zeros((N, L), dtype=np.int32)
    j_origin[0, :] = np.arange(L)  # each path in row 0 starts at its own column

    # Dynamic programming — vectorised row-by-row for speed
    for i in range(1, N):
        # Three predecessor options for each j:
        #   diag  = D[i-1, j-1]  (match step, counts as 1 step)
        #   ins   = D[i-1, j]    (query advances, host stays — "insertion")
        #   del_  = D[i, j-1]    (host advances, query stays — "deletion")
        # Standard step costs: diagonal=1, vertical/horizontal=1  (no extra penalty)
        # We use equal step weights so path length is meaningful for normalisation.

        diag = np.full(L, np.inf)
        p_diag = np.zeros(L, dtype=np.int32)
        diag[1:] = D[i-1, :-1]
        p_diag[1:] = P[i-1, :-1]

        ins = D[i-1, :]
        p_ins = P[i-1, :]

        del_ = np.full(L, np.inf)
        p_del = np.zeros(L, dtype=np.int32)
        del_[1:] = D[i, :-1]   # D[i, j-1] — must be filled left-to-right
        p_del[1:] = P[i, :-1]

        # We cannot vectorise the deletion (left-to-right dependency),
        # so we do a single scalar pass only for the deletion update:
        # Build D[i] in two sweeps:
        # Sweep 1: pick best of diag and ins (no left-dependency)
        # Sweep 2: propagate deletion left-to-right

        # Sweep 1 — best of diagonal and insertion
        stack_costs = np.stack([diag, ins], axis=0)      # (2, L)
        stack_paths = np.stack([p_diag, p_ins], axis=0)  # (2, L)
        best_idx_1 = np.argmin(stack_costs, axis=0)       # (L,)
        best_cost_1 = stack_costs[best_idx_1, np.arange(L)]
        best_path_1 = stack_paths[best_idx_1, np.arange(L)]
        parent_1 = best_idx_1.astype(np.int8)  # 0=diag, 1=ins

        # Propagate j_origin from sweep-1 winners
        j_origin_diag = np.zeros(L, dtype=np.int32)
        j_origin_diag[1:] = j_origin[i-1, :-1]
        j_origin_ins = j_origin[i-1, :]
        stack_origins = np.stack([j_origin_diag, j_origin_ins], axis=0)  # (2, L)
        origin_1 = stack_origins[best_idx_1, np.arange(L)]

        D[i]        = local_dist[i] + best_cost_1
        P[i]        = best_path_1 + 1
        parent[i]   = parent_1
        j_origin[i] = origin_1

        # Sweep 2 — left-to-right deletion propagation
        for j in range(1, L):
            del_cost = D[i, j-1]
            del_p    = P[i, j-1]
            if del_cost < D[i, j]:
                D[i, j]        = local_dist[i, j] + del_cost
                P[i, j]        = del_p + 1
                parent[i, j]   = 2  # deletion
                j_origin[i, j] = j_origin[i, j-1]  # origin inherited from left neighbour

    # 3. Find the best end position with optional minimum host-span constraint
    #
    #    Normalise by path length so position-independent average cost is compared.
    #    Then mask to positions whose HOST SPAN (j - j_origin + 1) >= min_span_frames.
    #    Without the span mask, a tiny 4-frame perfect match always wins over a longer
    #    region that is slightly weaker — the span constraint forces the algorithm to
    #    find a musically substantial match, not just the tightest one.
    last_row_norm = D[N-1, :] / np.maximum(P[N-1, :], 1)
    span_last_row = np.arange(L) - j_origin[N-1, :] + 1  # host frames covered by each path

    if min_span_frames > 0 and (span_last_row >= min_span_frames).any():
        mask = span_last_row >= min_span_frames
        # argmin within the valid set; map back to full index space
        j_end = int(np.where(mask)[0][np.argmin(last_row_norm[mask])])
    else:
        j_end = int(np.argmin(last_row_norm))
    min_cost = float(D[N-1, j_end])
    path_len = int(P[N-1, j_end])

    # 4. Backtrack using parent-pointer matrix
    i, j = N - 1, j_end
    path = []
    while i >= 0:
        path.append([i, j])
        if i == 0:
            break  # Reached the first query row — stop (subsequence start found)
        p = parent[i, j]
        if p == 0:    # diagonal
            i, j = i - 1, j - 1
        elif p == 1:  # insertion
            i, j = i - 1, j
        else:         # deletion
            i, j = i, j - 1

    path.reverse()  # chronological order
    j_start = path[0][1]

    score = min_cost / max(path_len, 1)

    return {
        "cost_matrix": D.tolist(),
        "warping_path": path,
        "start_frame": j_start,
        "end_frame": j_end,
        "score": score
    }
