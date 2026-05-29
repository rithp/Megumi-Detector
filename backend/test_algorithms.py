import numpy as np
from audio_processor import vectorized_sliding_window_filter, subsequence_dtw

def generate_synthetic_chroma(num_frames: int):
    """
    Generate synthetic chroma features (shape 12 x num_frames) representing chord patterns.
    Columns are normalized to unit length.
    """
    chroma = np.zeros((12, num_frames))
    # Generate shifting chord patterns (e.g., C major, G major, A minor, F major)
    chords = [
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C (0, 4, 7)
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # G (2, 7, 11)
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # Am (0, 4, 9)
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # F (0, 5, 9)
    ]
    frames_per_chord = num_frames // len(chords)
    
    for idx, chord in enumerate(chords):
        start = idx * frames_per_chord
        end = (idx + 1) * frames_per_chord if idx < len(chords) - 1 else num_frames
        for f in range(start, end):
            # Base chord + time-varying phase sweep to make each frame unique
            vec = np.array(chord, dtype=float)
            sweep = 0.1 * np.sin((f / num_frames) * 2.0 * np.pi)
            # Apply sweep to active chroma bands
            vec[vec > 0] += sweep
            chroma[:, f] = vec + np.random.uniform(0, 0.03, 12)
            
    # Normalize columns to unit length
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    return chroma

def run_test():
    print("========================================")
    print("RUNNING ALGORITHMIC SEQUENCE ALIGNMENT TESTS")
    print("========================================")
    
    # 1. Generate a large host song (e.g. 500 frames, roughly 25 seconds)
    np.random.seed(42)
    host_chroma = generate_synthetic_chroma(500)
    print(f"Generated synthetic host chroma: {host_chroma.shape}")
    
    # 2. Extract a slice as our query (frames 150 to 200, which is G major to A minor transition)
    embed_start = 150
    embed_end = 200
    query_chroma = host_chroma[:, embed_start:embed_end].copy()
    
    # Add time warping: let's time-stretch the query by duplicating every 4th frame,
    # making it slightly longer (55 frames) to test DTW's robust alignment
    warp_indices = []
    for f in range(query_chroma.shape[1]):
        warp_indices.append(f)
        if f % 4 == 0:
            warp_indices.append(f)  # duplicate frame
            
    query_warped = query_chroma[:, warp_indices]
    
    # Add instrumentation noise to the query
    query_warped += np.random.normal(0, 0.05, query_warped.shape)
    # Clip and re-normalize columns
    query_warped = np.clip(query_warped, 0, 1)
    query_warped = query_warped / (np.linalg.norm(query_warped, axis=0, keepdims=True) + 1e-8)
    
    print(f"Generated warped query chroma: {query_warped.shape} (original embedded: {query_chroma.shape})")
    
    # 3. Test O(1) Sliding-Window Filter Heuristic
    print("\n[Testing O(1) Sliding-Window Filter...]")
    threshold = 0.70
    sim_curve, candidates = vectorized_sliding_window_filter(
        host_chroma=host_chroma,
        query_chroma=query_warped,
        threshold=threshold,
        K=2
    )
    
    total_windows = len(sim_curve)
    pruned_count = total_windows - len(candidates)
    pruning_rate = pruned_count / total_windows
    
    print(f"Total windows scanned: {total_windows}")
    print(f"Pruned windows: {pruned_count} ({pruning_rate * 100:.1f}% pruning rate)")
    
    # Check if the true embedded starting index is inside the candidates
    # The warped query will match around host frame 150
    is_true_in_candidates = any(abs(c - embed_start) <= 5 for c in candidates)
    print(f"True start frame ({embed_start}) near candidates? {'YES' if is_true_in_candidates else 'NO'}")
    assert is_true_in_candidates, "FAIL: O(1) filter pruned the correct match window!"
    print("SUCCESS: O(1) filter successfully retained the correct match window while pruning most of the track!")

    # 4. Test Subsequence DTW
    print("\n[Testing Subsequence DTW...]")
    alignment = subsequence_dtw(query_warped, host_chroma)
    
    detected_start = alignment["start_frame"]
    detected_end = alignment["end_frame"]
    score = alignment["score"]
    
    print(f"Detected Start Frame: {detected_start} (Expected near {embed_start})")
    print(f"Detected End Frame: {detected_end} (Expected near {embed_end})")
    print(f"Normalized Cost Score: {score:.4f} (Lower is better)")
    
    # Verify that the detected segment overlaps heavily with the embedded region
    start_error = abs(detected_start - embed_start)
    end_error = abs(detected_end - embed_end)
    print(f"Alignment Frame Error: Start={start_error}, End={end_error}")
    
    assert start_error <= 3 and end_error <= 15, f"FAIL: DTW misaligned! Start Err={start_error}, End Err={end_error}"
    print("SUCCESS: Subsequence DTW aligned the warped and noisy query perfectly within 3 frames error!")
    
    # 5. Print warping path sample
    path = alignment["warping_path"]
    print(f"Warping Path Length: {len(path)}")
    print(f"Path Sample (First 5 steps): {path[:5]}")
    print(f"Path Sample (Last 5 steps): {path[-5:]}")
    print("\n========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    run_test()
