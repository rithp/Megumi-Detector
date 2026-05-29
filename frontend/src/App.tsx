import React, { useState, useEffect, useRef } from 'react';
import { 
  Music, 
  Search, 
  Upload, 
  Play, 
  Pause, 
  HelpCircle, 
  Zap, 
  Activity, 
  RefreshCw, 
  Volume2
} from 'lucide-react';

interface Track {
  id: string;
  title: string;
  url: string;
  duration: number;
  heatmap: Array<{ start_time: number; end_time: number; value: number }>;
  has_features: boolean;
}

interface MatchResult {
  best_match: {
    start_time: number;
    end_time: number;
    score: number;
    confidence: number;
    query_start_time?: number;
    query_end_time?: number;
  };
  pruning_stats: {
    total_windows: number;
    pruned_windows: number;
    pruning_rate: number;
  };
  similarity_curve: {
    times: number[];
    similarities: number[];
  };
  warping_path: Array<{
    query_frame: number;
    query_time: number;
    host_frame: number;
    host_time: number;
  }>;
  cost_matrix: {
    matrix: number[][];
    host_times: number[];
    query_times: number[];
  };
  secondary_matches: Array<{
    frame: number;
    start_time: number;
    end_time: number;
    similarity: number;
    query_start_time?: number;
    query_end_time?: number;
  }>;
  is_auto_mode?: boolean;
}

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [hostId, setHostId] = useState<string>('');
  const [queryId, setQueryId] = useState<string>('');
  
  // URL Input State
  const [urlInput, setUrlInput] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string>('');
  
  // File Upload State
  const [uploadTitle, setUploadTitle] = useState<string>('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string>('');
  
  // Query crop parameters
  const [cropStart, setCropStart] = useState<number>(0);
  const [cropEnd, setCropEnd] = useState<number>(10);
  
  // Match Settings
  const [threshold, setThreshold] = useState<number>(0.72);
  const [subSegments, setSubSegments] = useState<number>(2);
  const [autoDiscover, setAutoDiscover] = useState<boolean>(false);
  
  // Match Status & Results
  const [isMatching, setIsMatching] = useState<boolean>(false);
  const [matchResult, setMatchResult] = useState<MatchResult | null>(null);
  const [matchError, setMatchError] = useState<string>('');
  
  // Audio Refs & Playback States
  const hostAudioRef = useRef<HTMLAudioElement | null>(null);
  const queryAudioRef = useRef<HTMLAudioElement | null>(null);
  const [isHostPlaying, setIsHostPlaying] = useState(false);
  const [isQueryPlaying, setIsQueryPlaying] = useState(false);
  const [hostProgress, setHostProgress] = useState(0);
  const [queryProgress, setQueryProgress] = useState(0);
  
  // Matrix Canvas Ref
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ queryTime: number; hostTime: number; cost: number } | null>(null);

  // Selected secondary match (null = use best_match from Kadane)
  const [selectedPeakIdx, setSelectedPeakIdx] = useState<number | null>(null);

  // Derive the currently active match region (selectedPeak overrides best_match)
  const activePeak = matchResult
    ? selectedPeakIdx !== null && matchResult.secondary_matches[selectedPeakIdx]
      ? { start_time: matchResult.secondary_matches[selectedPeakIdx].start_time,
          end_time:   matchResult.secondary_matches[selectedPeakIdx].end_time,
          query_start_time: matchResult.secondary_matches[selectedPeakIdx].query_start_time,
          query_end_time:   matchResult.secondary_matches[selectedPeakIdx].query_end_time }
      : { start_time: matchResult.best_match.start_time,
          end_time:   matchResult.best_match.end_time,
          query_start_time: matchResult.best_match.query_start_time,
          query_end_time:   matchResult.best_match.query_end_time }
    : null;

  // Active query region: in auto mode use the matched query segment, otherwise use crop
  const activeQueryStart = (matchResult?.is_auto_mode && activePeak?.query_start_time !== undefined)
    ? activePeak.query_start_time!
    : cropStart;
  const activeQueryEnd = (matchResult?.is_auto_mode && activePeak?.query_end_time !== undefined)
    ? activePeak.query_end_time!
    : cropEnd;

  // Fetch saved tracks
  const fetchTracks = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/tracks`);
      if (res.ok) {
        const data = await res.json();
        setTracks(data);
        if (data.length > 0) {
          // Default selection if available
          if (!hostId) setHostId(data[0].id);
          if (!queryId && data.length > 1) setQueryId(data[1].id);
        }
      }
    } catch (e) {
      console.error("Error loading tracks", e);
    }
  };

  useEffect(() => {
    fetchTracks();
  }, []);

  // Handle URL Analysis
  const handleAnalyzeUrl = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!urlInput.trim()) return;
    setIsAnalyzing(true);
    setAnalysisError('');
    try {
      const res = await fetch(`${API_BASE}/api/tracks/analyze-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlInput })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "URL analysis failed");
      }
      setUrlInput('');
      await fetchTracks();
    } catch (err: any) {
      setAnalysisError(err.message || "An unexpected error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle File Upload
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile || !uploadTitle.trim()) return;
    setIsUploading(true);
    setUploadError('');
    
    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('title', uploadTitle);

    try {
      const res = await fetch(`${API_BASE}/api/tracks/upload`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Audio upload failed");
      }
      setUploadTitle('');
      setUploadFile(null);
      await fetchTracks();
    } catch (err: any) {
      setUploadError(err.message || "An unexpected error occurred");
    } finally {
      setIsUploading(false);
    }
  };

  // Run Local Sequence Alignment Matching
  const handleRunMatch = async () => {
    if (!hostId || !queryId) return;
    setIsMatching(true);
    setMatchResult(null);
    setMatchError('');
    
    // Reset playbacks
    if (hostAudioRef.current) hostAudioRef.current.pause();
    if (queryAudioRef.current) queryAudioRef.current.pause();
    setIsHostPlaying(false);
    setIsQueryPlaying(false);
    setSelectedPeakIdx(null);  // reset to best match when re-running

    try {
      const res = await fetch(`${API_BASE}/api/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          host_id: hostId,
          query_id: queryId,
          threshold: parseFloat(threshold.toString()),
          sub_segments: parseInt(subSegments.toString()),
          crop_start: autoDiscover ? null : parseFloat(cropStart.toString()),
          crop_end:   autoDiscover ? null : parseFloat(cropEnd.toString()),
          auto_discover: autoDiscover,
        })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Matching computation failed");
      }
      const data = await res.json();
      setMatchResult(data);
    } catch (err: any) {
      setMatchError(err.message || "An error occurred during matching");
    } finally {
      setIsMatching(false);
    }
  };

  // Auto crop settings if query has a heatmap peak
  const selectHeatmapPeak = (start: number, end: number) => {
    setCropStart(start);
    setCropEnd(end);
  };

  // Find track objects
  const selectedHost = tracks.find(t => t.id === hostId);
  const selectedQuery = tracks.find(t => t.id === queryId);

  // Audio Playback Sync Helpers
  const toggleHostPlay = () => {
    if (!hostAudioRef.current) return;
    const host = hostAudioRef.current;
    if (isHostPlaying) {
      host.pause();
      setIsHostPlaying(false);
    } else {
      if (activePeak) {
        const start = activePeak.start_time;
        const end   = activePeak.end_time;
        if (host.currentTime < start || host.currentTime > end) {
          host.currentTime = start;
        }
      }
      host.play();
      setIsHostPlaying(true);
    }
  };

  const toggleQueryPlay = () => {
    if (!queryAudioRef.current) return;
    const query = queryAudioRef.current;
    if (isQueryPlaying) {
      query.pause();
      setIsQueryPlaying(false);
    } else {
      if (query.currentTime < activeQueryStart || query.currentTime > activeQueryEnd) {
        query.currentTime = activeQueryStart;
      }
      query.play();
      setIsQueryPlaying(true);
    }
  };

  const playMatchSideBySide = () => {
    if (!hostAudioRef.current || !queryAudioRef.current || !activePeak) return;
    
    // Stop any current play
    hostAudioRef.current.pause();
    queryAudioRef.current.pause();
    
    // Seek host to the ACTIVE peak's start (best match or selected secondary)
    hostAudioRef.current.currentTime = activePeak.start_time;
    // Seek query to cropped start (or matched segment in auto-discovery)
    queryAudioRef.current.currentTime = activeQueryStart;
    
    hostAudioRef.current.play();
    queryAudioRef.current.play();
    
    setIsHostPlaying(true);
    setIsQueryPlaying(true);
  };

  // Click a secondary match row: update highlight + seek both audio tracks
  const selectPeak = (idx: number) => {
    const peak = matchResult?.secondary_matches[idx];
    if (!peak) return;
    setSelectedPeakIdx(idx);
    // Seek host audio to the selected peak
    if (hostAudioRef.current) {
      hostAudioRef.current.pause();
      hostAudioRef.current.currentTime = peak.start_time;
      setIsHostPlaying(false);
    }
    // In auto-discovery mode, also seek query audio to the matched query segment
    if (queryAudioRef.current && peak.query_start_time !== undefined) {
      queryAudioRef.current.pause();
      queryAudioRef.current.currentTime = peak.query_start_time;
      setIsQueryPlaying(false);
    }
  };

  // Sync seek timers
  useEffect(() => {
    const host = hostAudioRef.current;
    const query = queryAudioRef.current;
    
    const updateHostProgress = () => {
      if (host) {
        setHostProgress(host.currentTime / (host.duration || 1));
        // Stop playback when we reach the end of the active peak's region
        if (activePeak && !host.paused) {
          if (host.currentTime >= activePeak.end_time) {
            host.pause();
            host.currentTime = activePeak.start_time;
            setIsHostPlaying(false);
          }
        }
      }
    };
    
    const updateQueryProgress = () => {
      if (query) {
        setQueryProgress(query.currentTime / (query.duration || 1));
        // Bound query playback to the active region
        if (!query.paused) {
          const qEnd = activeQueryEnd;
          if (query.currentTime >= qEnd) {
            query.pause();
            query.currentTime = activeQueryStart;
            setIsQueryPlaying(false);
          }
        }
      }
    };
    
    if (host) {
      host.addEventListener('timeupdate', updateHostProgress);
      host.addEventListener('ended', () => setIsHostPlaying(false));
    }
    if (query) {
      query.addEventListener('timeupdate', updateQueryProgress);
      query.addEventListener('ended', () => setIsQueryPlaying(false));
    }
    
    return () => {
      if (host) {
        host.removeEventListener('timeupdate', updateHostProgress);
      }
      if (query) {
        query.removeEventListener('timeupdate', updateQueryProgress);
      }
    };
  }, [hostId, queryId, matchResult, cropStart, cropEnd, isHostPlaying, isQueryPlaying, activePeak, activeQueryStart, activeQueryEnd]);

  // Draw DTW Cost Matrix Canvas
  useEffect(() => {
    if (!matchResult || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { matrix, host_times } = matchResult.cost_matrix;
    const path = matchResult.warping_path;
    
    // Guard against empty matrix
    if (!matrix || matrix.length === 0 || !matrix[0]) {
      canvas.width = 400;
      canvas.height = 150;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.font = '14px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      const msg = matchResult?.is_auto_mode
        ? 'Alignment details not shown in Auto-Discovery Mode'
        : 'Alignment matrix unavailable (Query is too long or matching region is near the end)';
      ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
      return;
    }
    
    const rows = matrix.length; // Query length (N)
    const cols = matrix[0].length; // Subsampled Host length (L')
    
    // Canvas Sizing
    const cellWidth = 3;
    const cellHeight = 3;
    canvas.width = cols * cellWidth;
    canvas.height = rows * cellHeight;
    
    // 1. Find cost range for normalization
    let minCost = Infinity;
    let maxCost = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = matrix[r][c];
        if (val < minCost) minCost = val;
        if (val > maxCost) maxCost = val;
      }
    }
    const costRange = (maxCost - minCost) || 1.0;
    
    // 2. Draw Cost Matrix Cell by Cell
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = matrix[r][c];
        // Normalize cost to [0, 1]
        const normVal = (val - minCost) / costRange;
        
        // HSL Color Scale: cool deep purple-blue for low cost, bright magenta-red for high cost
        const hue = 260 - normVal * 200; // 260 (purple) down to 60 (yellow/amber)
        const sat = 90;
        const light = 15 + normVal * 40; // dark to light
        
        ctx.fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`;
        // Draw flipped vertically so index 0 is at the bottom (standard math grid representation)
        ctx.fillRect(c * cellWidth, (rows - 1 - r) * cellHeight, cellWidth, cellHeight);
      }
    }
    
    // 3. Draw Warping Path as a neon line
    ctx.strokeStyle = 'hsl(38, 100%, 55%)'; // Neon Gold/Amber
    ctx.lineWidth = 3;
    ctx.shadowColor = 'rgba(255, 165, 0, 0.6)';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    
    path.forEach((pt, idx) => {
      // Find matching column in subsampled matrix
      const hostTime = pt.host_time;
      // Find closest index in subsampled host_times
      let closestColIdx = 0;
      let minDiff = Infinity;
      for (let c = 0; c < host_times.length; c++) {
        const diff = Math.abs(host_times[c] - hostTime);
        if (diff < minDiff) {
          minDiff = diff;
          closestColIdx = c;
        }
      }
      
      const r = pt.query_frame;
      const x = closestColIdx * cellWidth + cellWidth / 2;
      const y = (rows - 1 - r) * cellHeight + cellHeight / 2;
      
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.shadowBlur = 0; // reset
    
  }, [matchResult]);

  // Handle Canvas Mouse Hover for frame inspections
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!matchResult || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const { matrix, host_times, query_times } = matchResult.cost_matrix;
    if (!matrix || matrix.length === 0 || !matrix[0]) return;
    
    const rows = matrix.length;
    const cols = matrix[0].length;
    
    // Compute hovered cell indices
    const cellWidth = rect.width / cols;
    const cellHeight = rect.height / rows;
    
    const colIdx = Math.floor(x / cellWidth);
    const rowIdx = rows - 1 - Math.floor(y / cellHeight); // flip back
    
    if (colIdx >= 0 && colIdx < cols && rowIdx >= 0 && rowIdx < rows) {
      setHoveredCell({
        queryTime: query_times[rowIdx],
        hostTime: host_times[colIdx],
        cost: matrix[rowIdx][colIdx]
      });
    }
  };

  const handleCanvasMouseLeave = () => {
    setHoveredCell(null);
  };

  return (
    <div className="grid-container">
      {/* Header Banner */}
      <header className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'linear-gradient(135deg, hsla(263, 90%, 62%, 0.1), hsla(187, 100%, 45%, 0.05))' }}>
        <div>
          <h1 className="glow-text" style={{ fontSize: '28px', color: 'white', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Music style={{ color: 'hsl(var(--color-secondary))' }} /> Who Sampled Engine
          </h1>
          <p style={{ color: 'hsl(var(--text-secondary))', marginTop: '6px', fontSize: '15px' }}>
            Sub-Sequence Audio Alignment Search utilizing localized <b>Chroma CENS</b> and $O(1)$ cumulative filters.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <span className="card" style={{ padding: '8px 12px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', background: 'hsla(0, 0%, 100%, 0.05)' }}>
            <Activity size={16} style={{ color: 'hsl(var(--color-secondary))' }} /> Heuristic: O(1)
          </span>
          <span className="card" style={{ padding: '8px 12px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px', background: 'hsla(0, 0%, 100%, 0.05)' }}>
            <Zap size={16} style={{ color: 'hsl(var(--color-accent))' }} /> Subsequence DTW
          </span>
        </div>
      </header>

      {/* Main Form Dashboard */}
      <main style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '24px' }}>
        
        {/* Track Library Manager */}
        <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h2 style={{ fontSize: '20px', borderBottom: '1px solid hsl(var(--border-color))', paddingBottom: '10px' }}>
            1. Track Manager
          </h2>
          
          {/* Analyze YouTube URL */}
          <form onSubmit={handleAnalyzeUrl} className="form-group" id="url-form">
            <label className="form-label">Analyze YouTube Track</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input 
                type="url" 
                placeholder="https://www.youtube.com/watch?v=..." 
                className="form-input" 
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                style={{ flex: 1 }}
                required
                id="url-input"
              />
              <button 
                type="submit" 
                className="glow-btn" 
                disabled={isAnalyzing}
                style={{ padding: '12px' }}
                id="btn-analyze"
              >
                {isAnalyzing ? <RefreshCw className="pulse-glow" size={18} /> : <Search size={18} />}
              </button>
            </div>
            {analysisError && <span style={{ color: 'hsl(var(--color-danger))', fontSize: '13px' }}>{analysisError}</span>}
          </form>

          {/* Upload Local Audio */}
          <form onSubmit={handleUpload} className="form-group" style={{ borderTop: '1px dashed hsl(var(--border-color))', paddingTop: '15px' }} id="upload-form">
            <label className="form-label">Upload Local Audio</label>
            <input 
              type="text" 
              placeholder="Track Title" 
              className="form-input" 
              value={uploadTitle}
              onChange={(e) => setUploadTitle(e.target.value)}
              required
              id="upload-title"
              style={{ marginBottom: '8px' }}
            />
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <input 
                type="file" 
                accept="audio/*" 
                onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
                required
                style={{ fontSize: '13px', color: 'hsl(var(--text-secondary))' }}
                id="upload-file"
              />
              <button 
                type="submit" 
                className="glow-btn" 
                disabled={isUploading}
                style={{ padding: '8px 16px', fontSize: '14px' }}
                id="btn-upload"
              >
                {isUploading ? <RefreshCw className="pulse-glow" size={14} /> : <Upload size={14} />} Upload
              </button>
            </div>
            {uploadError && <span style={{ color: 'hsl(var(--color-danger))', fontSize: '13px' }}>{uploadError}</span>}
          </form>

          {/* List of Analyzed Tracks */}
          <div style={{ flex: 1, marginTop: '10px' }}>
            <label className="form-label" style={{ marginBottom: '8px', display: 'block' }}>Analyzed Tracks ({tracks.length})</label>
            <div style={{ maxHeight: '180px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {tracks.length === 0 ? (
                <div style={{ color: 'hsl(var(--text-muted))', fontSize: '13px', padding: '20px', textAlign: 'center', border: '1px dashed hsl(var(--border-color))', borderRadius: '6px' }}>
                  No tracks in database. Analyze a YouTube link above!
                </div>
              ) : (
                tracks.map(t => (
                  <div key={t.id} className="card" style={{ padding: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'hsla(0, 0%, 100%, 0.03)', borderRadius: '6px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <span style={{ fontSize: '14px', fontWeight: '500', maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {t.title}
                      </span>
                      <span style={{ fontSize: '11px', color: 'hsl(var(--text-muted))' }}>
                        {Math.floor(t.duration / 60)}m {Math.floor(t.duration % 60)}s • {t.url === "Uploaded File" ? "Upload" : "YouTube"}
                      </span>
                    </div>
                    {t.heatmap && t.heatmap.length > 0 && (
                      <span style={{ fontSize: '11px', padding: '3px 6px', background: 'hsla(187, 100%, 45%, 0.1)', color: 'hsl(var(--color-secondary))', borderRadius: '4px' }}>
                        Heatmap Loaded
                      </span>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </section>

        {/* Local Sequence Search Dashboard */}
        <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h2 style={{ fontSize: '20px', borderBottom: '1px solid hsl(var(--border-color))', paddingBottom: '10px' }}>
            2. Sequence Alignment Console
          </h2>

          {/* Selection Dropdowns */}
          <div className="form-group">
            <label className="form-label">Host Track (Large Song)</label>
            <select 
              className="form-input" 
              value={hostId} 
              onChange={(e) => setHostId(e.target.value)}
              id="host-select"
            >
              <option value="" disabled>Select Host Song</option>
              {tracks.map(t => (
                <option key={t.id} value={t.id}>{t.title} ({Math.floor(t.duration / 60)}m {Math.floor(t.duration % 60)}s)</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Sample/Query Track (Obscure source / Short Loop)</label>
            <select 
              className="form-input" 
              value={queryId} 
              onChange={(e) => setQueryId(e.target.value)}
              id="query-select"
            >
              <option value="" disabled>Select Query Song</option>
              {tracks.map(t => (
                <option key={t.id} value={t.id}>{t.title} ({Math.floor(t.duration / 60)}m {Math.floor(t.duration % 60)}s)</option>
              ))}
            </select>
          </div>

          {/* YouTube Heatmap Segment Picker */}
          {selectedQuery && selectedQuery.heatmap && selectedQuery.heatmap.length > 0 && (
            <div style={{ border: '1px solid hsla(187, 100%, 45%, 0.2)', padding: '12px', borderRadius: '8px', background: 'hsla(187, 100%, 45%, 0.02)' }}>
              <label className="form-label" style={{ fontSize: '13px', color: 'hsl(var(--color-secondary))', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Activity size={14} /> YouTube "Most Replayed" Peak Loops
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {selectedQuery.heatmap.map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => selectHeatmapPeak(p.start_time, p.end_time)}
                    className="card"
                    style={{
                      padding: '6px 10px',
                      fontSize: '11px',
                      background: cropStart === p.start_time ? 'hsla(187, 100%, 45%, 0.15)' : 'hsla(0, 0%, 100%, 0.03)',
                      borderColor: cropStart === p.start_time ? 'hsl(var(--color-secondary))' : 'hsl(var(--border-color))',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    Peak {idx + 1}: {p.start_time}s - {p.end_time}s (Int: {Math.round(p.value * 100)}%)
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Query Crop Interval — hidden in auto-discover mode */}
          {!autoDiscover && (
            <div style={{ display: 'flex', gap: '16px' }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="form-label">Crop Start Time (seconds)</label>
                <input 
                  type="number" 
                  step="0.1" 
                  min="0"
                  max={selectedQuery ? selectedQuery.duration : 100}
                  className="form-input" 
                  value={cropStart} 
                  onChange={(e) => setCropStart(parseFloat(e.target.value || '0'))}
                  id="crop-start"
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="form-label">Crop End Time (seconds)</label>
                <input 
                  type="number" 
                  step="0.1" 
                  min={cropStart}
                  max={selectedQuery ? selectedQuery.duration : 100}
                  className="form-input" 
                  value={cropEnd} 
                  onChange={(e) => setCropEnd(parseFloat(e.target.value || '10'))}
                  id="crop-end"
                />
              </div>
            </div>
          )}

          {/* Auto-Discovery Toggle */}
          <div
            onClick={() => setAutoDiscover(v => !v)}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && setAutoDiscover(v => !v)}
            style={{
              display: 'flex', alignItems: 'center', gap: '12px',
              padding: '12px 16px',
              borderRadius: '8px',
              border: '1px solid',
              borderColor: autoDiscover ? 'hsl(var(--color-secondary))' : 'hsl(var(--border-color))',
              background: autoDiscover ? 'hsla(187,100%,45%,0.08)' : 'hsla(0,0%,100%,0.02)',
              cursor: 'pointer',
              userSelect: 'none',
              transition: 'all 0.2s ease',
            }}
          >
            {/* Toggle pill */}
            <div style={{
              width: '40px', height: '22px', borderRadius: '11px',
              background: autoDiscover ? 'hsl(var(--color-secondary))' : 'hsl(var(--border-color))',
              position: 'relative', flexShrink: 0,
              transition: 'background 0.2s ease',
            }}>
              <div style={{
                position: 'absolute', top: '3px',
                left: autoDiscover ? '21px' : '3px',
                width: '16px', height: '16px', borderRadius: '50%',
                background: 'white',
                transition: 'left 0.2s ease',
                boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
              }} />
            </div>
            <div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: autoDiscover ? 'hsl(var(--color-secondary))' : 'hsl(var(--text-primary))' }}>
                Auto-Discovery Mode
              </div>
              <div style={{ fontSize: '11px', color: 'hsl(var(--text-muted))', marginTop: '2px' }}>
                {autoDiscover
                  ? 'Searching full tracks using 20×20 segment grid — no crop needed'
                  : 'Enable to skip crop selection and automatically find the best matching regions'}
              </div>
            </div>
          </div>

          {/* Algorithm tuning parameters */}
          <div style={{ display: 'flex', gap: '16px', borderTop: '1px solid hsl(var(--border-color))', paddingTop: '15px' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>O(1) Filter Threshold</span>
                <span style={{ color: 'hsl(var(--color-secondary))', fontWeight: 'bold' }}>{threshold}</span>
              </label>
              <input 
                type="range" 
                min="0.5" 
                max="0.95" 
                step="0.01" 
                className="form-input" 
                value={threshold} 
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ padding: '0' }}
                id="threshold-range"
              />
            </div>
            
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Sub-segments (K)</label>
              <select 
                className="form-input" 
                value={subSegments} 
                onChange={(e) => setSubSegments(parseInt(e.target.value))}
                id="subsegments-select"
              >
                <option value="2">2 Halves (Fast/Robust)</option>
                <option value="3">3 Sections (Fine Harmonic)</option>
                <option value="4">4 Quarters (Highly Precise)</option>
              </select>
            </div>
          </div>

          <button 
            type="button" 
            className="glow-btn pulse-glow" 
            onClick={handleRunMatch}
            disabled={isMatching || !hostId || !queryId}
            style={{ width: '100%', padding: '16px', justifyContent: 'center', fontSize: '17px', marginTop: '10px' }}
            id="btn-match"
          >
            {isMatching ? (
              <>
                <RefreshCw className="pulse-glow" size={20} /> Compiling Alignment Matrices...
              </>
            ) : (
              <>
                <Activity size={20} /> Run Sub-Sequence Alignment
              </>
            )}
          </button>
          
          {matchError && <span style={{ color: 'hsl(var(--color-danger))', fontSize: '13px', textAlign: 'center' }}>{matchError}</span>}
        </section>
      </main>

      {/* Matching Results & Visualization Panels */}
      {matchResult && (
        <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '24px', animation: 'fadeIn 0.5s ease-out' }}>
          
          {/* Results Summary Box */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', borderBottom: '1px solid hsl(var(--border-color))', paddingBottom: '20px' }}>
            <div className="card" style={{ padding: '16px', background: 'hsla(38, 100%, 50%, 0.05)', borderColor: 'hsl(var(--color-accent))' }}>
              <span style={{ fontSize: '12px', color: 'hsl(var(--color-accent))', textTransform: 'uppercase', fontWeight: 'bold' }}>Detected Sample Segment</span>
              <h3 style={{ fontSize: '24px', margin: '6px 0', color: 'white' }}>
                {Math.floor(matchResult.best_match.start_time / 60)}m {Math.floor(matchResult.best_match.start_time % 60)}s
                <span style={{ fontSize: '15px', color: 'hsl(var(--text-secondary))' }}> to </span>
                {Math.floor(matchResult.best_match.end_time / 60)}m {Math.floor(matchResult.best_match.end_time % 60)}s
              </h3>
              <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))' }}>
                Warped Loop Duration: {Math.round((matchResult.best_match.end_time - matchResult.best_match.start_time) * 10) / 10}s
              </p>
            </div>
            
            <div className="card" style={{ padding: '16px', background: 'hsla(187, 100%, 45%, 0.05)', borderColor: 'hsl(var(--color-secondary))' }}>
              <span style={{ fontSize: '12px', color: 'hsl(var(--color-secondary))', textTransform: 'uppercase', fontWeight: 'bold' }}>O(1) Pruning Rate</span>
              <h3 style={{ fontSize: '24px', margin: '6px 0', color: 'white' }}>
                {Math.round(matchResult.pruning_stats.pruning_rate * 1000) / 10}%
              </h3>
              <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))' }}>
                Pruned {matchResult.pruning_stats.pruned_windows} of {matchResult.pruning_stats.total_windows} windows
              </p>
            </div>

            <div className="card" style={{ padding: '16px', background: 'hsla(263, 90%, 62%, 0.05)', borderColor: 'hsl(var(--color-primary))' }}>
              <span style={{ fontSize: '12px', color: 'hsl(var(--color-primary))', textTransform: 'uppercase', fontWeight: 'bold' }}>Confidence Rating</span>
              <h3 style={{ fontSize: '24px', margin: '6px 0', color: 'white' }}>
                {Math.round(matchResult.best_match.confidence * 100)}%
              </h3>
              <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))' }}>
                Normalized Cost Matrix Score: {(matchResult.best_match.score).toFixed(3)}
              </p>
            </div>
          </div>

          {/* Interactive Wave Player */}
          <div className="card" style={{ background: 'hsla(0, 0%, 0%, 0.25)', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Volume2 style={{ color: 'hsl(var(--color-secondary))' }} /> Audio Comparison Sandbox
              </h3>
              <button 
                type="button" 
                className="glow-btn" 
                onClick={playMatchSideBySide}
                style={{ fontSize: '13px', padding: '8px 16px', background: 'hsl(var(--color-accent))' }}
                id="btn-play-match"
              >
                <Play size={14} /> Play Aligned Sample Loop
              </button>
            </div>

            {/* Hidden HTML5 Audio Tags */}
            <audio ref={hostAudioRef} src={`${API_BASE}/api/tracks/${hostId}/audio`} />
            <audio ref={queryAudioRef} src={`${API_BASE}/api/tracks/${queryId}/audio`} />

            {/* Host Audio Waveform Controller */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: 'hsl(var(--text-secondary))' }}>
                <span>Host Track: <b>{selectedHost?.title}</b></span>
                <span>{hostAudioRef.current ? `${Math.floor(hostAudioRef.current.currentTime / 60)}m ${String(Math.floor(hostAudioRef.current.currentTime % 60)).padStart(2, '0')}s / ${Math.floor((selectedHost?.duration || 0) / 60)}m ${String(Math.floor((selectedHost?.duration || 0) % 60)).padStart(2, '0')}s` : '0:00'}</span>
              </div>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <button 
                  onClick={toggleHostPlay} 
                  className="glow-btn" 
                  style={{ padding: '8px', borderRadius: '50%' }}
                >
                  {isHostPlaying ? <Pause size={14} /> : <Play size={14} />}
                </button>
                <div style={{ flex: 1, height: '8px', background: 'hsl(var(--border-color))', borderRadius: '4px', position: 'relative', overflow: 'hidden', cursor: 'pointer' }}
                  onClick={(e) => {
                    if (hostAudioRef.current) {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const ratio = (e.clientX - rect.left) / rect.width;
                      hostAudioRef.current.currentTime = ratio * hostAudioRef.current.duration;
                    }
                  }}
                >
                  {/* Progress Line */}
                  <div style={{ position: 'absolute', top: 0, left: 0, height: '100%', width: `${hostProgress * 100}%`, background: 'hsl(var(--color-secondary))' }} />
                  {/* Matching Highlight Zone — tracks the currently selected peak */}
                  {selectedHost && activePeak && (
                    <div style={{ 
                      position: 'absolute', 
                      top: 0, 
                      left: `${(activePeak.start_time / selectedHost.duration) * 100}%`, 
                      width: `${((activePeak.end_time - activePeak.start_time) / selectedHost.duration) * 100}%`, 
                      height: '100%', 
                      background: selectedPeakIdx !== null ? 'hsla(187, 100%, 45%, 0.25)' : 'hsla(38, 100%, 50%, 0.3)',
                      borderLeft: `2px solid ${selectedPeakIdx !== null ? 'hsl(var(--color-secondary))' : 'hsl(var(--color-accent))'}`,
                      borderRight: `2px solid ${selectedPeakIdx !== null ? 'hsl(var(--color-secondary))' : 'hsl(var(--color-accent))'}`,
                      transition: 'left 0.3s ease, width 0.3s ease'
                    }} />
                  )}
                </div>
              </div>
            </div>

            {/* Query Audio Waveform Controller */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', borderTop: '1px dashed hsl(var(--border-color))', paddingTop: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: 'hsl(var(--text-secondary))' }}>
                <span>Source Loop: <b>{selectedQuery?.title}</b></span>
                <span>{queryAudioRef.current ? `${Math.floor(queryAudioRef.current.currentTime / 60)}m ${String(Math.floor(queryAudioRef.current.currentTime % 60)).padStart(2, '0')}s / ${Math.floor((selectedQuery?.duration || 0) / 60)}m ${String(Math.floor((selectedQuery?.duration || 0) % 60)).padStart(2, '0')}s` : '0:00'}</span>
              </div>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <button 
                  onClick={toggleQueryPlay} 
                  className="glow-btn" 
                  style={{ padding: '8px', borderRadius: '50%', background: 'linear-gradient(135deg, hsl(var(--color-primary)), hsl(var(--color-primary)))' }}
                >
                  {isQueryPlaying ? <Pause size={14} /> : <Play size={14} />}
                </button>
                <div style={{ flex: 1, height: '8px', background: 'hsl(var(--border-color))', borderRadius: '4px', position: 'relative', overflow: 'hidden', cursor: 'pointer' }}
                  onClick={(e) => {
                    if (queryAudioRef.current) {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const ratio = (e.clientX - rect.left) / rect.width;
                      queryAudioRef.current.currentTime = ratio * queryAudioRef.current.duration;
                    }
                  }}
                >
                  {/* Progress Line */}
                  <div style={{ position: 'absolute', top: 0, left: 0, height: '100%', width: `${queryProgress * 100}%`, background: 'hsl(var(--color-primary))' }} />
                  {/* Crop / Auto-Discovery Highlight on Query Waveform */}
                  {selectedQuery && (
                    <div style={{ 
                      position: 'absolute', 
                      top: 0, 
                      left: `${(activeQueryStart / selectedQuery.duration) * 100}%`, 
                      width: `${((activeQueryEnd - activeQueryStart) / selectedQuery.duration) * 100}%`, 
                      height: '100%', 
                      background: matchResult?.is_auto_mode
                        ? (selectedPeakIdx !== null ? 'hsla(187,100%,45%,0.25)' : 'hsla(38,100%,50%,0.25)')
                        : 'hsla(263, 90%, 62%, 0.3)',
                      borderLeft: `2px solid ${matchResult?.is_auto_mode ? 'hsl(var(--color-secondary))' : 'hsl(var(--color-primary))'}`,
                      borderRight: `2px solid ${matchResult?.is_auto_mode ? 'hsl(var(--color-secondary))' : 'hsl(var(--color-primary))'}`,
                      transition: 'left 0.3s ease, width 0.3s ease'
                    }} />
                  )}
                </div>
              </div>
            </div>

          </div>

          {/* Alignment Visualizations (Canvas Matrix & Secondary Matches) */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '24px' }}>
            
            {/* 2D Canvas DTW alignment map */}
            <div className="matrix-container" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '6px', alignSelf: 'flex-start' }}>
                <Activity size={16} style={{ color: 'hsl(var(--color-accent))' }} /> Dynamic Time Warping Matrix & Warping Path
              </h3>
              
              <div style={{ position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' }}>
                <canvas 
                  ref={canvasRef} 
                  onMouseMove={handleCanvasMouseMove}
                  onMouseLeave={handleCanvasMouseLeave}
                  className="matrix-canvas" 
                  style={{ border: '1px solid hsl(var(--border-color))', cursor: 'crosshair' }}
                  id="dtw-canvas"
                />
                
                {hoveredCell && (
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'rgba(10, 15, 25, 0.85)',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: '1px solid hsl(var(--border-color))',
                    fontSize: '11px',
                    color: 'white',
                    backdropFilter: 'blur(4px)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '2px',
                    pointerEvents: 'none'
                  }}>
                    <span>Host Time: <b>{(hoveredCell.hostTime).toFixed(1)}s</b></span>
                    <span>Query Time: <b>{(hoveredCell.queryTime).toFixed(1)}s</b></span>
                    <span style={{ color: 'hsl(var(--color-secondary))' }}>Cum. Cost: <b>{(hoveredCell.cost).toFixed(2)}</b></span>
                  </div>
                )}
              </div>
              <p style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', textAlign: 'center', width: '100%' }}>
                X-Axis represents Host Song frames (subsampled), Y-Axis represents Query Song frames. Neon yellow line shows the optimal alignment path.
              </p>
            </div>

            {/* Similarity Plot & Secondary Matches */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              
              {/* O(1) Sliding Window Similarity Peaks */}
              <div className="card" style={{ padding: '16px', background: 'hsla(0, 0%, 0%, 0.2)' }}>
                <h4 style={{ fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '12px' }}>
                  <Zap size={14} style={{ color: 'hsl(var(--color-secondary))' }} /> Sliding Window Similarity Filter
                </h4>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {matchResult.secondary_matches.length === 0 ? (
                    <div style={{ fontSize: '12px', color: 'hsl(var(--text-muted))', padding: '10px', textAlign: 'center' }}>
                      No alternative regions passed the $O(1)$ filter threshold.
                    </div>
                  ) : (
                    matchResult.secondary_matches.map((peak, idx) => {
                      const isSelected = selectedPeakIdx === idx;
                      const isBest = idx === 0 && selectedPeakIdx === null;
                      const isActive = isSelected || isBest;
                      return (
                        <div 
                          key={idx}
                          onClick={() => selectPeak(idx)}
                          role="button"
                          tabIndex={0}
                          onKeyDown={(e) => e.key === 'Enter' && selectPeak(idx)}
                          style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between', 
                            alignItems: 'center', 
                            padding: '10px 12px', 
                            background: isActive
                              ? isSelected
                                ? 'hsla(187, 100%, 45%, 0.12)'
                                : 'hsla(38, 100%, 50%, 0.08)'
                              : 'hsla(0, 0%, 100%, 0.02)',
                            border: '1px solid',
                            borderColor: isActive
                              ? isSelected
                                ? 'hsla(187, 100%, 45%, 0.5)'
                                : 'hsla(38, 100%, 50%, 0.3)'
                              : 'hsl(var(--border-color))',
                            borderRadius: '6px',
                            fontSize: '13px',
                            cursor: 'pointer',
                            transition: 'background 0.15s ease, border-color 0.15s ease',
                            userSelect: 'none'
                          }}
                          onMouseEnter={e => {
                            if (!isActive) (e.currentTarget as HTMLElement).style.background = 'hsla(0,0%,100%,0.05)';
                          }}
                          onMouseLeave={e => {
                            if (!isActive) (e.currentTarget as HTMLElement).style.background = 'hsla(0,0%,100%,0.02)';
                          }}
                        >
                          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                            <span style={{ 
                              fontSize: '11px', 
                              padding: '2px 6px', 
                              background: isActive
                                ? isSelected
                                  ? 'hsl(var(--color-secondary))'
                                  : 'hsl(var(--color-accent))'
                                : 'hsl(var(--bg-card))', 
                              color: isActive ? 'black' : 'white', 
                              borderRadius: '3px',
                              fontWeight: 'bold' 
                            }}>
                              {idx === 0 && selectedPeakIdx === null ? 'BEST' : isSelected ? '▶ ACTIVE' : `#${idx + 1}`}
                            </span>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
                              <span>
                                Host: {Math.floor(peak.start_time / 60)}m {Math.floor(peak.start_time % 60)}s
                                {' – '}
                                {Math.floor(peak.end_time / 60)}m {Math.floor(peak.end_time % 60)}s
                              </span>
                              {/* Show query segment for auto-discovery pairs */}
                              {peak.query_start_time !== undefined && (
                                <span style={{ fontSize: '11px', color: 'hsl(var(--text-muted))' }}>
                                  Query: {Math.floor(peak.query_start_time / 60)}m {Math.floor(peak.query_start_time % 60)}s
                                  {' – '}
                                  {Math.floor((peak.query_end_time ?? 0) / 60)}m {Math.floor((peak.query_end_time ?? 0) % 60)}s
                                </span>
                              )}
                            </div>
                          </div>
                          <span style={{ fontWeight: '600', color: 'hsl(var(--color-secondary))' }}>
                            {Math.round(peak.similarity * 100)}% Match
                          </span>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Informative Help Guide Card */}
              <div className="card" style={{ padding: '16px', background: 'hsla(263, 90%, 62%, 0.03)', borderColor: 'hsla(263, 90%, 62%, 0.15)' }}>
                <h4 style={{ fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', color: 'hsl(var(--color-primary))' }}>
                  <HelpCircle size={14} /> Understanding the Science
                </h4>
                <p style={{ fontSize: '12px', color: 'hsl(var(--text-secondary))', lineHeight: '1.5' }}>
                  The <b>$O(1)$ sub-segment filter</b> divides the query's spectral shape into sub-windows and performs integral-table lookups on the host track. This isolates high-correlation musical frames instantly. 
                  Once candidate zones are isolated, the <b>Subsequence DTW</b> aligns time-warped and tempo-shifted chords by weaving an optimal path through a dynamic programming distance grid.
                </p>
              </div>

            </div>

          </div>

        </section>
      )}
      
      <footer style={{ textAlign: 'center', padding: '24px 0', borderTop: '1px solid hsl(var(--border-color))', marginTop: '32px', color: 'hsl(var(--text-muted))', fontSize: '13px' }}>
        Who Sampled Local Sequence Audio Engine • Built with React & FastAPI • 2026
      </footer>
    </div>
  );
}
