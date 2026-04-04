/**
 * TrailerPlayer.tsx
 * =================
 * Cinematic fullscreen trailer overlay for MovieBuzz.
 *
 * HOW TO USE — add this comment wherever your Trailer button lives:
 *
 *   // ── TRAILER BUTTON ──────────────────────────────────────────────────────
 *   // Import TrailerPlayer at the top of this file:
 *   //   import TrailerPlayer from '@/components/TrailerPlayer'
 *   //
 *   // Add state near your other useState hooks:
 *   //   const [trailerMovieId, setTrailerMovieId] = useState<number | null>(null)
 *   //
 *   // Replace / update your existing trailer button with:
 *   //   <button onClick={() => setTrailerMovieId(movie.movieId)} ...>
 *   //     ▶ Trailer
 *   //   </button>
 *   //
 *   // Add the overlay anywhere inside your return JSX (e.g. just before </div>):
 *   //   <TrailerPlayer
 *   //     movieId={trailerMovieId}
 *   //     onClose={() => setTrailerMovieId(null)}
 *   //   />
 *   // ── END TRAILER BUTTON ──────────────────────────────────────────────────
 *
 * PROPS:
 *   movieId  — MovieLens movieId (number) or null (null = player is closed)
 *   onClose  — callback to set movieId back to null
 *
 * FEATURES:
 *   - Calls  GET /api/trailer/:movieId  on open
 *   - Shows  loading shimmer  while fetching
 *   - Plays  YouTube iframe  at full viewport size with cinematic letterbox
 *   - Escape key  or  click outside  closes the overlay
 *   - Graceful fallback: if no video_id returned, shows YouTube search button
 *   - Dark vignette backdrop with blur
 *   - Smooth open/close CSS transitions (no extra deps needed)
 */

import { useEffect, useState, useRef, useCallback } from "react";
import type { MouseEvent } from "react";
import { getMovieTrailer, type TrailerResponse } from "../lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────
type TrailerData = TrailerResponse;

interface TrailerPlayerProps {
  movieId: number | null;
  onClose: () => void;
}

// ─── API call ─────────────────────────────────────────────────────────────────
async function fetchTrailer(movieId: number): Promise<TrailerData> {
  return getMovieTrailer(movieId);
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function TrailerPlayer({ movieId, onClose }: TrailerPlayerProps) {
  const [data,    setData]    = useState<TrailerData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const [visible, setVisible] = useState(false);   // controls CSS enter/exit transition

  const overlayRef = useRef<HTMLDivElement>(null);
  const iframeRef  = useRef<HTMLIFrameElement>(null);

  // ── Fetch trailer when movieId changes ──────────────────────────────────────
  useEffect(() => {
    if (movieId === null) {
      // Trigger exit animation, then clear data
      setVisible(false);
      const t = setTimeout(() => {
        setData(null);
        setError(null);
      }, 350); // matches CSS transition duration
      return () => clearTimeout(t);
    }

    setLoading(true);
    setError(null);
    setData(null);
    setVisible(true);   // start enter animation immediately

    fetchTrailer(movieId)
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [movieId]);

  // ── Escape key closes overlay ────────────────────────────────────────────────
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === "Escape") onClose();
  }, [onClose]);

  useEffect(() => {
    if (movieId !== null) {
      window.addEventListener("keydown", handleKeyDown);
      // Prevent background scroll while overlay is open
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [movieId, handleKeyDown]);

  // ── Click outside iframe closes overlay ─────────────────────────────────────
  const handleOverlayClick = (e: MouseEvent<HTMLDivElement>) => {
    if (e.target === overlayRef.current) onClose();
  };

  // ── Don't render DOM at all when fully closed ────────────────────────────────
  if (movieId === null && !visible) return null;

  // ── YouTube search fallback URL ──────────────────────────────────────────────
  const ytSearchUrl = data
    ? `https://www.youtube.com/results?search_query=${encodeURIComponent(
        `${data.title} ${data.year ?? ""} official trailer`
      )}`
    : null;

  return (
    <>
      {/* ── Inject styles (scoped to .mb-trailer-*) ── */}
      <style>{STYLES}</style>

      {/* ── Backdrop overlay ── */}
      <div
        ref={overlayRef}
        className={`mb-trailer-overlay ${visible ? "mb-trailer-overlay--in" : "mb-trailer-overlay--out"}`}
        onClick={handleOverlayClick}
        role="dialog"
        aria-modal="true"
        aria-label="Movie trailer"
      >
        {/* ── Player container (same-sized as viewport, letterboxed) ── */}
        <div className={`mb-trailer-container ${visible ? "mb-trailer-container--in" : ""}`}>

          {/* ── Header bar ── */}
          <div className="mb-trailer-header">
            <div className="mb-trailer-title">
              {loading ? (
                <span className="mb-trailer-shimmer mb-trailer-shimmer--title" />
              ) : (
                <>
                  <span className="mb-trailer-label">TRAILER</span>
                  {data && (
                    <span className="mb-trailer-movie-name">
                      {data.title}
                      {data.year && <span className="mb-trailer-year"> ({data.year})</span>}
                    </span>
                  )}
                </>
              )}
            </div>

            {/* Close button */}
            <button
              className="mb-trailer-close"
              onClick={onClose}
              aria-label="Close trailer"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M3 3l14 14M17 3L3 17" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
              </svg>
            </button>
          </div>

          {/* ── Main player area ── */}
          <div className="mb-trailer-player-wrap">

            {/* Loading shimmer */}
            {loading && (
              <div className="mb-trailer-loading">
                <div className="mb-trailer-shimmer mb-trailer-shimmer--player" />
                <div className="mb-trailer-spinner">
                  <div className="mb-trailer-spinner-ring" />
                  <span className="mb-trailer-spinner-text">Loading trailer…</span>
                </div>
              </div>
            )}

            {/* Error state */}
            {!loading && error && (
              <div className="mb-trailer-fallback">
                <div className="mb-trailer-fallback-icon">⚠</div>
                <p className="mb-trailer-fallback-msg">Could not load trailer</p>
                <p className="mb-trailer-fallback-sub">{error}</p>
                {ytSearchUrl && (
                  <a
                    href={ytSearchUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mb-trailer-yt-btn"
                  >
                    Search on YouTube ↗
                  </a>
                )}
              </div>
            )}

            {/* No video_id from OMDB — graceful fallback */}
            {!loading && !error && data && !data.found && (
              <div className="mb-trailer-fallback">
                <div className="mb-trailer-fallback-icon">🎬</div>
                <p className="mb-trailer-fallback-msg">No trailer link in our database</p>
                <p className="mb-trailer-fallback-sub">
                  We couldn't find a direct trailer for <strong>{data.title}</strong>.
                </p>
                {ytSearchUrl && (
                  <a
                    href={ytSearchUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mb-trailer-yt-btn"
                  >
                    ▶&nbsp; Search YouTube for this trailer
                  </a>
                )}
              </div>
            )}

            {/* ── YouTube iframe — the main event ── */}
            {/* This iframe takes the full player area, 16:9 letterboxed */}
            {!loading && !error && data && data.found && data.embed_url && (
              <iframe
                ref={iframeRef}
                className="mb-trailer-iframe"
                src={data.embed_url}
                title={`${data.title} trailer`}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
              />
            )}
          </div>

          {/* ── Footer ── */}
          <div className="mb-trailer-footer">
            <span className="mb-trailer-footer-hint">Press ESC or click outside to close</span>
            {data?.found && (
              <a
                href={`https://www.youtube.com/watch?v=${data.video_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="mb-trailer-footer-yt"
              >
                Watch on YouTube ↗
              </a>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
// All scoped to .mb-trailer-* — won't conflict with anything in your app.
const STYLES = `
  /* ── Overlay backdrop ── */
  .mb-trailer-overlay {
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: opacity 0.35s ease, backdrop-filter 0.35s ease;
  }
  .mb-trailer-overlay--out { opacity: 0; pointer-events: none; }
  .mb-trailer-overlay--in  { opacity: 1; }

  /* ── Container: same width as window, 16:9 height, max 90vh ── */
  .mb-trailer-container {
    position: relative;
    width: 100vw;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    background: #0a0a0a;
    border-top: 1px solid rgba(255,255,255,0.08);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    transform: translateY(32px) scale(0.98);
    opacity: 0;
    transition: transform 0.4s cubic-bezier(0.22,1,0.36,1),
                opacity   0.35s ease;
  }
  .mb-trailer-container--in {
    transform: translateY(0) scale(1);
    opacity: 1;
  }

  /* ── Header ── */
  .mb-trailer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    flex-shrink: 0;
  }
  .mb-trailer-title {
    display: flex;
    align-items: center;
    gap: 12px;
    overflow: hidden;
  }
  .mb-trailer-label {
    font-family: 'Courier New', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: #e50914;
    background: rgba(229,9,20,0.12);
    border: 1px solid rgba(229,9,20,0.35);
    padding: 3px 8px;
    border-radius: 3px;
    flex-shrink: 0;
  }
  .mb-trailer-movie-name {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 16px;
    font-weight: 400;
    color: #f0f0f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: 0.01em;
  }
  .mb-trailer-year {
    color: #888;
    font-size: 14px;
  }

  /* ── Close button ── */
  .mb-trailer-close {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    color: #ccc;
    cursor: pointer;
    transition: background 0.2s, color 0.2s, transform 0.2s;
    flex-shrink: 0;
  }
  .mb-trailer-close:hover {
    background: rgba(229,9,20,0.2);
    border-color: rgba(229,9,20,0.5);
    color: #fff;
    transform: rotate(90deg);
  }

  /* ── Player wrap: 16:9 letterbox taking remaining height ── */
  .mb-trailer-player-wrap {
    position: relative;
    flex: 1;
    /* 16:9 aspect ratio via padding trick — capped by container height */
    aspect-ratio: 16 / 9;
    max-height: calc(90vh - 100px); /* header + footer */
    background: #000;
    overflow: hidden;
  }

  /* ── YouTube iframe: fills player wrap exactly ── */
  .mb-trailer-iframe {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    border: none;
    display: block;
  }

  /* ── Loading state ── */
  .mb-trailer-loading {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 20px;
  }
  .mb-trailer-spinner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .mb-trailer-spinner-ring {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(255,255,255,0.1);
    border-top-color: #e50914;
    border-radius: 50%;
    animation: mb-spin 0.8s linear infinite;
  }
  @keyframes mb-spin { to { transform: rotate(360deg); } }
  .mb-trailer-spinner-text {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #666;
    letter-spacing: 0.1em;
  }

  /* ── Shimmer skeleton ── */
  .mb-trailer-shimmer {
    background: linear-gradient(
      90deg,
      rgba(255,255,255,0.04) 0%,
      rgba(255,255,255,0.09) 50%,
      rgba(255,255,255,0.04) 100%
    );
    background-size: 200% 100%;
    animation: mb-shimmer 1.4s ease-in-out infinite;
    border-radius: 4px;
    display: block;
  }
  .mb-trailer-shimmer--title {
    width: 220px;
    height: 18px;
  }
  .mb-trailer-shimmer--player {
    position: absolute;
    inset: 0;
    border-radius: 0;
  }
  @keyframes mb-shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  /* ── Fallback (no trailer found) ── */
  .mb-trailer-fallback {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 32px;
    text-align: center;
  }
  .mb-trailer-fallback-icon {
    font-size: 40px;
    margin-bottom: 4px;
    filter: grayscale(0.4);
  }
  .mb-trailer-fallback-msg {
    font-family: Georgia, serif;
    font-size: 18px;
    color: #ddd;
    margin: 0;
  }
  .mb-trailer-fallback-sub {
    font-size: 13px;
    color: #666;
    margin: 0;
  }
  .mb-trailer-fallback-sub strong { color: #999; }

  /* ── YouTube search fallback button ── */
  .mb-trailer-yt-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
    padding: 10px 22px;
    background: #e50914;
    color: #fff;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-decoration: none;
    border-radius: 4px;
    transition: background 0.2s, transform 0.15s;
  }
  .mb-trailer-yt-btn:hover {
    background: #c0060f;
    transform: translateY(-1px);
  }

  /* ── Footer ── */
  .mb-trailer-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    border-top: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
  }
  .mb-trailer-footer-hint {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #444;
    letter-spacing: 0.06em;
  }
  .mb-trailer-footer-yt {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #666;
    text-decoration: none;
    letter-spacing: 0.06em;
    transition: color 0.2s;
  }
  .mb-trailer-footer-yt:hover { color: #e50914; }

  /* ── Mobile: stack vertically, smaller header ── */
  @media (max-width: 640px) {
    .mb-trailer-movie-name { font-size: 13px; }
    .mb-trailer-header     { padding: 10px 14px; }
    .mb-trailer-footer     { padding: 8px 14px; }
    .mb-trailer-footer-hint { display: none; }
  }
`;
