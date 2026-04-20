import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
} from "react";

import { getMovieTrailer, type TrailerResponse } from "../lib/api";

type TrailerData = TrailerResponse;

interface TrailerPlayerProps {
  movieId: number | null;
  fallbackUrl?: string | null;
  onClose: () => void;
}

const trailerCache = new Map<number, TrailerData>();
const trailerRequests = new Map<number, Promise<TrailerData>>();

async function fetchTrailer(movieId: number): Promise<TrailerData> {
  const cached = trailerCache.get(movieId);
  if (cached) {
    return cached;
  }

  const inflight = trailerRequests.get(movieId);
  if (inflight) {
    return inflight;
  }

  const request = getMovieTrailer(movieId)
    .then((data) => {
      trailerCache.set(movieId, data);
      trailerRequests.delete(movieId);
      return data;
    })
    .catch((error) => {
      trailerRequests.delete(movieId);
      throw error;
    });

  trailerRequests.set(movieId, request);
  return request;
}

export default function TrailerPlayer({
  movieId,
  fallbackUrl = null,
  onClose,
}: TrailerPlayerProps) {
  const [data, setData] = useState<TrailerData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);
  const fallbackHandledForMovieRef = useRef<number | null>(null);

  useEffect(() => {
    let active = true;
    fallbackHandledForMovieRef.current = null;

    if (movieId === null) {
      setVisible(false);
      const timeoutId = window.setTimeout(() => {
        if (!active) {
          return;
        }
        setData(null);
        setError(null);
        setLoading(false);
      }, 220);

      return () => {
        active = false;
        window.clearTimeout(timeoutId);
      };
    }

    setVisible(true);
    setLoading(true);
    setError(null);
    setData(null);

    void fetchTrailer(movieId)
      .then((response) => {
        if (!active) {
          return;
        }
        setData(response);
      })
      .catch((fetchError: unknown) => {
        if (!active) {
          return;
        }
        setError(
          fetchError instanceof Error
            ? fetchError.message
            : "Unable to load the trailer right now",
        );
        setData(null);
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [movieId]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    },
    [onClose],
  );

  useEffect(() => {
    if (movieId === null) {
      return undefined;
    }

    window.addEventListener("keydown", handleKeyDown);
    document.body.style.overflow = "hidden";

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [handleKeyDown, movieId]);

  const handleOverlayClick = (event: MouseEvent<HTMLDivElement>) => {
    if (event.target === overlayRef.current) {
      onClose();
    }
  };

  const youtubeSearchUrl = useMemo(() => {
    if (!data) {
      return null;
    }
    const searchText = `${data.title} ${data.year ?? ""} official trailer`.trim();
    return `https://www.youtube.com/results?search_query=${encodeURIComponent(searchText)}`;
  }, [data]);

  const youtubeWatchUrl = useMemo(() => {
    if (!data?.video_id) {
      return null;
    }
    return `https://www.youtube.com/watch?v=${data.video_id}`;
  }, [data?.video_id]);

  const navigationFallbackUrl = youtubeWatchUrl || fallbackUrl || youtubeSearchUrl;

  const openInCurrentTab = useCallback((url: string) => {
    try {
      const openedWindow = window.open(url, "_self");
      if (!openedWindow) {
        window.location.assign(url);
      }
    } catch {
      window.location.assign(url);
    }
  }, []);

  useEffect(() => {
    if (movieId === null || loading || !navigationFallbackUrl) {
      return;
    }

    const shouldFallback =
      Boolean(error) || (Boolean(data) && (!data.found || !data.embed_url));

    if (!shouldFallback || fallbackHandledForMovieRef.current === movieId) {
      return;
    }

    fallbackHandledForMovieRef.current = movieId;
    onClose();
    openInCurrentTab(navigationFallbackUrl);
  }, [data, error, loading, movieId, navigationFallbackUrl, onClose, openInCurrentTab]);

  if (movieId === null && !visible) {
    return null;
  }

  return (
    <>
      <style>{STYLES}</style>
      <div
        ref={overlayRef}
        className={`trailer-overlay ${visible ? "trailer-overlay--open" : ""}`}
        onClick={handleOverlayClick}
        role="dialog"
        aria-modal="true"
        aria-label="Movie trailer"
      >
        <div className={`trailer-shell ${visible ? "trailer-shell--open" : ""}`}>
          <div className="trailer-header">
            <div className="trailer-copy">
              <span className="trailer-chip">Trailer</span>
              <div className="trailer-heading">
                <strong>{data?.title || "Loading trailer"}</strong>
                {data?.year ? <span>({data.year})</span> : null}
              </div>
            </div>
            <button
              type="button"
              className="trailer-close"
              onClick={onClose}
              aria-label="Close trailer"
            >
              x
            </button>
          </div>

          <div className="trailer-stage">
            {loading ? (
              <div className="trailer-state">
                <div className="trailer-spinner" />
                <p>Loading trailer...</p>
              </div>
            ) : null}

            {!loading && error ? (
              <div className="trailer-state">
                <p>Could not load the trailer.</p>
                <span>{error}</span>
                {youtubeSearchUrl ? (
                  <a
                    href={youtubeSearchUrl}
                    target="_self"
                    className="trailer-link"
                  >
                    Search on YouTube
                  </a>
                ) : null}
              </div>
            ) : null}

            {!loading && !error && data && !data.found ? (
              <div className="trailer-state">
                <p>No direct trailer found for this movie.</p>
                <span>Try the YouTube search fallback.</span>
                {youtubeSearchUrl ? (
                  <a
                    href={youtubeSearchUrl}
                    target="_self"
                    className="trailer-link"
                  >
                    Search on YouTube
                  </a>
                ) : null}
              </div>
            ) : null}

            {!loading && !error && data?.found && data.embed_url ? (
              <iframe
                className="trailer-iframe"
                src={data.embed_url}
                title={`${data.title} trailer`}
                loading="lazy"
                referrerPolicy="strict-origin-when-cross-origin"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
                onError={() => {
                  setError("The embedded trailer could not be loaded.");
                }}
              />
            ) : null}
          </div>

          <div className="trailer-footer">
            <span>Press Esc or click outside to close</span>
            {data?.video_id ? (
              <a
                href={`https://www.youtube.com/watch?v=${data.video_id}`}
                target="_self"
                className="trailer-link trailer-link--subtle"
              >
                Open on YouTube
              </a>
            ) : null}
          </div>
        </div>
      </div>
    </>
  );
}

const STYLES = `
  .trailer-overlay {
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background: rgba(0, 0, 0, 0.78);
    backdrop-filter: blur(10px);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
  }

  .trailer-overlay--open {
    opacity: 1;
    pointer-events: auto;
  }

  .trailer-shell {
    width: min(1200px, 100%);
    max-height: min(92vh, 860px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    background: #09090b;
    color: #f4f4f5;
    transform: translateY(12px);
    transition: transform 0.2s ease;
  }

  .trailer-shell--open {
    transform: translateY(0);
  }

  .trailer-header,
  .trailer-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 16px;
    background: rgba(255, 255, 255, 0.03);
  }

  .trailer-copy {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }

  .trailer-chip {
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(220, 38, 38, 0.16);
    color: #fca5a5;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .trailer-heading {
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 0;
    font-size: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .trailer-heading strong {
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .trailer-heading span {
    color: #a1a1aa;
    flex-shrink: 0;
  }

  .trailer-close {
    width: 36px;
    height: 36px;
    border: none;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    color: #f4f4f5;
    cursor: pointer;
    font-size: 18px;
    line-height: 1;
  }

  .trailer-stage {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #000;
  }

  .trailer-iframe {
    width: 100%;
    height: 100%;
    border: 0;
    display: block;
  }

  .trailer-state {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 24px;
    text-align: center;
    color: #e4e4e7;
  }

  .trailer-state span {
    color: #a1a1aa;
    font-size: 13px;
  }

  .trailer-spinner {
    width: 38px;
    height: 38px;
    border-radius: 999px;
    border: 3px solid rgba(255, 255, 255, 0.14);
    border-top-color: #f87171;
    animation: trailer-spin 0.8s linear infinite;
  }

  .trailer-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 10px 14px;
    border-radius: 8px;
    background: #dc2626;
    color: #fff;
    text-decoration: none;
    font-size: 13px;
    font-weight: 700;
  }

  .trailer-link--subtle {
    background: transparent;
    color: #fca5a5;
    padding: 0;
  }

  .trailer-footer {
    color: #a1a1aa;
    font-size: 12px;
  }

  @keyframes trailer-spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (max-width: 640px) {
    .trailer-overlay {
      padding: 12px;
    }

    .trailer-header,
    .trailer-footer {
      padding: 12px;
    }

    .trailer-heading {
      font-size: 13px;
    }

    .trailer-footer {
      flex-direction: column;
      align-items: flex-start;
    }
  }
`;
