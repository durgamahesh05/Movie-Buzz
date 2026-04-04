import { useEffect, useState } from "react";
import { Bookmark, BookmarkCheck, ExternalLink } from "lucide-react";
import type { Movie } from "../lib/api";
import { createMoviePosterDataUrl } from "../lib/movie-fallbacks";

type Props = {
  movie: Movie;
  theme?: "dark" | "light";
  isWishlisted?: boolean;
  onToggleWishlist?: (movie: Movie) => void;
  onOpenDetails?: (movie: Movie) => void;
};

export default function MovieCard({
  movie,
  theme = "dark",
  isWishlisted = false,
  onToggleWishlist = () => {},
  onOpenDetails = () => {},
}: Props) {
  const isLight = theme === "light";
  const fallbackPoster = createMoviePosterDataUrl(
    movie.clean_title || movie.title,
    movie.year || "",
    movie.genres || "",
  );
  const remotePoster = movie.poster?.trim() || "";
  const hasRemotePoster = Boolean(remotePoster) && remotePoster !== fallbackPoster;
  const [posterSrc, setPosterSrc] = useState(
    hasRemotePoster ? fallbackPoster : (remotePoster || fallbackPoster),
  );
  const [posterLoading, setPosterLoading] = useState(hasRemotePoster);

  useEffect(() => {
    let isCancelled = false;
    let timeoutId: number | null = null;
    const clearPosterTimeout = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    setPosterSrc(fallbackPoster);
    setPosterLoading(hasRemotePoster);

    if (!hasRemotePoster) {
      return () => {
        isCancelled = true;
      };
    }

    const image = new Image();
    image.referrerPolicy = "no-referrer";
    image.onload = () => {
      if (isCancelled) {
        return;
      }
      clearPosterTimeout();
      setPosterSrc(remotePoster);
      setPosterLoading(false);
    };
    image.onerror = () => {
      if (isCancelled) {
        return;
      }
      clearPosterTimeout();
      setPosterSrc(fallbackPoster);
      setPosterLoading(false);
    };
    image.src = remotePoster;

    timeoutId = window.setTimeout(() => {
      if (isCancelled) {
        return;
      }
      setPosterSrc(fallbackPoster);
      setPosterLoading(false);
    }, 3500);

    return () => {
      isCancelled = true;
      clearPosterTimeout();
    };
  }, [fallbackPoster, hasRemotePoster, remotePoster]);

  return (
    <article className={`movie-card${isLight ? " movie-card--light" : ""}`}>
      <button
        type="button"
        className="movie-card__poster-button"
        onClick={() => onOpenDetails(movie)}
        aria-label={`Open details for ${movie.title}`}
      >
        <div className="movie-card__poster-shell">
          {posterSrc ? (
            <img
              src={posterSrc}
              alt={movie.title}
              className="movie-poster"
              loading="lazy"
              referrerPolicy="no-referrer"
              onError={() => {
                setPosterSrc(fallbackPoster);
                setPosterLoading(false);
              }}
            />
          ) : (
            <div className="no-poster">No Image</div>
          )}
        </div>
      </button>

      <div className="movie-info">
        <button
          type="button"
          className="movie-card__title-button"
          onClick={() => onOpenDetails(movie)}
        >
          <h3>{movie.clean_title || movie.title}</h3>
        </button>

        <p>
          ⭐ {movie.rating ?? "N/A"}
          {movie.year ? ` · ${movie.year}` : ""}
        </p>

        <div className="movie-card__actions">
          <button
            type="button"
            className={`movie-card__action${
              isWishlisted ? " movie-card__action--active" : ""
            }`}
            onClick={() => onToggleWishlist(movie)}
          >
            {isWishlisted ? <BookmarkCheck size={14} /> : <Bookmark size={14} />}
            {isWishlisted ? "Wishlisted" : "Wishlist"}
          </button>
          <button
            type="button"
            className="movie-card__action"
            onClick={() => onOpenDetails(movie)}
          >
            <ExternalLink size={14} />
            Details
          </button>
        </div>
      </div>
    </article>
  );
}
