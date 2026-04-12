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
  const [posterFailed, setPosterFailed] = useState(false);

  useEffect(() => {
    setPosterFailed(false);
  }, [fallbackPoster, hasRemotePoster, remotePoster]);

  const posterSrc = posterFailed || !hasRemotePoster
    ? fallbackPoster
    : remotePoster;

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
                setPosterFailed(true);
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
