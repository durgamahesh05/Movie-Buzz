import { useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Bookmark,
  ChevronDown,
  ExternalLink,
  LogOut,
  Moon,
  Search,
  Sun,
  Trash2,
  UserRound,
  X,
} from "lucide-react";
import { Footer } from "./footer";
import MovieCard from "./movie-card";
import TrailerPlayer from "./TrailerPlayer";
import {
  addWishlistMovie,
  ApiError,
  confirmDeleteAccount,
  getHomeMovies,
  getWishlist,
  recommendCatalogMovies,
  removeWishlistMovie,
  requestDeleteAccountOtp,
  searchCatalogMovies,
  type Movie,
} from "../lib/api";
import {
  HOME_CACHE_INVALIDATED_EVENT,
  getHomeCacheKey,
  isHomeCacheRefreshKey,
} from "../lib/movie-cache";
import {
  createMoviePosterDataUrl,
  enrichMovie,
  enrichMovies,
  getRecommendedMovies,
  movieMatchesGenre,
  searchMoviesLocally,
} from "../lib/movie-fallbacks";
import { useAppStore } from "../store/appStore";
import { useNavigate } from "react-router-dom";
import { MovieBuzzLogo } from "./moviebuzz-logo";

const GENRES = [
  "All",
  "Action",
  "Comedy",
  "Drama",
  "Sci-Fi",
  "Thriller",
  "Horror",
  "Romance",
  "Animation",
  "Fantasy",
  "Crime",
];

const WISHLIST_CACHE_PREFIX = "moviebuzz-wishlist-";

type DeleteStep = "request" | "confirm";

interface DropdownItemProps {
  icon: ReactNode;
  iconBg: string;
  iconColor: string;
  label: string;
  textColor: string;
  hoverBg: string;
  onClick: () => void;
}

function DropdownItem({
  icon,
  iconBg,
  iconColor,
  label,
  textColor,
  hoverBg,
  onClick,
}: DropdownItemProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: "100%",
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "8px 10px",
        borderRadius: "12px",
        border: "none",
        backgroundColor: hovered ? hoverBg : "transparent",
        cursor: "pointer",
        transition: "all 0.15s ease",
        transform: hovered ? "scale(1.02)" : "scale(1)",
        textAlign: "left",
      }}
    >
      <span
        style={{
          width: "28px",
          height: "28px",
          borderRadius: "8px",
          backgroundColor: iconBg,
          color: iconColor,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {icon}
      </span>
      <span style={{ fontSize: "12px", fontWeight: 600, color: textColor }}>
        {label}
      </span>
    </button>
  );
}

function readCachedMovies(key: string): Movie[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as Movie[];
    return Array.isArray(parsed) ? enrichMovies(parsed) : [];
  } catch {
    return [];
  }
}

function writeCachedMovies(key: string, items: Movie[]) {
  try {
    localStorage.setItem(key, JSON.stringify(items));
  } catch {
    // Ignore localStorage failures.
  }
}

function dedupeMovies(items: Movie[]) {
  const unique = new Map<string, Movie>();

  for (const movie of items) {
    const enriched = enrichMovie(movie);
    if (!unique.has(enriched.movie_key)) {
      unique.set(enriched.movie_key, enriched);
    }
  }

  return [...unique.values()];
}

function getMovieDescription(movie: Movie): string {
  if (movie.description?.trim()) {
    return movie.description.trim();
  }
  if (movie.plot?.trim()) {
    return movie.plot.trim();
  }

  const title = movie.clean_title || movie.title;
  const genreText = movie.genres?.trim() || "movie";
  const yearText = movie.year ? ` from ${movie.year}` : "";
  return `${title} is a ${genreText.toLowerCase()} title${yearText}. Open the trailer to preview it and save it to your wishlist.`;
}

function MovieGridSkeleton({
  count = 10,
  theme = "dark",
}: {
  count?: number;
  theme?: "dark" | "light";
}) {
  const isLight = theme === "light";

  return (
    <div className="movie-grid" aria-hidden="true">
      {Array.from({ length: count }, (_, index) => (
        <article
          key={`movie-skeleton-${index}`}
          className={`movie-card movie-card--skeleton${isLight ? " movie-card--light" : ""}`}
        >
          <div className="movie-card__poster-button">
            <div
              className={`movie-poster-skeleton${
                isLight ? " movie-poster-skeleton--light" : ""
              }`}
            />
          </div>
          <div className="movie-info">
            <div
              className={`movie-skeleton-line movie-skeleton-line--title${
                isLight ? " movie-skeleton-line--light" : ""
              }`}
            />
            <div
              className={`movie-skeleton-line movie-skeleton-line--meta${
                isLight ? " movie-skeleton-line--light" : ""
              }`}
            />
            <div className="movie-card__actions">
              <div
                className={`movie-skeleton-line movie-skeleton-line--button${
                  isLight ? " movie-skeleton-line--light" : ""
                }`}
              />
              <div
                className={`movie-skeleton-line movie-skeleton-line--button${
                  isLight ? " movie-skeleton-line--light" : ""
                }`}
              />
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { user: appUser, isDark: d, toggleTheme, logout } = useAppStore();
  const user = appUser || {
    name: "Guest User",
    email: "guest@example.com",
    role: "user",
    preferredGenres: [],
    preferredMoods: [],
  };
  const preferenceSignature = [
    ...(user.preferredGenres ?? []),
    "--",
    ...(user.preferredMoods ?? []),
  ].join("|");
  const recommendationMood = user.preferredMoods?.[0]?.trim() || undefined;

  const [query, setQuery] = useState("");
  const [activeGenre, setActiveGenre] = useState("All");
  const [showDropdown, setShowDropdown] = useState(false);
  const [focused, setFocused] = useState(false);
  const [movies, setMovies] = useState<Movie[]>([]);
  const [wishlist, setWishlist] = useState<Movie[]>([]);
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null);
  const [showWishlistOnly, setShowWishlistOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [wishlistLoading, setWishlistLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<Movie[]>([]);
  const [recommendedMovies, setRecommendedMovies] = useState<Movie[]>([]);
  const [trailerMovieId, setTrailerMovieId] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteOtp, setDeleteOtp] = useState("");
  const [deleteStep, setDeleteStep] = useState<DeleteStep>("request");
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [homeRefreshKey, setHomeRefreshKey] = useState(0);

  const wishlistCacheKey = `${WISHLIST_CACHE_PREFIX}${user.email.toLowerCase()}`;
  const homeCacheKey = useMemo(
    () => getHomeCacheKey(activeGenre, user.email),
    [activeGenre, user.email],
  );
  const queryDisplayValue = query.trim();
  const queryValue = useDeferredValue(queryDisplayValue);

  useEffect(() => {
    const refreshHomeMovies = () => {
      setHomeRefreshKey((current) => current + 1);
    };
    const handleStorage = (event: StorageEvent) => {
      if (isHomeCacheRefreshKey(event.key)) {
        refreshHomeMovies();
      }
    };

    window.addEventListener(
      HOME_CACHE_INVALIDATED_EVENT,
      refreshHomeMovies as EventListener,
    );
    window.addEventListener("storage", handleStorage);

    return () => {
      window.removeEventListener(
        HOME_CACHE_INVALIDATED_EVENT,
        refreshHomeMovies as EventListener,
      );
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  useEffect(() => {
    let isCancelled = false;
    const cachedMovies = readCachedMovies(homeCacheKey);

    if (cachedMovies.length) {
      setMovies(cachedMovies);
      setLoading(false);
    } else {
      setMovies([]);
      setLoading(true);
    }

    const loadHome = async () => {
      try {
        const genreFilter = activeGenre === "All" ? undefined : activeGenre;
        const homeMovies = enrichMovies(
          await getHomeMovies(50, genreFilter, user.email.toLowerCase()),
        );
        if (isCancelled) {
          return;
        }
        setMovies(homeMovies);
        writeCachedMovies(homeCacheKey, homeMovies);
        setError("");
      } catch (loadError) {
        if (!isCancelled && !cachedMovies.length) {
          setError(
            loadError instanceof ApiError
              ? loadError.message
              : "Unable to load movies right now",
          );
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };

    void loadHome();

    return () => {
      isCancelled = true;
    };
  }, [activeGenre, homeCacheKey, homeRefreshKey, user.email]);

  useEffect(() => {
    let isCancelled = false;
    const cachedWishlist = readCachedMovies(wishlistCacheKey);

    if (cachedWishlist.length) {
      setWishlist(cachedWishlist);
      setWishlistLoading(false);
    }

    const loadWishlistItems = async () => {
      try {
        const items = enrichMovies(await getWishlist(user.email.toLowerCase()));
        if (isCancelled) {
          return;
        }
        setWishlist(items);
        writeCachedMovies(wishlistCacheKey, items);
      } catch {
        if (!isCancelled && !cachedWishlist.length) {
          setWishlist([]);
        }
      } finally {
        if (!isCancelled) {
          setWishlistLoading(false);
        }
      }
    };

    void loadWishlistItems();

    return () => {
      isCancelled = true;
    };
  }, [user.email, wishlistCacheKey]);

  const wishlistKeys = useMemo(
    () => new Set(wishlist.map((movie) => movie.movie_key)),
    [wishlist],
  );
  const homeMovies = useMemo(() => dedupeMovies(movies), [movies]);

  const wishlistMovies = useMemo(() => dedupeMovies(wishlist), [wishlist]);

  const filteredBrowseMovies = useMemo(() => {
    const source = showWishlistOnly ? wishlistMovies : homeMovies;
    const genreFiltered = source.filter((movie) => movieMatchesGenre(movie, activeGenre));

    if (!queryValue) {
      return genreFiltered;
    }

    return searchMoviesLocally(
      queryValue,
      genreFiltered,
      Math.max(genreFiltered.length, 50),
    );
  }, [activeGenre, homeMovies, queryValue, showWishlistOnly, wishlistMovies]);

  const visibleSearchResults = useMemo(() => {
    if (!queryValue || showWishlistOnly) {
      return [];
    }
    return searchResults.filter((movie) => movieMatchesGenre(movie, activeGenre));
  }, [activeGenre, queryValue, searchResults, showWishlistOnly]);

  const visibleRecommendedMovies = useMemo(() => {
    if (!queryValue || showWishlistOnly) {
      return [];
    }

    const taken = new Set(visibleSearchResults.map((movie) => movie.movie_key));
    return recommendedMovies.filter(
      (movie) =>
        !taken.has(movie.movie_key) && movieMatchesGenre(movie, activeGenre),
    );
  }, [
    activeGenre,
    queryValue,
    recommendedMovies,
    showWishlistOnly,
    visibleSearchResults,
  ]);

  const displayedMovies = queryValue && !showWishlistOnly
    ? visibleSearchResults
    : filteredBrowseMovies;
  const showMovieSkeleton = loading && !homeMovies.length;

  useEffect(() => {
    let isCancelled = false;

    if (!queryValue || showWishlistOnly) {
      setSearchResults([]);
      setRecommendedMovies([]);
      setSearchLoading(false);
      return () => {
        isCancelled = true;
      };
    }

    const runSearch = async () => {
      const localResults = searchMoviesLocally(queryValue, homeMovies, 50);
      const anchorMovie = localResults[0] ?? null;
      const fallbackRecommendations = anchorMovie
        ? getRecommendedMovies(anchorMovie, homeMovies, 50)
        : [];

      if (isCancelled) {
        return;
      }

      setSearchResults(localResults);
      setRecommendedMovies(fallbackRecommendations);

      const shouldSearchRemote = queryValue.length >= 2;
      const shouldLoadRemoteRecommendations = queryValue.length >= 3;

      if (!shouldSearchRemote && !shouldLoadRemoteRecommendations) {
        setSearchLoading(false);
        return;
      }

      setSearchLoading(true);

      let remoteResults: Movie[] = [];
      try {
        if (shouldSearchRemote) {
          remoteResults = enrichMovies(await searchCatalogMovies(queryValue, 50));
        }
      } catch {
        remoteResults = [];
      }

      const combinedResults = remoteResults.length
        ? dedupeMovies([...remoteResults, ...localResults])
        : dedupeMovies(localResults);
      const remoteAnchorMovie = remoteResults[0] ?? combinedResults[0] ?? anchorMovie;

      let remoteRecommendations: Movie[] = [];
      if (shouldLoadRemoteRecommendations && remoteAnchorMovie) {
        try {
          remoteRecommendations = enrichMovies(
            await recommendCatalogMovies(
              remoteAnchorMovie.clean_title || remoteAnchorMovie.title,
              1,
              50,
              {
                mood: recommendationMood,
                userEmail: user.email.toLowerCase(),
              },
            ),
          );
        } catch {
          remoteRecommendations = [];
        }
      }

      if (isCancelled) {
        return;
      }

      setSearchResults(combinedResults);
      setRecommendedMovies(
        dedupeMovies([...remoteRecommendations, ...fallbackRecommendations]),
      );
      setSearchLoading(false);
    };

    const timeoutId = window.setTimeout(() => {
      void runSearch();
    }, queryValue.length >= 2 ? 260 : 120);

    return () => {
      isCancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [
    homeMovies,
    preferenceSignature,
    queryValue,
    recommendationMood,
    showWishlistOnly,
    user.email,
  ]);

  const toggleWishlist = async (movie: Movie) => {
    const email = user.email.toLowerCase();
    const exists = wishlistKeys.has(movie.movie_key);
    const nextWishlist = exists
      ? wishlist.filter((item) => item.movie_key !== movie.movie_key)
      : [movie, ...wishlist.filter((item) => item.movie_key !== movie.movie_key)];

    setWishlist(nextWishlist);
    writeCachedMovies(wishlistCacheKey, nextWishlist);

    try {
      if (exists) {
        await removeWishlistMovie(email, movie.movie_key);
      } else {
        await addWishlistMovie(email, movie);
      }
    } catch (wishlistError) {
      setWishlist(wishlist);
      writeCachedMovies(wishlistCacheKey, wishlist);
      alert(
        wishlistError instanceof ApiError
          ? wishlistError.message
          : "Unable to update wishlist right now",
      );
    }
  };

  const handleDeleteRequest = async () => {
    setDeleteLoading(true);
    try {
      const data = await requestDeleteAccountOtp(user.email.toLowerCase());
      alert(data.msg || "Account deletion OTP sent to your email");
      setDeleteStep("confirm");
    } catch (deleteError) {
      alert(
        deleteError instanceof ApiError
          ? deleteError.message
          : "Unable to request account deletion right now",
      );
    } finally {
      setDeleteLoading(false);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!deleteOtp.trim()) {
      alert("Please enter the OTP from your email");
      return;
    }

    setDeleteLoading(true);
    try {
      const data = await confirmDeleteAccount(user.email.toLowerCase(), deleteOtp.trim());
      writeCachedMovies(wishlistCacheKey, []);
      alert(data.msg || "Account deleted successfully");
      setShowDeleteModal(false);
      logout();
      navigate("/login");
    } catch (deleteError) {
      alert(
        deleteError instanceof ApiError
          ? deleteError.message
          : "Unable to delete account right now",
      );
    } finally {
      setDeleteLoading(false);
    }
  };

  const resetDeleteModal = () => {
    setShowDeleteModal(false);
    setDeleteOtp("");
    setDeleteStep("request");
    setDeleteLoading(false);
  };

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        background: d
          ? "radial-gradient(circle at top, rgba(239,68,68,0.12), transparent 28%), #0a0a0f"
          : "linear-gradient(180deg, #fff7f7 0%, #f4f5f7 30%, #eef1f4 100%)",
        minHeight: "100vh",
      }}
    >
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 40,
          display: "flex",
          alignItems: "center",
          gap: "16px",
          padding: "0 24px",
          height: "60px",
          backgroundColor: d ? "rgba(13,13,20,0.97)" : "rgba(255,255,255,0.97)",
          borderBottom: d ? "1px solid rgba(255,255,255,0.08)" : "1px solid #e4e4e7",
          backdropFilter: "blur(16px)",
          boxShadow: d ? "none" : "0 1px 20px rgba(0,0,0,0.07)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
          <MovieBuzzLogo
            size={30}
            theme={d ? "dark" : "light"}
            subtitle=""
            imageClassName={
              d
                ? "rounded-xl border-white/10 bg-black/40 shadow-none"
                : "rounded-xl border-zinc-200 bg-white shadow-none"
            }
            wordmarkClassName={d ? "text-sm tracking-[0.22em]" : "text-sm tracking-[0.22em]"}
          />
        </div>

        <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              width: "100%",
              maxWidth: "540px",
              borderRadius: "14px",
              border: focused
                ? d
                  ? "1px solid rgba(255,255,255,0.3)"
                  : "1px solid rgba(239,68,68,0.5)"
                : d
                  ? "1px solid rgba(255,255,255,0.1)"
                  : "1px solid #d4d4d8",
              padding: "7px 12px",
              backgroundColor: d ? "rgba(255,255,255,0.05)" : "#ffffff",
              boxShadow: focused
                ? d
                  ? "0 0 0 3px rgba(255,255,255,0.05)"
                  : "0 0 0 3px rgba(239,68,68,0.08)"
                : d
                  ? "none"
                  : "0 1px 4px rgba(0,0,0,0.06)",
              transition: "all 0.2s ease",
            }}
          >
            <input
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder="Search movies"
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                fontSize: "13px",
                color: d ? "#ffffff" : "#18181b",
              }}
            />
            <Search size={14} style={{ color: d ? "#52525b" : "#a1a1aa", flexShrink: 0 }} />
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
          <button
            aria-label="Wishlist"
            onClick={() => setShowWishlistOnly((current) => !current)}
            style={{
              position: "relative",
              padding: "8px",
              borderRadius: "12px",
              border: d ? "1px solid rgba(255,255,255,0.1)" : "1px solid #e4e4e7",
              backgroundColor: showWishlistOnly
                ? "#ef4444"
                : d
                  ? "rgba(255,255,255,0.05)"
                  : "#ffffff",
              color: showWishlistOnly ? "#ffffff" : d ? "#a1a1aa" : "#71717a",
              cursor: "pointer",
              transition: "all 0.2s ease",
              boxShadow: showWishlistOnly
                ? "0 10px 24px rgba(239,68,68,0.35)"
                : d
                  ? "none"
                  : "0 1px 4px rgba(0,0,0,0.06)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Bookmark size={17} />
            <span
              style={{
                position: "absolute",
                top: "-4px",
                right: "-4px",
                width: "16px",
                height: "16px",
                backgroundColor: "#ef4444",
                borderRadius: "50%",
                color: "white",
                fontSize: "9px",
                fontWeight: 900,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 2px 6px rgba(239,68,68,0.5)",
              }}
            >
              {wishlistLoading ? "…" : wishlist.length}
            </span>
          </button>

          <div style={{ position: "relative" }}>
            <button
              onClick={(event) => {
                event.stopPropagation();
                setShowDropdown((current) => !current);
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                paddingLeft: "4px",
                paddingRight: "12px",
                paddingTop: "4px",
                paddingBottom: "4px",
                borderRadius: "14px",
                border: d ? "1px solid rgba(255,255,255,0.1)" : "1px solid #e4e4e7",
                backgroundColor: d ? "rgba(255,255,255,0.05)" : "#ffffff",
                cursor: "pointer",
                transition: "all 0.2s ease",
                boxShadow: d ? "none" : "0 1px 4px rgba(0,0,0,0.06)",
              }}
            >
              <div
                style={{
                  width: "30px",
                  height: "30px",
                  borderRadius: "10px",
                  background: "linear-gradient(135deg, #ef4444, #f97316, #facc15)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  boxShadow: "0 3px 10px rgba(239,68,68,0.35)",
                }}
              >
                <span style={{ color: "white", fontSize: "12px", fontWeight: 900 }}>
                  {user.name.charAt(0).toUpperCase()}
                </span>
              </div>
              <span style={{ fontSize: "12px", fontWeight: 700, color: d ? "#d4d4d8" : "#3f3f46" }}>
                {user.name}
              </span>
              <ChevronDown
                size={12}
                style={{
                  color: d ? "#52525b" : "#a1a1aa",
                  transform: showDropdown ? "rotate(180deg)" : "rotate(0deg)",
                  transition: "transform 0.2s ease",
                }}
              />
            </button>

            {showDropdown && (
              <>
                <div style={{ position: "fixed", inset: 0, zIndex: 40 }} onClick={() => setShowDropdown(false)} />
                <div
                  style={{
                    position: "absolute",
                    right: 0,
                    marginTop: "8px",
                    width: "220px",
                    borderRadius: "18px",
                    border: d ? "1px solid rgba(255,255,255,0.1)" : "1px solid #e4e4e7",
                    backgroundColor: d ? "#131318" : "#ffffff",
                    boxShadow: d ? "0 20px 60px rgba(0,0,0,0.7)" : "0 20px 60px rgba(0,0,0,0.12)",
                    zIndex: 50,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      padding: "14px 16px",
                      borderBottom: d ? "1px solid rgba(255,255,255,0.06)" : "1px solid #f4f4f5",
                      display: "flex",
                      alignItems: "center",
                      gap: "12px",
                    }}
                  >
                    <div
                      style={{
                        width: "36px",
                        height: "36px",
                        borderRadius: "10px",
                        background: "linear-gradient(135deg, #ef4444, #f97316, #facc15)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        flexShrink: 0,
                        boxShadow: "0 3px 10px rgba(239,68,68,0.35)",
                      }}
                    >
                      <span style={{ color: "white", fontSize: "13px", fontWeight: 900 }}>
                        {user.name.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <p style={{ fontWeight: 700, fontSize: "13px", color: d ? "#ffffff" : "#18181b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {user.name}
                      </p>
                      <p style={{ fontSize: "11px", color: "#71717a", marginTop: "2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {user.email}
                      </p>
                    </div>
                  </div>

                  <div style={{ padding: "8px" }}>
                    <DropdownItem
                      icon={<UserRound size={14} />}
                      iconBg={d ? "rgba(96,165,250,0.16)" : "#eff6ff"}
                      iconColor={d ? "#93c5fd" : "#2563eb"}
                      label="Profile"
                      textColor={d ? "#d4d4d8" : "#3f3f46"}
                      hoverBg={d ? "rgba(255,255,255,0.06)" : "#f4f4f5"}
                      onClick={() => {
                        setShowDropdown(false);
                        navigate("/preferences-setup");
                      }}
                    />
                    <DropdownItem
                      icon={d ? <Sun size={14} /> : <Moon size={14} />}
                      iconBg={d ? "rgba(251,191,36,0.15)" : "#eef2ff"}
                      iconColor={d ? "#fbbf24" : "#6366f1"}
                      label={d ? "Switch to Light" : "Switch to Dark"}
                      textColor={d ? "#d4d4d8" : "#3f3f46"}
                      hoverBg={d ? "rgba(255,255,255,0.06)" : "#f4f4f5"}
                      onClick={() => {
                        toggleTheme();
                        setShowDropdown(false);
                      }}
                    />
                    <DropdownItem
                      icon={<Trash2 size={14} />}
                      iconBg="rgba(239,68,68,0.12)"
                      iconColor="#ef4444"
                      label="Delete Account"
                      textColor="#ef4444"
                      hoverBg="rgba(239,68,68,0.08)"
                      onClick={() => {
                        setShowDeleteModal(true);
                        setDeleteStep("request");
                        setDeleteOtp("");
                        setShowDropdown(false);
                      }}
                    />
                    <div style={{ height: "1px", backgroundColor: d ? "rgba(255,255,255,0.06)" : "#f4f4f5", margin: "4px 0" }} />
                    <DropdownItem
                      icon={<LogOut size={14} />}
                      iconBg={d ? "rgba(255,255,255,0.06)" : "#f4f4f5"}
                      iconColor={d ? "#a1a1aa" : "#71717a"}
                      label="Sign Out"
                      textColor={d ? "#d4d4d8" : "#3f3f46"}
                      hoverBg={d ? "rgba(255,255,255,0.06)" : "#f4f4f5"}
                      onClick={() => {
                        setShowDropdown(false);
                        handleLogout();
                      }}
                    />
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      <div style={{ display: "flex", justifyContent: "center", padding: "16px 24px 8px" }}>
        <div
          style={{
            display: "inline-flex",
            flexWrap: "wrap",
            justifyContent: "center",
            alignItems: "center",
            gap: "4px",
            padding: "5px",
            borderRadius: "18px",
            border: d ? "1px solid rgba(255,255,255,0.14)" : "1px solid #c4c4c8",
            backgroundColor: d ? "rgba(255,255,255,0.03)" : "#ffffff",
            boxShadow: d ? "0 0 0 1px rgba(255,255,255,0.03)" : "0 2px 16px rgba(0,0,0,0.09)",
          }}
        >
          {GENRES.map((genre) => {
            const isActive = activeGenre === genre;
            return (
              <button
                key={genre}
                onClick={() => setActiveGenre(genre)}
                style={{
                  padding: "6px 14px",
                  borderRadius: "12px",
                  border: "none",
                  fontSize: "11px",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                  cursor: "pointer",
                  transition: "all 0.18s ease",
                  backgroundColor: isActive ? "#dc2626" : "transparent",
                  color: isActive ? "#ffffff" : d ? "#71717a" : "#52525b",
                  boxShadow: isActive ? "0 3px 12px rgba(220,38,38,0.4)" : "none",
                }}
              >
                {genre}
              </button>
            );
          })}
        </div>
      </div>

      <main style={{ flex: 1, padding: "0 24px 32px" }}>
        <section style={{ maxWidth: "1320px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "20px" }}>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "12px", padding: "20px 0 4px" }}>
            <div>
              <h1 style={{ fontSize: "32px", fontWeight: 900, letterSpacing: "-0.04em", color: d ? "#ffffff" : "#18181b" }}>
                {showWishlistOnly
                  ? "Your Wishlist"
                  : queryDisplayValue
                    ? `Results for "${queryDisplayValue}"`
                    : "Explore MovieBuzz"}
              </h1>
            </div>
          </div>

          {error && !homeMovies.length && (
            <div
              style={{
                padding: "16px 18px",
                borderRadius: "16px",
                backgroundColor: d ? "rgba(127,29,29,0.28)" : "#fef2f2",
                border: "1px solid rgba(239,68,68,0.22)",
                color: d ? "#fecaca" : "#991b1b",
              }}
            >
              {error}
            </div>
          )}

          {showMovieSkeleton ? (
            <MovieGridSkeleton theme={d ? "dark" : "light"} />
          ) : null}

          {searchLoading && !!queryDisplayValue && !showWishlistOnly ? (
            <div
              style={{
                padding: "14px 16px",
                borderRadius: "18px",
                backgroundColor: d ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.78)",
                border: d ? "1px solid rgba(255,255,255,0.08)" : "1px solid #e4e4e7",
                color: d ? "#d4d4d8" : "#3f3f46",
              }}
            >
              Finding more matches from the full catalog...
            </div>
          ) : null}

          {!loading && !searchLoading && !displayedMovies.length && (
            <div
              style={{
                padding: "28px",
                borderRadius: "18px",
                backgroundColor: d ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.82)",
                border: d ? "1px solid rgba(255,255,255,0.08)" : "1px solid #e4e4e7",
                color: d ? "#d4d4d8" : "#3f3f46",
              }}
            >
              {showWishlistOnly
                ? queryValue
                  ? "No wishlist movies matched your search and genre filter."
                  : "Your wishlist is empty. Add a few movies and they will appear here."
                : queryValue
                  ? "No movies matched your current search and genre filter."
                  : "No movies matched your current genre filter."}
            </div>
          )}

          {!!displayedMovies.length && (
            <div className="movie-grid">
              {displayedMovies.map((movie) => (
                <MovieCard
                  key={movie.movie_key}
                  movie={movie}
                  theme={d ? "dark" : "light"}
                  isWishlisted={wishlistKeys.has(movie.movie_key)}
                  onToggleWishlist={toggleWishlist}
                  onOpenDetails={setSelectedMovie}
                />
              ))}
            </div>
          )}

          {!!queryValue && !showWishlistOnly && !!visibleRecommendedMovies.length && (
            <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
              <div>
                <h2
                  style={{
                    fontSize: "24px",
                    fontWeight: 800,
                    letterSpacing: "-0.03em",
                    color: d ? "#ffffff" : "#18181b",
                  }}
                >
                  Recommended
                </h2>
                <p
                  style={{
                    marginTop: "6px",
                    color: d ? "#a1a1aa" : "#52525b",
                  }}
                >
                  Similar picks based on your current search.
                </p>
              </div>

              <div className="movie-grid">
                {visibleRecommendedMovies.map((movie) => (
                  <MovieCard
                    key={`recommended-${movie.movie_key}`}
                    movie={movie}
                    theme={d ? "dark" : "light"}
                    isWishlisted={wishlistKeys.has(movie.movie_key)}
                    onToggleWishlist={toggleWishlist}
                    onOpenDetails={setSelectedMovie}
                  />
                ))}
              </div>
            </div>
          )}
        </section>
      </main>

      <Footer isDark={d} />

      {selectedMovie && (
        <div
          onClick={() => setSelectedMovie(null)}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0, 0, 0, 0.72)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "24px",
            zIndex: 80,
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "min(960px, 100%)",
              maxHeight: "90vh",
              overflowY: "auto",
              borderRadius: "24px",
              backgroundColor: d ? "#12131a" : "#ffffff",
              color: d ? "#f8fafc" : "#111827",
              boxShadow: "0 28px 80px rgba(0,0,0,0.45)",
              border: d ? "1px solid rgba(255,255,255,0.08)" : "1px solid #e5e7eb",
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "24px",
              padding: "24px",
            }}
          >
            <div>
              <div
                style={{
                  borderRadius: "18px",
                  overflow: "hidden",
                  backgroundColor: d ? "#1f2937" : "#f3f4f6",
                }}
              >
                {selectedMovie.poster ? (
                  <img
                    src={selectedMovie.poster}
                    alt={selectedMovie.title}
                    loading="lazy"
                    referrerPolicy="no-referrer"
                    onError={(event) => {
                      event.currentTarget.src = createMoviePosterDataUrl(
                        selectedMovie.clean_title || selectedMovie.title,
                        selectedMovie.year || "",
                        selectedMovie.genres || "",
                      );
                    }}
                    style={{ width: "100%", aspectRatio: "2 / 3", objectFit: "cover" }}
                  />
                ) : (
                  <div
                    style={{
                      width: "100%",
                      aspectRatio: "2 / 3",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    No Image
                  </div>
                )}
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                  gap: "12px",
                }}
              >
                <div>
                  <div style={{ marginBottom: "12px" }}>
                    <MovieBuzzLogo
                      size={32}
                      theme={d ? "dark" : "light"}
                      subtitle="Movie details"
                      imageClassName={
                        d
                          ? "rounded-xl border-white/10 bg-black/40 shadow-none"
                          : "rounded-xl border-zinc-200 bg-white shadow-none"
                      }
                      wordmarkClassName="text-sm tracking-[0.2em]"
                    />
                  </div>
                  <h2 style={{ fontSize: "30px", fontWeight: 900, letterSpacing: "-0.04em" }}>
                    {selectedMovie.clean_title || selectedMovie.title}
                  </h2>
                  <p
                    style={{
                      marginTop: "8px",
                      color: d ? "#cbd5e1" : "#4b5563",
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "10px",
                    }}
                  >
                    <span>Release Year: {selectedMovie.year || "N/A"}</span>
                    <span>
                      IMDb / Rating: {selectedMovie.imdb_rating || selectedMovie.rating || "N/A"}
                    </span>
                    <span>Runtime: {selectedMovie.runtime || "N/A"}</span>
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedMovie(null)}
                  aria-label="Close movie details"
                  style={{
                    width: "40px",
                    height: "40px",
                    borderRadius: "999px",
                    border: "none",
                    backgroundColor: d ? "rgba(255,255,255,0.08)" : "#f3f4f6",
                    color: d ? "#f8fafc" : "#111827",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "pointer",
                    flexShrink: 0,
                  }}
                >
                  <X size={18} />
                </button>
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
                <span
                  style={{
                    padding: "8px 12px",
                    borderRadius: "999px",
                    backgroundColor: d ? "rgba(239,68,68,0.15)" : "#fee2e2",
                    color: d ? "#fecaca" : "#b91c1c",
                    fontSize: "13px",
                    fontWeight: 700,
                  }}
                >
                  {selectedMovie.genres || "Genre unavailable"}
                </span>
                <span
                  style={{
                    padding: "8px 12px",
                    borderRadius: "999px",
                    backgroundColor: d ? "rgba(59,130,246,0.14)" : "#dbeafe",
                    color: d ? "#bfdbfe" : "#1d4ed8",
                    fontSize: "13px",
                    fontWeight: 700,
                  }}
                >
                  Rating: {selectedMovie.rating ?? "N/A"}
                </span>
              </div>

              <div>
                <h3 style={{ fontSize: "16px", fontWeight: 800, marginBottom: "8px" }}>
                  Description
                </h3>
                <p style={{ lineHeight: 1.7, color: d ? "#d4d4d8" : "#374151" }}>
                  {getMovieDescription(selectedMovie)}
                </p>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                  gap: "12px",
                }}
              >
                <div
                  style={{
                    padding: "14px",
                    borderRadius: "16px",
                    backgroundColor: d ? "rgba(255,255,255,0.04)" : "#f8fafc",
                  }}
                >
                  <strong style={{ display: "block", marginBottom: "6px" }}>Director</strong>
                  <span style={{ color: d ? "#d4d4d8" : "#4b5563" }}>
                    {selectedMovie.director || "N/A"}
                  </span>
                </div>
                <div
                  style={{
                    padding: "14px",
                    borderRadius: "16px",
                    backgroundColor: d ? "rgba(255,255,255,0.04)" : "#f8fafc",
                  }}
                >
                  <strong style={{ display: "block", marginBottom: "6px" }}>Cast</strong>
                  <span style={{ color: d ? "#d4d4d8" : "#4b5563" }}>
                    {selectedMovie.cast || "N/A"}
                  </span>
                </div>
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
                <button
                  type="button"
                  onClick={() => toggleWishlist(selectedMovie)}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "12px 16px",
                    borderRadius: "14px",
                    border: "none",
                    backgroundColor: wishlistKeys.has(selectedMovie.movie_key)
                      ? "#ef4444"
                      : d
                        ? "rgba(255,255,255,0.08)"
                        : "#111827",
                    color: "#ffffff",
                    cursor: "pointer",
                    fontWeight: 700,
                  }}
                >
                  <Bookmark size={16} />
                  {wishlistKeys.has(selectedMovie.movie_key)
                    ? "Remove from Wishlist"
                    : "Add to Wishlist"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (selectedMovie.movie_id) {
                      setTrailerMovieId(selectedMovie.movie_id);
                      return;
                    }
                    if (selectedMovie.youtube_link) {
                      window.open(selectedMovie.youtube_link, "_blank", "noopener,noreferrer");
                    }
                  }}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "12px 16px",
                    borderRadius: "14px",
                    border: d ? "1px solid rgba(255,255,255,0.1)" : "1px solid #d1d5db",
                    backgroundColor: "transparent",
                    color: d ? "#f8fafc" : "#111827",
                    fontWeight: 700,
                    cursor: "pointer",
                  }}
                >
                  <ExternalLink size={16} />
                  Watch Trailer
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <TrailerPlayer
        movieId={trailerMovieId}
        onClose={() => setTrailerMovieId(null)}
      />

      {showDeleteModal && (
        <div
          onClick={resetDeleteModal}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "rgba(0, 0, 0, 0.64)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "24px",
            zIndex: 85,
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "min(460px, 100%)",
              borderRadius: "22px",
              padding: "24px",
              backgroundColor: d ? "#12131a" : "#ffffff",
              border: d ? "1px solid rgba(255,255,255,0.08)" : "1px solid #e5e7eb",
              boxShadow: "0 24px 80px rgba(0,0,0,0.35)",
              color: d ? "#f8fafc" : "#111827",
              }}
            >
              <div
                style={{
                  display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: "12px",
                marginBottom: "14px",
                }}
              >
                <div>
                  <div style={{ marginBottom: "12px" }}>
                    <MovieBuzzLogo
                      size={34}
                      theme={d ? "dark" : "light"}
                      subtitle="Security confirmation"
                      imageClassName={
                        d
                          ? "rounded-xl border-white/10 bg-black/40 shadow-none"
                          : "rounded-xl border-zinc-200 bg-white shadow-none"
                      }
                      wordmarkClassName="text-sm tracking-[0.2em]"
                    />
                  </div>
                  <h2 style={{ fontSize: "22px", fontWeight: 900 }}>Delete Account</h2>
                  <p style={{ marginTop: "6px", color: d ? "#cbd5e1" : "#4b5563" }}>
                    {deleteStep === "request"
                    ? "Send a deletion OTP to your email before removing your account."
                    : "Enter the OTP from your email to permanently delete your account."}
                </p>
              </div>
              <button
                type="button"
                onClick={resetDeleteModal}
                style={{
                  width: "38px",
                  height: "38px",
                  borderRadius: "999px",
                  border: "none",
                  backgroundColor: d ? "rgba(255,255,255,0.08)" : "#f3f4f6",
                  color: d ? "#f8fafc" : "#111827",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                }}
              >
                <X size={18} />
              </button>
            </div>

            <div
              style={{
                padding: "14px 16px",
                borderRadius: "16px",
                backgroundColor: d ? "rgba(239,68,68,0.12)" : "#fef2f2",
                color: d ? "#fecaca" : "#991b1b",
                marginBottom: "16px",
              }}
            >
              This action permanently deletes your account, wishlist, preferences, and history.
            </div>

            {deleteStep === "request" ? (
              <button
                type="button"
                onClick={handleDeleteRequest}
                disabled={deleteLoading}
                style={{
                  width: "100%",
                  padding: "12px 16px",
                  borderRadius: "14px",
                  border: "none",
                  backgroundColor: "#ef4444",
                  color: "#ffffff",
                  fontWeight: 800,
                  cursor: deleteLoading ? "not-allowed" : "pointer",
                  opacity: deleteLoading ? 0.7 : 1,
                }}
              >
                {deleteLoading ? "Sending OTP..." : "Send Delete OTP"}
              </button>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                <input
                  type="text"
                  value={deleteOtp}
                  onChange={(event) => setDeleteOtp(event.target.value)}
                  placeholder="Enter deletion OTP"
                  style={{
                    width: "100%",
                    padding: "12px 14px",
                    borderRadius: "14px",
                    border: d ? "1px solid rgba(255,255,255,0.12)" : "1px solid #d1d5db",
                    backgroundColor: d ? "rgba(255,255,255,0.04)" : "#ffffff",
                    color: d ? "#f8fafc" : "#111827",
                    outline: "none",
                  }}
                />
                <button
                  type="button"
                  onClick={handleDeleteConfirm}
                  disabled={deleteLoading}
                  style={{
                    width: "100%",
                    padding: "12px 16px",
                    borderRadius: "14px",
                    border: "none",
                    backgroundColor: "#ef4444",
                    color: "#ffffff",
                    fontWeight: 800,
                    cursor: deleteLoading ? "not-allowed" : "pointer",
                    opacity: deleteLoading ? 0.7 : 1,
                  }}
                >
                  {deleteLoading ? "Deleting..." : "Delete My Account"}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
