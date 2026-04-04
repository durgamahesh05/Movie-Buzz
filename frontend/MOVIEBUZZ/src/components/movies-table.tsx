import { useEffect, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Loader2,
  PlusCircle,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  addAdminMoviesManual,
  ApiError,
  deleteAdminMovie,
  getAdminMovies,
  uploadAdminMoviesCsv,
  type Movie,
} from "../lib/api";
import { invalidateHomeMovieCache } from "../lib/movie-cache";

interface MoviesTableProps {
  limit?: number;
  refreshToken?: number;
  onCatalogChange?: () => void;
}

function getDisplayTitle(movie: Movie) {
  return movie.clean_title || movie.title;
}

export function MoviesTable({
  limit,
  refreshToken = 0,
  onCatalogChange,
}: MoviesTableProps) {
  const isSummaryView = typeof limit === "number";
  const pageSize = limit ?? 1000;
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showAddModal, setShowAddModal] = useState(false);
  const [addMethod, setAddMethod] = useState<"csv" | "manual">("manual");
  const [submitting, setSubmitting] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [availableGenres, setAvailableGenres] = useState<string[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [genreFilter, setGenreFilter] = useState("");
  const [form, setForm] = useState({
    title: "",
    genre: "",
    rating: "",
    year: "",
    poster: "",
  });

  useEffect(() => {
    if (isSummaryView) {
      return;
    }

    const timerId = window.setTimeout(() => {
      setOffset(0);
      setSearchQuery(searchInput.trim());
    }, 250);

    return () => {
      window.clearTimeout(timerId);
    };
  }, [isSummaryView, searchInput]);

  useEffect(() => {
    let cancelled = false;

    const loadMovies = async () => {
      setLoading(true);
      try {
        const data = await getAdminMovies({
          limit: pageSize,
          offset,
          search: isSummaryView ? undefined : searchQuery,
          genre: isSummaryView || !genreFilter ? undefined : genreFilter,
        });
        if (cancelled) {
          return;
        }
        setMovies(data.items);
        setTotal(data.total);
        setHasMore(data.has_more);
        setAvailableGenres(data.genres);
        setError("");
      } catch (loadError) {
        if (cancelled) {
          return;
        }
        setError(
          loadError instanceof ApiError
            ? loadError.message
            : "Unable to load movies right now",
        );
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void loadMovies();

    return () => {
      cancelled = true;
    };
  }, [genreFilter, isSummaryView, offset, pageSize, refreshKey, refreshToken, searchQuery]);

  const refreshMovies = () => {
    setRefreshKey((current) => current + 1);
  };

  const resetModal = () => {
    setShowAddModal(false);
    setAddMethod("manual");
    setCsvFile(null);
    setSubmitting(false);
    setForm({
      title: "",
      genre: "",
      rating: "",
      year: "",
      poster: "",
    });
  };

  const handleAdd = async () => {
    if (addMethod === "manual" && !form.title.trim()) {
      return;
    }
    if (addMethod === "csv" && !csvFile) {
      alert("Choose a CSV file first");
      return;
    }

    setSubmitting(true);
    try {
      if (addMethod === "manual") {
        const response = await addAdminMoviesManual([
          {
            title: form.title.trim(),
            genres: form.genre.trim(),
            rating: form.rating ? Number(form.rating) : undefined,
            year: form.year.trim(),
            poster: form.poster.trim(),
          },
        ]);
        if (response.inserted > 0) {
          onCatalogChange?.();
        }
      } else if (csvFile) {
        const response = await uploadAdminMoviesCsv(csvFile);
        if (response.inserted > 0) {
          onCatalogChange?.();
        }
      }

      invalidateHomeMovieCache();
      if (!isSummaryView) {
        setOffset(0);
      }
      refreshMovies();
      resetModal();
    } catch (submitError) {
      alert(
        submitError instanceof ApiError
          ? submitError.message
          : "Unable to save movies right now",
      );
      setSubmitting(false);
    }
  };

  const handleDelete = async (movie: Movie) => {
    if (!movie.movie_id) {
      return;
    }
    if (!window.confirm(`Delete ${getDisplayTitle(movie)} from the admin catalog?`)) {
      return;
    }

    setDeletingId(movie.movie_id);
    try {
      await deleteAdminMovie(movie.movie_id);
      invalidateHomeMovieCache();
      onCatalogChange?.();
      if (!isSummaryView && movies.length === 1 && offset > 0) {
        setOffset(Math.max(0, offset - pageSize));
      }
      refreshMovies();
    } catch (deleteError) {
      alert(
        deleteError instanceof ApiError
          ? deleteError.message
          : "Unable to delete movie right now",
      );
    } finally {
      setDeletingId(null);
    }
  };

  const visibleStart = total > 0 ? offset + 1 : 0;
  const visibleEnd = offset + movies.length;

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-200/60 bg-white text-zinc-900 shadow-sm dark:border-zinc-800/60 dark:bg-zinc-900 dark:text-zinc-100">
      {!isSummaryView && (
        <div className="border-b border-zinc-200/60 bg-zinc-50/80 p-4 dark:border-zinc-800/60 dark:bg-zinc-900/60">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h3 className="text-sm text-zinc-500 dark:text-zinc-400">
                Movie Records From SQLite
              </h3>
              <p className="mt-1 text-2xl font-semibold text-zinc-950 dark:text-white">
                {loading ? "..." : total.toLocaleString()}
              </p>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                Loads {pageSize.toLocaleString()} rows at a time so the admin page stays
                responsive.
              </p>
            </div>
            <Button onClick={() => setShowAddModal(true)}>
              <PlusCircle className="mr-2 h-4 w-4" />
              Add Movie
            </Button>
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_220px_auto]">
            <input
              type="search"
              placeholder="Search title, year, or genre"
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              className="flex h-10 w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:placeholder:text-zinc-500 dark:focus:ring-zinc-100"
            />
            <select
              value={genreFilter}
              onChange={(event) => {
                setGenreFilter(event.target.value);
                setOffset(0);
              }}
              className="flex h-10 w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:focus:ring-zinc-100"
            >
              <option value="">All genres</option>
              {availableGenres.map((genreOption) => (
                <option key={genreOption} value={genreOption}>
                  {genreOption}
                </option>
              ))}
            </select>
            <div className="flex items-center text-sm text-zinc-500 dark:text-zinc-400">
              {loading
                ? "Loading rows..."
                : `Showing ${visibleStart.toLocaleString()}-${visibleEnd.toLocaleString()} of ${total.toLocaleString()}`}
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="border-b border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-950/60 dark:bg-rose-950/30 dark:text-rose-200">
          {error}
        </div>
      )}

      <Table>
        <TableHeader>
          <TableRow className="border-zinc-200/60 bg-zinc-50/80 hover:bg-zinc-50 dark:border-zinc-800/60 dark:bg-zinc-900/60 dark:hover:bg-zinc-900/60">
            <TableHead className="text-zinc-600 dark:text-zinc-400">Title</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Genre</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Rating</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Year</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Source</TableHead>
            <TableHead className="text-right text-zinc-600 dark:text-zinc-400">
              Actions
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell
                colSpan={6}
                className="py-8 text-center text-zinc-500 dark:text-zinc-400"
              >
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading movie rows...
                </span>
              </TableCell>
            </TableRow>
          ) : movies.length ? (
            movies.map((movie) => {
              const numericRating = movie.imdb_rating || movie.rating || "N/A";
              const canDelete = Boolean(movie.can_delete && movie.movie_id);
              const isDeleting = deletingId === movie.movie_id;

              return (
                <TableRow
                  key={`${movie.source_label}-${movie.movie_key}`}
                  className="border-zinc-200/60 hover:bg-zinc-50 dark:border-zinc-800/60 dark:hover:bg-zinc-800/50"
                >
                  <TableCell className="font-medium text-zinc-900 dark:text-zinc-300">
                    {getDisplayTitle(movie)}
                  </TableCell>
                  <TableCell className="text-zinc-700 dark:text-zinc-300">
                    {movie.genres || "Unspecified"}
                  </TableCell>
                  <TableCell className="text-zinc-700 dark:text-zinc-300">
                    <span className="text-amber-600 dark:text-amber-400">
                      IMDb {numericRating}
                    </span>
                  </TableCell>
                  <TableCell className="text-zinc-700 dark:text-zinc-300">
                    {movie.year || "N/A"}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        movie.source === "wishlist"
                          ? "secondary"
                          : movie.source === "admin"
                            ? "default"
                            : "outline"
                      }
                    >
                      {movie.source_label || "Catalog"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      disabled={!canDelete || isDeleting}
                      className="text-red-500 hover:bg-red-50 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-40 dark:hover:bg-red-950/30"
                      onClick={() => handleDelete(movie)}
                      title={
                        canDelete
                          ? "Delete admin-added movie"
                          : "Only admin-added catalog rows can be deleted"
                      }
                    >
                      {isDeleting ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })
          ) : (
            <TableRow>
              <TableCell
                colSpan={6}
                className="py-8 text-center text-zinc-500 dark:text-zinc-400"
              >
                No movie rows matched the current filters.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      {!isSummaryView && total > 0 ? (
        <div className="flex flex-col gap-3 border-t border-zinc-200/60 bg-zinc-50/60 px-4 py-3 text-sm dark:border-zinc-800/60 dark:bg-zinc-900/40 md:flex-row md:items-center md:justify-between">
          <p className="text-zinc-500 dark:text-zinc-400">
            Browse through the catalog in chunks of {pageSize.toLocaleString()} rows.
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={loading || offset === 0}
              onClick={() => setOffset((current) => Math.max(0, current - pageSize))}
            >
              <ChevronLeft className="mr-2 h-4 w-4" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={loading || !hasMore}
              onClick={() => setOffset((current) => current + pageSize)}
            >
              Next
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      ) : null}

      {showAddModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4 backdrop-blur-sm"
          onClick={resetModal}
        >
          <div
            className="relative w-full max-w-md rounded-xl border border-zinc-200/60 bg-white p-6 text-zinc-900 shadow-2xl dark:border-zinc-800/60 dark:bg-zinc-900 dark:text-zinc-100"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              onClick={resetModal}
              className="absolute right-4 top-4 text-zinc-400 transition-colors hover:text-zinc-600 dark:hover:text-zinc-200"
            >
              <X className="h-5 w-5" />
            </button>
            <h2 className="mb-6 text-xl font-semibold dark:text-white">Add Movie</h2>

            <div className="mb-6 flex gap-2 rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
              <button
                className={`flex-1 rounded-md py-2 text-sm font-medium transition-colors ${
                  addMethod === "csv"
                    ? "bg-white text-zinc-900 shadow dark:bg-zinc-700 dark:text-white"
                    : "text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                }`}
                onClick={() => setAddMethod("csv")}
              >
                Upload CSV
              </button>
              <button
                className={`flex-1 rounded-md py-2 text-sm font-medium transition-colors ${
                  addMethod === "manual"
                    ? "bg-white text-zinc-900 shadow dark:bg-zinc-700 dark:text-white"
                    : "text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                }`}
                onClick={() => setAddMethod("manual")}
              >
                Add Manually
              </button>
            </div>

            {addMethod === "csv" ? (
              <div className="space-y-4">
                <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-zinc-200 bg-zinc-50 p-8 text-center dark:border-zinc-700 dark:bg-zinc-950/50">
                  <Upload className="mb-3 h-5 w-5 text-zinc-500 dark:text-zinc-400" />
                  <span className="mb-1 text-sm text-zinc-600 dark:text-zinc-300">
                    {csvFile ? csvFile.name : "Choose a CSV file to import"}
                  </span>
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">
                    CSV columns: title, genres, rating, year, poster
                  </span>
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={(event) => {
                      setCsvFile(event.target.files?.[0] ?? null);
                    }}
                  />
                </label>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Movie Title
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Inception"
                    value={form.title}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, title: event.target.value }))
                    }
                    className="flex h-10 w-full rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:placeholder:text-zinc-500 dark:focus:ring-zinc-100"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Genre
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Action Sci-Fi"
                    value={form.genre}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, genre: event.target.value }))
                    }
                    className="flex h-10 w-full rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:placeholder:text-zinc-500 dark:focus:ring-zinc-100"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                      Rating
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="10"
                      step="0.1"
                      placeholder="8.5"
                      value={form.rating}
                      onChange={(event) =>
                        setForm((current) => ({ ...current, rating: event.target.value }))
                      }
                      className="flex h-10 w-full rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:placeholder:text-zinc-500 dark:focus:ring-zinc-100"
                    />
                  </div>
                  <div>
                    <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                      Release Year
                    </label>
                    <input
                      type="number"
                      placeholder="2024"
                      value={form.year}
                      onChange={(event) =>
                        setForm((current) => ({ ...current, year: event.target.value }))
                      }
                      className="flex h-10 w-full rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:placeholder:text-zinc-500 dark:focus:ring-zinc-100"
                    />
                  </div>
                </div>
                <div>
                  <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    Poster URL (optional)
                  </label>
                  <input
                    type="url"
                    placeholder="https://..."
                    value={form.poster}
                    onChange={(event) =>
                      setForm((current) => ({ ...current, poster: event.target.value }))
                    }
                    className="flex h-10 w-full rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:placeholder:text-zinc-500 dark:focus:ring-zinc-100"
                  />
                </div>
              </div>
            )}

            <div className="mt-8 flex justify-end gap-3">
              <Button variant="ghost" onClick={resetModal} disabled={submitting}>
                Cancel
              </Button>
              <Button
                onClick={handleAdd}
                disabled={
                  submitting ||
                  (addMethod === "manual" && !form.title.trim()) ||
                  (addMethod === "csv" && !csvFile)
                }
              >
                {submitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : addMethod === "manual" ? (
                  "Add Movie"
                ) : (
                  "Upload CSV"
                )}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
