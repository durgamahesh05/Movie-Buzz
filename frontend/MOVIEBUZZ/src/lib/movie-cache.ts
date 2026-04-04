export const HOME_CACHE_PREFIX = "moviebuzz-home-movies-v4-";
export const HOME_CACHE_INVALIDATED_EVENT = "moviebuzz:home-cache-invalidated";
const HOME_CACHE_REFRESH_KEY = "moviebuzz-home-movies-refresh";

export function getHomeCacheKey(genre: string, userEmail = "") {
  const normalizedGenre =
    genre.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-") || "all";
  const normalizedUser =
    userEmail.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-") || "guest";
  return `${HOME_CACHE_PREFIX}${normalizedUser}-${normalizedGenre}`;
}

export function isHomeCacheRefreshKey(key: string | null) {
  return key === HOME_CACHE_REFRESH_KEY;
}

export function invalidateHomeMovieCache() {
  if (typeof window === "undefined") {
    return;
  }

  const keysToRemove: string[] = [];
  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index);
    if (key?.startsWith(HOME_CACHE_PREFIX)) {
      keysToRemove.push(key);
    }
  }

  for (const key of keysToRemove) {
    window.localStorage.removeItem(key);
  }

  const refreshToken = String(Date.now());
  window.localStorage.setItem(HOME_CACHE_REFRESH_KEY, refreshToken);
  window.dispatchEvent(
    new CustomEvent(HOME_CACHE_INVALIDATED_EVENT, {
      detail: refreshToken,
    }),
  );
}
