const EXPLICIT_API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "";
const API_BASE_STORAGE_KEY = "moviebuzz-api-base-url";

function normalizeBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

function readStoredApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return window.localStorage.getItem(API_BASE_STORAGE_KEY)?.trim() || "";
}

function storeApiBaseUrl(value: string) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(API_BASE_STORAGE_KEY, value);
}

function getApiBaseCandidates(): string[] {
  const candidates: string[] = [];
  const seen = new Set<string>();

  const add = (value: string) => {
    const normalized = normalizeBaseUrl(value);
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    candidates.push(normalized);
  };

  add(readStoredApiBaseUrl());
  add(EXPLICIT_API_BASE_URL);

  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    const peerHosts =
      hostname === "localhost"
        ? ["localhost", "127.0.0.1"]
        : hostname === "127.0.0.1"
          ? ["127.0.0.1", "localhost"]
          : [hostname];

    for (const host of peerHosts) {
      add(`${protocol}//${host}:8000`);
      add(`${protocol}//${host}:8001`);
    }
  }

  add("http://127.0.0.1:8000");
  add("http://127.0.0.1:8001");
  add("http://localhost:8000");
  add("http://localhost:8001");

  return candidates;
}

export type Movie = {
  movie_key: string;
  movie_id?: number | null;
  title: string;
  clean_title?: string;
  year?: string;
  genres?: string;
  poster?: string;
  plot?: string;
  description?: string;
  cast?: string;
  director?: string;
  imdb_rating?: string;
  runtime?: string;
  rating?: number | string;
  trending_score?: number;
  youtube_link?: string;
  created_at?: string;
  source?: string;
  source_label?: string;
  can_delete?: boolean;
};

export type AdminOverview = {
  total_users: number;
  verified_users: number;
  catalog_movies: number;
  wishlist_items: number;
};

export type AdminModelMetrics = {
  run_id?: string;
  model?: string;
  updated_at?: string;
  report_generated_at?: string;
  test_ratio?: number;
  ncf_auc?: number;
  ncf_bce?: number;
  ncf_bpr?: number;
  ncf_f1?: number;
  ncf_precision?: number;
  ncf_recall?: number;
  svd_mse?: number;
  ndcg_10?: number;
  xgb_auc?: number;
  xgb_f1?: number;
  xgb_logloss?: number;
  available_models?: string[];
  missing_models?: string[];
  comparison?: Array<{
    model: string;
    auc?: number | null;
    f1?: number | null;
    precision?: number | null;
    recall?: number | null;
    loss?: number | null;
    loss_label?: string;
  }>;
  report_metrics?: Record<string, Record<string, number>>;
};

export type AdminMetricPlotKind = "comparison" | "loss" | "availability";

export type AdminUser = {
  id: number;
  name: string;
  email: string;
  verified: number | boolean;
  role?: "user" | "mod" | "admin";
  created_at?: string;
};

export type UserPreferences = {
  age?: number | null;
  preferred_genres?: string[];
  preferred_moods?: string[];
};

export type TrailerResponse = {
  movie_id: number;
  title: string;
  year?: string | null;
  video_id?: string | null;
  embed_url?: string | null;
  found: boolean;
};

export class ApiError extends Error {
  status?: number;
  isNetworkError: boolean;

  constructor(
    message: string,
    options?: { status?: number; isNetworkError?: boolean },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options?.status;
    this.isNetworkError = options?.isNetworkError ?? false;
  }
}

type ApiResponse = {
  success?: boolean;
  msg?: string;
  detail?: string;
  [key: string]: unknown;
};

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers ?? {});
  const isFormData =
    typeof FormData !== "undefined" && init?.body instanceof FormData;

  if (!isFormData && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const requestInit: RequestInit = {
    ...init,
    headers,
  };

  let lastError: unknown = null;

  for (const baseUrl of getApiBaseCandidates()) {
    try {
      const response = await fetch(`${baseUrl}${path}`, requestInit);
      storeApiBaseUrl(baseUrl);
      return response;
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError ?? new Error("Backend is unavailable");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await apiFetch(path, init);
  } catch {
    throw new ApiError("Backend is unavailable", { isNetworkError: true });
  }

  let data: ApiResponse | null = null;

  try {
    data = (await response.json()) as ApiResponse;
  } catch {
    data = null;
  }

  const message =
    data?.msg || data?.detail || `Request failed with status ${response.status}`;

  if (!response.ok) {
    throw new ApiError(message, { status: response.status });
  }

  if (data && typeof data.success === "boolean" && !data.success) {
    throw new ApiError(message, { status: response.status });
  }

  return (data as T) ?? ({} as T);
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  let response: Response;

  try {
    response = await apiFetch(path, init);
  } catch {
    throw new ApiError("Backend is unavailable", { isNetworkError: true });
  }

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;

    try {
      const data = (await response.json()) as ApiResponse;
      message = data?.msg || data?.detail || message;
    } catch {
      // Keep the default message when the response is not JSON.
    }

    throw new ApiError(message, { status: response.status });
  }

  return response.blob();
}

export type LoginResponse = {
  success: boolean;
  msg: string;
  name: string;
  email: string;
  role?: "user" | "mod" | "admin";
  age?: number | null;
  preferred_genres?: string[];
  preferred_moods?: string[];
};

type MessageResponse = {
  success: boolean;
  msg: string;
};

type MovieListResponse = {
  success?: boolean;
  msg?: string;
  results?: Movie[];
  items?: Movie[];
  total?: number;
  limit?: number;
  offset?: number;
  has_more?: boolean;
  genres?: string[];
};

type RecommendResponse = {
  resolved_title?: string;
  results?: Movie[];
};

type AdminUsersResponse = AdminUser[];

export type AdminMovieList = {
  items: Movie[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  genres: string[];
};

export function loginUser(email: string, password: string) {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function signupUser(
  name: string,
  email: string,
  password: string,
  preferences?: UserPreferences,
) {
  return request<{ success: boolean; msg: string }>("/auth/signup", {
    method: "POST",
    body: JSON.stringify({
      name,
      email,
      password,
      age: preferences?.age ?? null,
      preferred_genres: preferences?.preferred_genres ?? [],
      preferred_moods: preferences?.preferred_moods ?? [],
    }),
  });
}

export function verifyUserOtp(email: string, otp: string) {
  return request<{
    success: boolean;
    msg: string;
    welcome_email_sent?: boolean;
    next_target?: string;
  }>("/auth/verify-otp", {
    method: "POST",
    body: JSON.stringify({ email, otp }),
  });
}

export function resendUserOtp(email: string) {
  return request<{ success: boolean; msg: string }>("/auth/resend-otp", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function requestPasswordResetOtp(email: string) {
  return request<MessageResponse>("/auth/forgot-password/request-otp", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function verifyPasswordResetOtp(email: string, otp: string) {
  return request<MessageResponse>("/auth/forgot-password/verify-otp", {
    method: "POST",
    body: JSON.stringify({ email, otp }),
  });
}

export function resetPassword(email: string, otp: string, newPassword: string) {
  return request<MessageResponse>("/auth/forgot-password/reset", {
    method: "POST",
    body: JSON.stringify({ email, otp, new_password: newPassword }),
  });
}

export function requestDeleteAccountOtp(email: string) {
  return request<MessageResponse>("/auth/delete/request-otp", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export function confirmDeleteAccount(email: string, otp: string) {
  return request<MessageResponse>("/auth/delete/confirm", {
    method: "POST",
    body: JSON.stringify({ email, otp }),
  });
}

export async function getHomeMovies(limit = 50, genre?: string, userEmail?: string) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (genre?.trim()) {
    params.set("genre", genre.trim());
  }
  if (userEmail?.trim()) {
    params.set("user_email", userEmail.trim().toLowerCase());
  }
  const data = await request<MovieListResponse>(`/movies/home?${params.toString()}`);
  return data.results ?? [];
}

export function searchCatalogMovies(query: string, limit = 50) {
  return request<Movie[]>(
    `/search?q=${encodeURIComponent(query)}&limit=${limit}`,
  );
}

export async function recommendCatalogMovies(
  title: string,
  userId = 1,
  limit = 50,
  options?: {
    mood?: string;
    userEmail?: string;
  },
) {
  const params = new URLSearchParams({
    title,
    user_id: String(userId),
    limit: String(limit),
  });
  if (options?.mood?.trim()) {
    params.set("mood", options.mood.trim());
  }
  if (options?.userEmail?.trim()) {
    params.set("user_email", options.userEmail.trim().toLowerCase());
  }
  const data = await request<RecommendResponse>(
    `/recommend?${params.toString()}`,
  );
  return data.results ?? [];
}

export function getUserPreferences(email: string) {
  return request<{
    success: boolean;
    name?: string;
    email?: string;
    age?: number | null;
    preferred_genres?: string[];
    preferred_moods?: string[];
  }>(`/auth/preferences/${encodeURIComponent(email.trim().toLowerCase())}`);
}

export function saveUserPreferences(email: string, preferences: UserPreferences) {
  return request<{
    success: boolean;
    msg: string;
    name?: string;
    email?: string;
    age?: number | null;
    preferred_genres?: string[];
    preferred_moods?: string[];
  }>("/auth/preferences", {
    method: "POST",
    body: JSON.stringify({
      email: email.trim().toLowerCase(),
      age: preferences.age ?? null,
      preferred_genres: preferences.preferred_genres ?? [],
      preferred_moods: preferences.preferred_moods ?? [],
    }),
  });
}

export async function getWishlist(email: string) {
  const data = await request<MovieListResponse>(
    `/auth/wishlist/${encodeURIComponent(email)}`,
  );
  return data.items ?? [];
}

export function addWishlistMovie(email: string, movie: Movie) {
  return request<MessageResponse>("/auth/wishlist", {
    method: "POST",
    body: JSON.stringify({ email, movie }),
  });
}

export function removeWishlistMovie(email: string, movieKey: string) {
  return request<MessageResponse>("/auth/wishlist/remove", {
    method: "POST",
    body: JSON.stringify({ email, movie_key: movieKey }),
  });
}

export function getAdminOverview() {
  return request<AdminOverview>("/admin/overview");
}

export function getAdminModelMetrics() {
  return request<AdminModelMetrics>("/admin/model-metrics");
}

export function getAdminModelMetricPlot(
  kind: AdminMetricPlotKind,
  theme: "light" | "dark" = "light",
) {
  const params = new URLSearchParams({
    kind,
    theme,
  });
  return requestBlob(`/admin/model-metrics/plot?${params.toString()}`);
}

export function getAdminUsers() {
  return request<AdminUsersResponse>("/auth/admin/users");
}

export function deleteAdminUser(email: string) {
  return request<MessageResponse>(`/auth/admin/users/${encodeURIComponent(email)}`, {
    method: "DELETE",
  });
}

export async function getAdminMovies(options?: {
  limit?: number;
  offset?: number;
  search?: string;
  genre?: string;
}) {
  const params = new URLSearchParams();
  if (typeof options?.limit === "number") {
    params.set("limit", String(options.limit));
  }
  if (typeof options?.offset === "number" && options.offset > 0) {
    params.set("offset", String(options.offset));
  }
  if (options?.search?.trim()) {
    params.set("search", options.search.trim());
  }
  if (options?.genre?.trim()) {
    params.set("genre", options.genre.trim());
  }

  const suffix = params.size ? `?${params.toString()}` : "";
  const data = await request<MovieListResponse>(`/admin/movies${suffix}`);
  return {
    items: data.items ?? [],
    total: data.total ?? 0,
    limit: data.limit ?? options?.limit ?? 0,
    offset: data.offset ?? options?.offset ?? 0,
    has_more: Boolean(data.has_more),
    genres: data.genres ?? [],
  } satisfies AdminMovieList;
}

export function addAdminMoviesManual(movies: Array<{
  title: string;
  genres?: string;
  rating?: number;
  year?: string;
  poster?: string;
}>) {
  return request<{ inserted: number; status: string }>("/admin/movies/manual", {
    method: "POST",
    body: JSON.stringify(movies),
  });
}

export function uploadAdminMoviesCsv(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  return request<{ inserted: number; status: string; filename: string }>(
    "/admin/movies/csv",
    {
      method: "POST",
      body: formData,
    },
  );
}

export function deleteAdminMovie(movieId: number) {
  return request<MessageResponse>(`/admin/movies/${movieId}`, {
    method: "DELETE",
  });
}

export function getMovieTrailer(movieId: number) {
  return request<TrailerResponse>(`/api/trailer/${movieId}`);
}
