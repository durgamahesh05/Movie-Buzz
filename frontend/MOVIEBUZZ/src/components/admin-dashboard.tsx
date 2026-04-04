import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AdminSidebar } from "./admin-sidebar";
import { StatsCards } from "./stats-cards";
import { UsersTable } from "./users-table";
import { MoviesTable } from "./movies-table";
import { AdminNavbar } from "./admin-navbar";
import {
  ApiError,
  getAdminModelMetricPlot,
  getAdminModelMetrics,
  type AdminMetricPlotKind,
  type AdminModelMetrics,
} from "../lib/api";
import { useAppStore } from "../store/appStore";

type ViewType = "dashboard" | "users" | "movies" | "analytics";

const EMPTY_MODEL_METRICS: AdminModelMetrics = {};
const EMPTY_METRIC_PLOTS: Record<AdminMetricPlotKind, string> = {
  comparison: "",
  loss: "",
  availability: "",
};
const MODEL_PLOT_KINDS: AdminMetricPlotKind[] = ["comparison", "loss", "availability"];

function revokePlotUrls(plotUrls: Record<AdminMetricPlotKind, string>) {
  if (typeof URL === "undefined") {
    return;
  }
  for (const plotUrl of Object.values(plotUrls)) {
    if (plotUrl) {
      URL.revokeObjectURL(plotUrl);
    }
  }
}

function formatMetric(value?: number, digits = 4) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "N/A";
}

function formatPercentMetric(value?: number, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "N/A";
}

function formatMetricsTimestamp(value?: string) {
  if (!value) {
    return "No saved training run yet";
  }

  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString();
}

function renderChartFallback(message: string) {
  return (
    <div className="flex h-72 items-center justify-center rounded-lg border border-dashed border-zinc-200 bg-zinc-50 text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-950/60 dark:text-zinc-400">
      {message}
    </div>
  );
}

export function AdminDashboard() {
  const navigate = useNavigate();
  const { isDark, toggleTheme, logout, user } = useAppStore();
  const [activeView, setActiveView] = useState<ViewType>("dashboard");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [catalogRefreshToken, setCatalogRefreshToken] = useState(0);
  const [modelMetrics, setModelMetrics] = useState<AdminModelMetrics>(EMPTY_MODEL_METRICS);
  const [plotUrls, setPlotUrls] =
    useState<Record<AdminMetricPlotKind, string>>(EMPTY_METRIC_PLOTS);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState("");

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const handleCatalogChange = () => {
    setCatalogRefreshToken((current) => current + 1);
  };

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }

    const root = document.documentElement;
    const { body } = document;
    const hadRootDarkClass = root.classList.contains("dark");
    const hadDarkClass = body.classList.contains("dark");
    const previousRootColorScheme = root.style.colorScheme;
    const previousColorScheme = body.style.colorScheme;

    root.classList.toggle("dark", isDark);
    body.classList.toggle("dark", isDark);
    root.style.colorScheme = isDark ? "dark" : "light";
    body.style.colorScheme = isDark ? "dark" : "light";

    return () => {
      root.classList.toggle("dark", hadRootDarkClass);
      body.classList.toggle("dark", hadDarkClass);
      root.style.colorScheme = previousRootColorScheme;
      body.style.colorScheme = previousColorScheme;
    };
  }, [isDark]);

  useEffect(() => {
    let cancelled = false;

    if (activeView !== "analytics") {
      return () => {
        cancelled = true;
      };
    }

    const loadModelMetrics = async () => {
      setMetricsLoading(true);
      try {
        const theme = isDark ? "dark" : "light";
        const [metrics, plotResults] = await Promise.all([
          getAdminModelMetrics(),
          Promise.allSettled(
            MODEL_PLOT_KINDS.map((kind) => getAdminModelMetricPlot(kind, theme)),
          ),
        ]);

        const nextPlotUrls: Record<AdminMetricPlotKind, string> = {
          comparison: "",
          loss: "",
          availability: "",
        };

        MODEL_PLOT_KINDS.forEach((kind, index) => {
          const result = plotResults[index];
          if (result?.status === "fulfilled") {
            nextPlotUrls[kind] = URL.createObjectURL(result.value);
          }
        });

        if (!cancelled) {
          setModelMetrics(metrics);
          setPlotUrls(nextPlotUrls);
          setMetricsError("");
        } else {
          revokePlotUrls(nextPlotUrls);
        }
      } catch (loadError) {
        if (!cancelled) {
          setModelMetrics(EMPTY_MODEL_METRICS);
          setPlotUrls(EMPTY_METRIC_PLOTS);
          setMetricsError(
            loadError instanceof ApiError
              ? loadError.message
              : "Unable to load model metrics",
          );
        }
      } finally {
        if (!cancelled) {
          setMetricsLoading(false);
        }
      }
    };

    void loadModelMetrics();

    return () => {
      cancelled = true;
    };
  }, [activeView, isDark]);

  useEffect(() => {
    return () => {
      revokePlotUrls(plotUrls);
    };
  }, [plotUrls]);

  const summaryCards = [
    {
      title: "NCF AUC",
      value: formatPercentMetric(modelMetrics.ncf_auc),
      subtitle: "Ranking quality from the latest saved evaluation report",
    },
    {
      title: "NCF BCE",
      value: formatMetric(modelMetrics.ncf_bce),
      subtitle: "Lower is better for the neural collaborative model",
    },
    {
      title: "XGB AUC",
      value: formatPercentMetric(modelMetrics.xgb_auc),
      subtitle: "Tree re-ranker quality on the evaluation split",
    },
    {
      title: "XGB LogLoss",
      value: formatMetric(modelMetrics.xgb_logloss),
      subtitle: "Probability calibration loss for the XGBoost stage",
    },
  ];

  const comparisonData = (modelMetrics.comparison ?? []).map((entry) => ({
    model: entry.model,
    auc: typeof entry.auc === "number" ? Number((entry.auc * 100).toFixed(2)) : 0,
    f1: typeof entry.f1 === "number" ? Number((entry.f1 * 100).toFixed(2)) : 0,
    precision:
      typeof entry.precision === "number"
        ? Number((entry.precision * 100).toFixed(2))
        : 0,
    recall:
      typeof entry.recall === "number" ? Number((entry.recall * 100).toFixed(2)) : 0,
    loss: typeof entry.loss === "number" ? Number(entry.loss.toFixed(4)) : 0,
    lossLabel: entry.loss_label || "Loss",
  }));

  const lossChartData = comparisonData.filter((entry) => entry.loss > 0);
  const hasLiveStackData = Boolean(
    modelMetrics.available_models?.length || modelMetrics.missing_models?.length,
  );
  const availableModelsLabel =
    modelMetrics.available_models?.length
      ? modelMetrics.available_models.join(", ")
      : "N/A";
  const missingModelsLabel =
    modelMetrics.missing_models?.length
      ? modelMetrics.missing_models.join(", ")
      : "None";

  const renderView = () => {
    switch (activeView) {
      case "dashboard":
        return (
          <div className="space-y-8">
            <div>
              <h1 className="text-3xl font-bold dark:text-white">Dashboard Overview</h1>
              <p className="mt-1 text-zinc-600 dark:text-zinc-400">
                Monitor your platform's performance and activity
              </p>
            </div>

            <StatsCards refreshToken={catalogRefreshToken} />

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div>
                <h2 className="mb-4 text-xl font-bold dark:text-white">Recent Users</h2>
                <UsersTable limit={5} />
              </div>
              <div>
                <h2 className="mb-4 text-xl font-bold dark:text-white">Recent Movies</h2>
                <MoviesTable limit={5} refreshToken={catalogRefreshToken} />
              </div>
            </div>
          </div>
        );

      case "users":
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold dark:text-white">Manage Users</h1>
              <p className="mt-1 text-zinc-600 dark:text-zinc-400">
                View and manage all registered users
              </p>
            </div>
            <UsersTable />
          </div>
        );

      case "movies":
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold dark:text-white">Manage Movies</h1>
              <p className="mt-1 text-zinc-600 dark:text-zinc-400">
                Add, edit, or remove movies from the platform
              </p>
            </div>
            <MoviesTable
              refreshToken={catalogRefreshToken}
              onCatalogChange={handleCatalogChange}
            />
          </div>
        );

      case "analytics":
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold dark:text-white">Analytics</h1>
              <p className="mt-1 text-zinc-600 dark:text-zinc-400">
                Latest training insights and recommendation quality signals
              </p>
            </div>

            <StatsCards refreshToken={catalogRefreshToken} />

            {metricsError ? (
              <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-950/60 dark:bg-rose-950/30 dark:text-rose-200">
                {metricsError}
              </div>
            ) : null}

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
              {summaryCards.map((metric) => (
                <div
                  key={metric.title}
                  className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
                >
                  <p className="text-sm text-zinc-500 dark:text-zinc-400">
                    {metric.title}
                  </p>
                  <div className="mt-3 text-3xl font-bold tracking-tight text-zinc-950 dark:text-white">
                    {metricsLoading ? "Loading..." : metric.value}
                  </div>
                  <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">
                    {metric.subtitle}
                  </p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                <h2 className="text-lg font-semibold dark:text-white">Latest Training Run</h2>
                <div className="mt-4 space-y-3 text-sm text-zinc-600 dark:text-zinc-400">
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      Run ID:
                    </span>{" "}
                    {metricsLoading ? "Loading..." : modelMetrics.run_id || "N/A"}
                  </p>
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      Updated:
                    </span>{" "}
                    {metricsLoading
                      ? "Loading..."
                      : formatMetricsTimestamp(
                          modelMetrics.report_generated_at || modelMetrics.updated_at,
                        )}
                  </p>
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      Available models:
                    </span>{" "}
                    {metricsLoading ? "Loading..." : availableModelsLabel}
                  </p>
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      Missing models:
                    </span>{" "}
                    {metricsLoading ? "Loading..." : missingModelsLabel}
                  </p>
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      Test split:
                    </span>{" "}
                    {metricsLoading
                      ? "Loading..."
                      : formatPercentMetric(modelMetrics.test_ratio, 0)}
                  </p>
                </div>
              </div>

              <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                <h2 className="text-lg font-semibold dark:text-white">
                  Recommendation Health
                </h2>
                <p className="mt-4 text-sm leading-6 text-zinc-600 dark:text-zinc-400">
                  Search now resolves franchise titles more accurately before the
                  recommender picks similar movies, and the analytics panel is reading the
                  latest saved evaluation report instead of showing placeholder values.
                </p>
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-zinc-600 dark:text-zinc-400">
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      NCF F1:
                    </span>{" "}
                    {metricsLoading ? "Loading..." : formatPercentMetric(modelMetrics.ncf_f1)}
                  </p>
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      NCF Recall:
                    </span>{" "}
                    {metricsLoading
                      ? "Loading..."
                      : formatPercentMetric(modelMetrics.ncf_recall)}
                  </p>
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      NCF Precision:
                    </span>{" "}
                    {metricsLoading
                      ? "Loading..."
                      : formatPercentMetric(modelMetrics.ncf_precision)}
                  </p>
                  <p>
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      XGB F1:
                    </span>{" "}
                    {metricsLoading ? "Loading..." : formatPercentMetric(modelMetrics.xgb_f1)}
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
              <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                <div className="mb-4">
                  <h2 className="text-lg font-semibold dark:text-white">
                    Model Quality Comparison
                  </h2>
                  <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                    AUC, F1, precision, and recall from the latest saved evaluation.
                  </p>
                </div>

                {metricsLoading ? (
                  renderChartFallback("Loading comparison graph...")
                ) : plotUrls.comparison ? (
                  <div className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-800 dark:bg-zinc-950/60">
                    <img
                      src={plotUrls.comparison}
                      alt="Model quality comparison graph"
                      className="h-auto w-full rounded-md"
                    />
                  </div>
                ) : comparisonData.length ? (
                  renderChartFallback("Matplotlib comparison graph is unavailable.")
                ) : (
                  renderChartFallback("No saved comparison metrics are available yet.")
                )}
              </div>

              <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                <div className="mb-4">
                  <h2 className="text-lg font-semibold dark:text-white">Loss Snapshot</h2>
                  <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                    Latest saved loss values for the trained ranking stages.
                  </p>
                </div>

                {metricsLoading ? (
                  renderChartFallback("Loading loss graph...")
                ) : plotUrls.loss ? (
                  <div className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-800 dark:bg-zinc-950/60">
                    <img
                      src={plotUrls.loss}
                      alt="Ranking loss graph"
                      className="h-auto w-full rounded-md"
                    />
                  </div>
                ) : lossChartData.length ? (
                  renderChartFallback("Matplotlib loss graph is unavailable.")
                ) : (
                  renderChartFallback("No saved loss metrics are available yet.")
                )}
              </div>

              <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
                <div className="mb-4">
                  <h2 className="text-lg font-semibold dark:text-white">
                    Live Recommender Stack
                  </h2>
                  <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                    Availability of the models currently loaded by the backend engine.
                  </p>
                </div>

                {metricsLoading ? (
                  renderChartFallback("Loading live engine graph...")
                ) : plotUrls.availability ? (
                  <div className="overflow-hidden rounded-lg border border-zinc-200 bg-zinc-50 p-2 dark:border-zinc-800 dark:bg-zinc-950/60">
                    <img
                      src={plotUrls.availability}
                      alt="Live recommender stack graph"
                      className="h-auto w-full rounded-md"
                    />
                  </div>
                ) : hasLiveStackData ? (
                  renderChartFallback("Matplotlib live engine graph is unavailable.")
                ) : (
                  renderChartFallback("No live engine availability metrics are available yet.")
                )}
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-950 transition-colors duration-300 dark:bg-zinc-950 dark:text-zinc-50">
      <div className="min-h-screen">
        <AdminNavbar
          onLogout={handleLogout}
          darkMode={isDark}
          isSidebarOpen={isSidebarOpen}
          onToggleDark={toggleTheme}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
          userName={user?.name}
          userEmail={user?.email}
        />

        <div className="flex w-full">
          <AdminSidebar
            activeView={activeView}
            onViewChange={setActiveView}
            isOpen={isSidebarOpen}
            isDark={isDark}
          />

          <main className="min-w-0 w-full flex-1 p-8 transition-all duration-300">
            {renderView()}
          </main>
        </div>
      </div>
    </div>
  );
}
