import { type ReactNode, useEffect, useState } from "react";
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

interface PageHeaderProps {
  title: string;
  subtitle: string;
  text: string;
  subtext: string;
}

function PageHeader({ title, subtitle, text, subtext }: PageHeaderProps) {
  return (
    <div style={{ marginBottom: "28px" }}>
      <h1
        style={{
          fontSize: "26px",
          fontWeight: 900,
          color: text,
          margin: 0,
          letterSpacing: "-0.025em",
        }}
      >
        {title}
      </h1>
      <p style={{ fontSize: "14px", color: subtext, margin: 0, marginTop: "4px" }}>
        {subtitle}
      </p>
    </div>
  );
}

function ChartFallback({
  message,
  borderColor,
  isDark,
  subtext,
}: {
  message: string;
  borderColor: string;
  isDark: boolean;
  subtext: string;
}) {
  return (
    <div
      style={{
        height: "288px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: "12px",
        border: `1px dashed ${borderColor}`,
        backgroundColor: isDark ? "rgba(255,255,255,0.02)" : "#fafafa",
        fontSize: "13px",
        color: subtext,
      }}
    >
      {message}
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: string;
  subtitle: string;
  color: string;
  bgColor: string;
  cardBg: string;
  borderColor: string;
  text: string;
  subtext: string;
  isDark: boolean;
  loading: boolean;
}

function MetricCard({
  title,
  value,
  subtitle,
  color,
  bgColor,
  cardBg,
  borderColor,
  text,
  subtext,
  isDark,
  loading,
}: MetricCardProps) {
  return (
    <div
      style={{
        backgroundColor: cardBg,
        border: `1px solid ${borderColor}`,
        borderRadius: "16px",
        padding: "20px",
        boxShadow: isDark ? "none" : "0 1px 4px rgba(0,0,0,0.06)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "12px",
        }}
      >
        <p
          style={{
            fontSize: "12px",
            fontWeight: 600,
            color: subtext,
            margin: 0,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          {title}
        </p>
        <div
          style={{
            width: "24px",
            height: "24px",
            borderRadius: "999px",
            backgroundColor: bgColor,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: color,
              boxShadow: `0 0 6px ${color}`,
            }}
          />
        </div>
      </div>
      <div style={{ fontSize: "30px", fontWeight: 800, color: text, letterSpacing: "-0.03em" }}>
        {loading ? <span style={{ fontSize: "14px", color: subtext }}>Loading...</span> : value}
      </div>
      <p style={{ fontSize: "12px", color: subtext, margin: 0, marginTop: "6px" }}>
        {subtitle}
      </p>
    </div>
  );
}

function InfoCard({
  title,
  children,
  cardBg,
  borderColor,
  text,
  isDark,
}: {
  title: string;
  children: ReactNode;
  cardBg: string;
  borderColor: string;
  text: string;
  isDark: boolean;
}) {
  return (
    <div
      style={{
        backgroundColor: cardBg,
        border: `1px solid ${borderColor}`,
        borderRadius: "16px",
        padding: "22px",
        boxShadow: isDark ? "none" : "0 1px 4px rgba(0,0,0,0.06)",
      }}
    >
      <h2
        style={{
          fontSize: "15px",
          fontWeight: 700,
          color: text,
          margin: 0,
          marginBottom: "16px",
          paddingBottom: "12px",
          borderBottom: `1px solid ${borderColor}`,
        }}
      >
        {title}
      </h2>
      {children}
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

  const bg = isDark ? "#09090b" : "#f9f9fb";
  const cardBg = isDark ? "#18181b" : "#ffffff";
  const border = isDark ? "rgba(255,255,255,0.08)" : "#e4e4e7";
  const text = isDark ? "#f4f4f5" : "#09090b";
  const subtext = "#71717a";

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
      subtitle: "Ranking quality from latest evaluation",
      color: "#3b82f6",
      bgColor: isDark ? "rgba(59,130,246,0.12)" : "#eff6ff",
    },
    {
      title: "NCF BCE",
      value: formatMetric(modelMetrics.ncf_bce),
      subtitle: "Lower is better - neural collaborative",
      color: "#10b981",
      bgColor: isDark ? "rgba(16,185,129,0.12)" : "#f0fdf4",
    },
    {
      title: "XGB AUC",
      value: formatPercentMetric(modelMetrics.xgb_auc),
      subtitle: "Tree re-ranker quality on eval split",
      color: "#8b5cf6",
      bgColor: isDark ? "rgba(139,92,246,0.12)" : "#f5f3ff",
    },
    {
      title: "XGB LogLoss",
      value: formatMetric(modelMetrics.xgb_logloss),
      subtitle: "Probability calibration loss for XGBoost",
      color: "#f59e0b",
      bgColor: isDark ? "rgba(245,158,11,0.12)" : "#fffbeb",
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
          <div>
            <PageHeader
              title="Dashboard Overview"
              subtitle="Monitor your platform's performance and activity"
              text={text}
              subtext={subtext}
            />
            <StatsCards refreshToken={catalogRefreshToken} />

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
                gap: "24px",
                marginTop: "28px",
              }}
            >
              <div>
                <h2
                  style={{
                    fontSize: "16px",
                    fontWeight: 700,
                    color: text,
                    margin: 0,
                    marginBottom: "14px",
                  }}
                >
                  Recent Users
                </h2>
                <UsersTable limit={5} />
              </div>
              <div>
                <h2
                  style={{
                    fontSize: "16px",
                    fontWeight: 700,
                    color: text,
                    margin: 0,
                    marginBottom: "14px",
                  }}
                >
                  Recent Movies
                </h2>
                <MoviesTable limit={5} refreshToken={catalogRefreshToken} />
              </div>
            </div>
          </div>
        );

      case "users":
        return (
          <div>
            <PageHeader
              title="Manage Users"
              subtitle="View and manage all registered users"
              text={text}
              subtext={subtext}
            />
            <UsersTable />
          </div>
        );

      case "movies":
        return (
          <div>
            <PageHeader
              title="Manage Movies"
              subtitle="Add, edit, or remove movies from the platform"
              text={text}
              subtext={subtext}
            />
            <MoviesTable
              refreshToken={catalogRefreshToken}
              onCatalogChange={handleCatalogChange}
            />
          </div>
        );

      case "analytics":
        return (
          <div>
            <PageHeader
              title="Analytics"
              subtitle="Latest training insights and recommendation quality signals"
              text={text}
              subtext={subtext}
            />

            <StatsCards refreshToken={catalogRefreshToken} />

            {metricsError ? (
              <div
                style={{
                  marginTop: "20px",
                  padding: "14px 16px",
                  borderRadius: "12px",
                  border: "1px solid rgba(239,68,68,0.3)",
                  backgroundColor: isDark ? "rgba(239,68,68,0.1)" : "#fef2f2",
                  color: isDark ? "#fca5a5" : "#b91c1c",
                  fontSize: "13px",
                }}
              >
                {metricsError}
              </div>
            ) : null}

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: "16px",
                marginTop: "24px",
              }}
            >
              {summaryCards.map((metric) => (
                <MetricCard
                  key={metric.title}
                  {...metric}
                  cardBg={cardBg}
                  borderColor={border}
                  text={text}
                  subtext={subtext}
                  isDark={isDark}
                  loading={metricsLoading}
                />
              ))}
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
                gap: "16px",
                marginTop: "20px",
              }}
            >
              <InfoCard
                title="Latest Training Run"
                cardBg={cardBg}
                borderColor={border}
                text={text}
                isDark={isDark}
              >
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "10px",
                    fontSize: "13px",
                  }}
                >
                  {[
                    ["Run ID", metricsLoading ? "Loading..." : modelMetrics.run_id || "N/A"],
                    [
                      "Updated",
                      metricsLoading
                        ? "Loading..."
                        : formatMetricsTimestamp(
                            modelMetrics.report_generated_at || modelMetrics.updated_at,
                          ),
                    ],
                    [
                      "Available models",
                      metricsLoading ? "Loading..." : availableModelsLabel,
                    ],
                    [
                      "Missing models",
                      metricsLoading ? "Loading..." : missingModelsLabel,
                    ],
                    [
                      "Test split",
                      metricsLoading
                        ? "Loading..."
                        : formatPercentMetric(modelMetrics.test_ratio, 0),
                    ],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      style={{ display: "flex", justifyContent: "space-between", gap: "12px" }}
                    >
                      <span style={{ color: subtext, flexShrink: 0 }}>{label}</span>
                      <span
                        style={{ color: text, fontWeight: 600, textAlign: "right" }}
                      >
                        {value}
                      </span>
                    </div>
                  ))}
                </div>
              </InfoCard>

              <InfoCard
                title="Recommendation Health"
                cardBg={cardBg}
                borderColor={border}
                text={text}
                isDark={isDark}
              >
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "10px",
                    fontSize: "13px",
                  }}
                >
                  {[
                    ["NCF F1", formatPercentMetric(modelMetrics.ncf_f1)],
                    ["NCF Recall", formatPercentMetric(modelMetrics.ncf_recall)],
                    ["NCF Precision", formatPercentMetric(modelMetrics.ncf_precision)],
                    ["XGB F1", formatPercentMetric(modelMetrics.xgb_f1)],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      style={{
                        padding: "10px 12px",
                        borderRadius: "10px",
                        backgroundColor: isDark ? "rgba(255,255,255,0.04)" : "#f8fafc",
                        border: `1px solid ${border}`,
                      }}
                    >
                      <p
                        style={{ fontSize: "11px", color: subtext, margin: 0, marginBottom: "4px" }}
                      >
                        {label}
                      </p>
                      <p style={{ fontSize: "16px", fontWeight: 700, color: text, margin: 0 }}>
                        {metricsLoading ? "..." : value}
                      </p>
                    </div>
                  ))}
                </div>
              </InfoCard>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                gap: "16px",
                marginTop: "20px",
              }}
            >
              {[
                {
                  key: "comparison" as AdminMetricPlotKind,
                  title: "Model Quality Comparison",
                  subtitle: "AUC, F1, precision, recall from latest saved evaluation",
                  alt: "Model quality comparison graph",
                  emptyMsg: "No saved comparison metrics available yet.",
                },
                {
                  key: "loss" as AdminMetricPlotKind,
                  title: "Loss Snapshot",
                  subtitle: "Latest saved loss values for trained ranking stages",
                  alt: "Ranking loss graph",
                  emptyMsg: "No saved loss metrics available yet.",
                },
                {
                  key: "availability" as AdminMetricPlotKind,
                  title: "Live Recommender Stack",
                  subtitle: "Availability of models currently loaded by backend",
                  alt: "Live recommender stack graph",
                  emptyMsg: "No live engine availability metrics available yet.",
                },
              ].map((chart) => (
                <div
                  key={chart.key}
                  style={{
                    backgroundColor: cardBg,
                    border: `1px solid ${border}`,
                    borderRadius: "16px",
                    padding: "20px",
                    boxShadow: isDark ? "none" : "0 1px 4px rgba(0,0,0,0.06)",
                  }}
                >
                  <div style={{ marginBottom: "14px" }}>
                    <h2 style={{ fontSize: "14px", fontWeight: 700, color: text, margin: 0 }}>
                      {chart.title}
                    </h2>
                    <p style={{ fontSize: "12px", color: subtext, margin: 0, marginTop: "3px" }}>
                      {chart.subtitle}
                    </p>
                  </div>

                  {metricsLoading ? (
                    <ChartFallback
                      message={`Loading ${chart.title.toLowerCase()}...`}
                      borderColor={border}
                      isDark={isDark}
                      subtext={subtext}
                    />
                  ) : plotUrls[chart.key] ? (
                    <div
                      style={{
                        overflow: "hidden",
                        borderRadius: "10px",
                        border: `1px solid ${border}`,
                        backgroundColor: isDark ? "rgba(255,255,255,0.02)" : "#f9f9f9",
                        padding: "8px",
                      }}
                    >
                      <img
                        src={plotUrls[chart.key]}
                        alt={chart.alt}
                        style={{ width: "100%", height: "auto", borderRadius: "6px" }}
                      />
                    </div>
                  ) : chart.key === "comparison" && comparisonData.length ? (
                    <ChartFallback
                      message="Matplotlib comparison graph is unavailable."
                      borderColor={border}
                      isDark={isDark}
                      subtext={subtext}
                    />
                  ) : chart.key === "loss" && lossChartData.length ? (
                    <ChartFallback
                      message="Matplotlib loss graph is unavailable."
                      borderColor={border}
                      isDark={isDark}
                      subtext={subtext}
                    />
                  ) : chart.key === "availability" && hasLiveStackData ? (
                    <ChartFallback
                      message="Matplotlib live engine graph is unavailable."
                      borderColor={border}
                      isDark={isDark}
                      subtext={subtext}
                    />
                  ) : (
                    <ChartFallback
                      message={chart.emptyMsg}
                      borderColor={border}
                      isDark={isDark}
                      subtext={subtext}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: bg,
        color: text,
        transition: "background-color 0.3s, color 0.3s",
      }}
    >
      <AdminNavbar
        onLogout={handleLogout}
        darkMode={isDark}
        isSidebarOpen={isSidebarOpen}
        onToggleDark={toggleTheme}
        onToggleSidebar={() => setIsSidebarOpen((open) => !open)}
        userName={user?.name}
        userEmail={user?.email}
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: isSidebarOpen ? "240px 1fr" : "0px 1fr",
          transition: "grid-template-columns 0.3s ease",
          minHeight: "calc(100vh - 64px)",
          alignItems: "start",
        }}
      >
        <AdminSidebar
          activeView={activeView}
          onViewChange={setActiveView}
          isOpen={isSidebarOpen}
          isDark={isDark}
        />

        <main
          style={{
            minWidth: 0,
            width: "100%",
            padding: "32px",
            boxSizing: "border-box",
          }}
        >
          {renderView()}
        </main>
      </div>
    </div>
  );
}
