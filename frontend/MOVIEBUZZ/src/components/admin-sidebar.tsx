import {
  BarChart3,
  ChevronRight,
  Clapperboard,
  LayoutDashboard,
  Users,
} from "lucide-react";

type AdminView = "dashboard" | "users" | "movies" | "analytics";

interface AdminSidebarProps {
  activeView: string;
  onViewChange: (view: AdminView) => void;
  isOpen?: boolean;
  onClose?: () => void;
  isDark?: boolean;
}

const menuItems: Array<{
  id: AdminView;
  label: string;
  icon: typeof LayoutDashboard;
  description: string;
}> = [
  {
    id: "dashboard",
    label: "Dashboard",
    icon: LayoutDashboard,
    description: "Overview & stats",
  },
  {
    id: "users",
    label: "Manage Users",
    icon: Users,
    description: "User management",
  },
  {
    id: "movies",
    label: "Manage Movies",
    icon: Clapperboard,
    description: "Movie catalog",
  },
  {
    id: "analytics",
    label: "Analytics",
    icon: BarChart3,
    description: "Model metrics",
  },
];

export function AdminSidebar({
  activeView,
  onViewChange,
  isOpen = true,
  isDark,
}: AdminSidebarProps) {
  const bg = isDark ? "#18181b" : "#ffffff";
  const border = isDark ? "rgba(255,255,255,0.07)" : "#e4e4e7";
  const text = isDark ? "#f4f4f5" : "#09090b";
  const subtext = isDark ? "#71717a" : "#a1a1aa";
  const hoverBg = isDark ? "rgba(255,255,255,0.05)" : "#f4f4f5";
  const activeBg = isDark ? "rgba(220,38,38,0.12)" : "#fef2f2";
  const activeText = "#dc2626";
  const activeBorder = "rgba(220,38,38,0.3)";

  return (
    <aside
      style={{
        position: "sticky",
        top: "64px",
        alignSelf: "start",
        height: "calc(100vh - 64px)",
        overflowX: "hidden",
        overflowY: "auto",
        width: isOpen ? "240px" : "0px",
        opacity: isOpen ? 1 : 0,
        pointerEvents: isOpen ? "auto" : "none",
        transition: "width 0.3s ease, opacity 0.25s ease",
        backgroundColor: bg,
        borderRight: isOpen ? `1px solid ${border}` : "none",
        boxSizing: "border-box",
      }}
      aria-hidden={!isOpen}
    >
      <div
        style={{
          width: "240px",
          padding: "20px 12px",
          transform: isOpen ? "translateX(0)" : "translateX(-100%)",
          opacity: isOpen ? 1 : 0,
          transition: "transform 0.3s ease, opacity 0.2s ease",
        }}
      >
        <p
          style={{
            fontSize: "10px",
            fontWeight: 700,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            color: subtext,
            padding: "0 8px",
            marginBottom: "10px",
          }}
        >
          Navigation
        </p>

        <nav style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;

            return (
              <button
                key={item.id}
                onClick={() => onViewChange(item.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                  width: "100%",
                  padding: "10px 12px",
                  borderRadius: "12px",
                  border: isActive ? `1px solid ${activeBorder}` : "1px solid transparent",
                  backgroundColor: isActive ? activeBg : "transparent",
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "background-color 0.15s, border-color 0.15s",
                  color: isActive ? activeText : text,
                  position: "relative",
                }}
                onMouseEnter={(event) => {
                  if (!isActive) {
                    event.currentTarget.style.backgroundColor = hoverBg;
                  }
                }}
                onMouseLeave={(event) => {
                  if (!isActive) {
                    event.currentTarget.style.backgroundColor = "transparent";
                  }
                }}
              >
                <div
                  style={{
                    width: "34px",
                    height: "34px",
                    borderRadius: "10px",
                    backgroundColor: isActive
                      ? "rgba(220,38,38,0.15)"
                      : isDark
                        ? "rgba(255,255,255,0.06)"
                        : "#f4f4f5",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    transition: "background-color 0.15s",
                  }}
                >
                  <Icon
                    size={17}
                    color={isActive ? activeText : subtext}
                    strokeWidth={isActive ? 2.2 : 1.8}
                  />
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <p
                    style={{
                      fontSize: "13px",
                      fontWeight: isActive ? 700 : 500,
                      color: isActive ? activeText : text,
                      margin: 0,
                      lineHeight: 1.2,
                    }}
                  >
                    {item.label}
                  </p>
                  <p
                    style={{
                      fontSize: "11px",
                      color: subtext,
                      margin: 0,
                      marginTop: "1px",
                    }}
                  >
                    {item.description}
                  </p>
                </div>

                {isActive ? (
                  <ChevronRight size={14} color={activeText} style={{ flexShrink: 0 }} />
                ) : null}
              </button>
            );
          })}
        </nav>

        <div
          style={{
            marginTop: "32px",
            padding: "14px",
            borderRadius: "14px",
            background: isDark
              ? "linear-gradient(135deg, rgba(220,38,38,0.12), rgba(245,158,11,0.08))"
              : "linear-gradient(135deg, #fef2f2, #fff7ed)",
            border: "1px solid rgba(220,38,38,0.2)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
            <img
              src="/favicon.svg"
              alt="MovieBuzz"
              width={22}
              height={22}
              style={{ borderRadius: "6px" }}
            />
            <span
              style={{
                fontSize: "12px",
                fontWeight: 800,
                letterSpacing: "0.05em",
                color: "#dc2626",
              }}
            >
              MOVIEBUZZ
            </span>
          </div>
          <p style={{ fontSize: "10px", color: subtext, margin: 0, lineHeight: 1.5 }}>
            Admin Control Panel · v2.0
          </p>
        </div>
      </div>
    </aside>
  );
}
