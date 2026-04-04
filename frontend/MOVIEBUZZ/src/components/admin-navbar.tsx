import { type ReactNode, useState } from "react";
import { Bell, LogOut, Menu, Moon, Settings, ShieldCheck, Sun } from "lucide-react";

interface AdminNavbarProps {
  onLogout?: () => void;
  onToggleSidebar?: () => void;
  darkMode: boolean;
  isSidebarOpen?: boolean;
  onToggleDark: () => void;
  userName?: string;
  userEmail?: string;
}

const NOTIFICATIONS = [
  "Analytics data sync completed.",
  "Movie catalog is ready for review.",
  "Admin workspace is using the latest UI version.",
];

export function AdminNavbar({
  onLogout,
  onToggleSidebar,
  darkMode,
  isSidebarOpen = true,
  onToggleDark,
  userName = "Admin",
  userEmail = "admin@moviebuzz.com",
}: AdminNavbarProps) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);

  const displayName = userName.trim() || "Admin";
  const displayEmail = userEmail.trim() || "admin@moviebuzz.com";
  const initials = displayName
    .split(" ")
    .map((word) => word[0] ?? "")
    .join("")
    .toUpperCase()
    .slice(0, 2);

  const bg = darkMode ? "#18181b" : "#ffffff";
  const border = darkMode ? "1px solid rgba(255,255,255,0.08)" : "1px solid #e4e4e7";
  const text = darkMode ? "#f4f4f5" : "#09090b";
  const subtext = "#71717a";
  const hoverBg = darkMode ? "rgba(255,255,255,0.06)" : "#f4f4f5";
  const divider = darkMode ? "rgba(255,255,255,0.08)" : "#e4e4e7";

  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        height: "64px",
        backgroundColor: bg,
        borderBottom: border,
        boxShadow: darkMode
          ? "0 1px 0 rgba(255,255,255,0.04)"
          : "0 1px 3px rgba(0,0,0,0.06)",
        transition: "background-color 0.3s, border-color 0.3s",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 24px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", userSelect: "none" }}>
          <img
            src="/favicon.svg"
            alt="MovieBuzz"
            width={36}
            height={36}
            style={{
              borderRadius: "10px",
              border: darkMode ? "1px solid rgba(255,255,255,0.1)" : "1px solid #e4e4e7",
              backgroundColor: darkMode ? "rgba(0,0,0,0.4)" : "#f9f9f9",
              objectFit: "cover",
            }}
          />
          <div style={{ display: "flex", alignItems: "baseline", gap: "1px" }}>
            <span
              style={{
                fontSize: "18px",
                fontWeight: 900,
                fontStyle: "italic",
                letterSpacing: "-0.02em",
                background: "linear-gradient(135deg, #dc2626, #f87171)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                lineHeight: 1,
              }}
            >
              MOVIE
            </span>
            <span
              style={{
                fontSize: "18px",
                fontWeight: 900,
                fontStyle: "italic",
                letterSpacing: "-0.02em",
                color: text,
                lineHeight: 1,
              }}
            >
              BUZZ
            </span>
          </div>
          <span
            style={{
              fontSize: "9px",
              fontWeight: 700,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "#dc2626",
              backgroundColor: darkMode ? "rgba(220,38,38,0.15)" : "#fef2f2",
              border: "1px solid rgba(220,38,38,0.3)",
              borderRadius: "6px",
              padding: "2px 7px",
            }}
          >
            Admin
          </span>
        </div>

        <button
          onClick={onToggleSidebar}
          aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
          style={{
            width: "34px",
            height: "34px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "10px",
            border: "none",
            backgroundColor: "transparent",
            color: subtext,
            cursor: "pointer",
            transition: "background-color 0.15s, color 0.15s",
          }}
          onMouseEnter={(event) => {
            event.currentTarget.style.backgroundColor = hoverBg;
            event.currentTarget.style.color = text;
          }}
          onMouseLeave={(event) => {
            event.currentTarget.style.backgroundColor = "transparent";
            event.currentTarget.style.color = subtext;
          }}
        >
          <Menu
            size={18}
            style={{
              transform: isSidebarOpen ? "rotate(0deg)" : "rotate(-90deg)",
              transition: "transform 0.3s",
            }}
          />
        </button>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
        <button
          onClick={onToggleDark}
          title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
          style={{
            width: "34px",
            height: "34px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "10px",
            border: "none",
            backgroundColor: "transparent",
            color: darkMode ? "#f59e0b" : "#7c3aed",
            cursor: "pointer",
            transition: "background-color 0.15s",
          }}
          onMouseEnter={(event) => {
            event.currentTarget.style.backgroundColor = hoverBg;
          }}
          onMouseLeave={(event) => {
            event.currentTarget.style.backgroundColor = "transparent";
          }}
        >
          {darkMode ? <Sun size={17} /> : <Moon size={17} />}
        </button>

        <div style={{ position: "relative" }}>
          <button
            onClick={() => {
              setNotifOpen((open) => !open);
              setDropdownOpen(false);
            }}
            aria-label="Notifications"
            style={{
              width: "34px",
              height: "34px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "10px",
              border: "none",
              backgroundColor: "transparent",
              color: subtext,
              cursor: "pointer",
              transition: "background-color 0.15s",
              position: "relative",
            }}
            onMouseEnter={(event) => {
              event.currentTarget.style.backgroundColor = hoverBg;
            }}
            onMouseLeave={(event) => {
              event.currentTarget.style.backgroundColor = "transparent";
            }}
          >
            <Bell size={17} />
            <span
              style={{
                position: "absolute",
                top: "7px",
                right: "7px",
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                backgroundColor: "#ef4444",
                border: `2px solid ${bg}`,
              }}
            />
          </button>

          {notifOpen ? (
            <>
              <div
                style={{ position: "fixed", inset: 0, zIndex: 40 }}
                onClick={() => setNotifOpen(false)}
              />
              <div
                style={{
                  position: "absolute",
                  top: "calc(100% + 8px)",
                  right: 0,
                  zIndex: 50,
                  width: "260px",
                  borderRadius: "16px",
                  border,
                  backgroundColor: bg,
                  boxShadow: darkMode
                    ? "0 20px 60px rgba(0,0,0,0.6)"
                    : "0 8px 40px rgba(0,0,0,0.12)",
                  overflow: "hidden",
                  padding: "10px",
                }}
              >
                <div
                  style={{
                    padding: "6px 8px 10px",
                    borderBottom: `1px solid ${divider}`,
                    marginBottom: "8px",
                  }}
                >
                  <p style={{ fontSize: "12px", fontWeight: 700, color: text, margin: 0 }}>
                    Notifications
                  </p>
                  <p style={{ fontSize: "11px", color: subtext, margin: "4px 0 0" }}>
                    Latest admin updates for MovieBuzz.
                  </p>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {NOTIFICATIONS.map((message) => (
                    <div
                      key={message}
                      style={{
                        padding: "10px 12px",
                        borderRadius: "12px",
                        backgroundColor: darkMode ? "rgba(255,255,255,0.04)" : "#fafafa",
                        border,
                      }}
                    >
                      <p style={{ fontSize: "12px", color: text, margin: 0 }}>{message}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : null}
        </div>

        <div
          style={{
            width: "1px",
            height: "24px",
            backgroundColor: divider,
            margin: "0 4px",
          }}
        />

        <div style={{ position: "relative" }}>
          <button
            onClick={() => {
              setDropdownOpen((open) => !open);
              setNotifOpen(false);
            }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "5px 10px 5px 5px",
              borderRadius: "12px",
              border,
              backgroundColor: darkMode ? "rgba(255,255,255,0.04)" : "#fafafa",
              cursor: "pointer",
              transition: "background-color 0.15s",
            }}
            onMouseEnter={(event) => {
              event.currentTarget.style.backgroundColor = hoverBg;
            }}
            onMouseLeave={(event) => {
              event.currentTarget.style.backgroundColor = darkMode
                ? "rgba(255,255,255,0.04)"
                : "#fafafa";
            }}
          >
            <div
              style={{
                width: "32px",
                height: "32px",
                borderRadius: "9px",
                background: "linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #f59e0b 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#ffffff",
                fontSize: "12px",
                fontWeight: 800,
                letterSpacing: "0.02em",
                flexShrink: 0,
                boxShadow: "0 2px 8px rgba(220,38,38,0.35)",
              }}
            >
              {initials || <ShieldCheck size={16} />}
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                gap: "1px",
              }}
            >
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: 700,
                  color: text,
                  maxWidth: "100px",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {displayName}
              </span>
              <span
                style={{
                  fontSize: "10px",
                  fontWeight: 500,
                  color: "#dc2626",
                  letterSpacing: "0.04em",
                }}
              >
                Administrator
              </span>
            </div>
          </button>

          {dropdownOpen ? (
            <>
              <div
                style={{ position: "fixed", inset: 0, zIndex: 40 }}
                onClick={() => setDropdownOpen(false)}
              />
              <div
                style={{
                  position: "absolute",
                  top: "calc(100% + 8px)",
                  right: 0,
                  zIndex: 50,
                  width: "220px",
                  borderRadius: "16px",
                  border,
                  backgroundColor: bg,
                  boxShadow: darkMode
                    ? "0 20px 60px rgba(0,0,0,0.6)"
                    : "0 8px 40px rgba(0,0,0,0.12)",
                  overflow: "hidden",
                  padding: "8px",
                }}
              >
                <div
                  style={{
                    padding: "10px 12px 12px",
                    borderBottom: `1px solid ${divider}`,
                    marginBottom: "6px",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div
                      style={{
                        width: "36px",
                        height: "36px",
                        borderRadius: "10px",
                        background:
                          "linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #f59e0b 100%)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "#ffffff",
                        fontSize: "13px",
                        fontWeight: 800,
                        flexShrink: 0,
                        boxShadow: "0 2px 8px rgba(220,38,38,0.35)",
                      }}
                    >
                      {initials || <ShieldCheck size={16} />}
                    </div>
                    <div>
                      <p style={{ fontSize: "13px", fontWeight: 700, color: text, margin: 0 }}>
                        {displayName}
                      </p>
                      <p
                        style={{
                          fontSize: "11px",
                          color: subtext,
                          margin: 0,
                          marginTop: "1px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          maxWidth: "140px",
                        }}
                      >
                        {displayEmail}
                      </p>
                    </div>
                  </div>
                </div>

                <DropdownBtn
                  icon={darkMode ? <Sun size={14} /> : <Moon size={14} />}
                  label={darkMode ? "Light Mode" : "Dark Mode"}
                  iconColor={darkMode ? "#f59e0b" : "#7c3aed"}
                  textColor={darkMode ? "#f59e0b" : "#7c3aed"}
                  hoverBg={hoverBg}
                  onClick={() => {
                    onToggleDark();
                    setDropdownOpen(false);
                  }}
                />

                <DropdownBtn
                  icon={<Settings size={14} />}
                  label="Settings"
                  iconColor={subtext}
                  textColor={text}
                  hoverBg={hoverBg}
                  onClick={() => setDropdownOpen(false)}
                />

                <div style={{ height: "1px", backgroundColor: divider, margin: "6px 0" }} />

                <DropdownBtn
                  icon={<LogOut size={14} />}
                  label="Log out"
                  iconColor="#ef4444"
                  textColor="#ef4444"
                  hoverBg={darkMode ? "rgba(239,68,68,0.1)" : "#fef2f2"}
                  onClick={() => {
                    setDropdownOpen(false);
                    onLogout?.();
                  }}
                />
              </div>
            </>
          ) : null}
        </div>
      </div>
    </nav>
  );
}

function DropdownBtn({
  icon,
  label,
  iconColor,
  textColor,
  hoverBg,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  iconColor: string;
  textColor: string;
  hoverBg: string;
  onClick: () => void;
}) {
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
        borderRadius: "10px",
        border: "none",
        backgroundColor: hovered ? hoverBg : "transparent",
        cursor: "pointer",
        transition: "background-color 0.15s",
        color: textColor,
      }}
    >
      <span style={{ color: iconColor, display: "flex", alignItems: "center" }}>{icon}</span>
      <span style={{ fontSize: "12px", fontWeight: 600 }}>{label}</span>
    </button>
  );
}
