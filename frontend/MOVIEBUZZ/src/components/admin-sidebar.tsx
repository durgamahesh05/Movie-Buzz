import { LayoutDashboard, Users, Clapperboard, BarChart3 } from "lucide-react";
import { cn } from "../lib/utils";

// ── Google Fonts ──────────────────────────────────────────────────────────────
// Inject the Inter font from Google Fonts once at the sidebar level.
// (You can move this to index.html <head> if you prefer, but this works too.)
if (typeof document !== "undefined" && !document.getElementById("gf-inter")) {
  const link = document.createElement("link");
  link.id = "gf-inter";
  link.rel = "stylesheet";
  link.href =
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap";
  document.head.appendChild(link);
}

interface AdminSidebarProps {
  activeView: string;
  onViewChange: (view: "dashboard" | "users" | "movies" | "analytics") => void;
  isOpen?: boolean;
  onClose?: () => void;
  isDark?: boolean;
}

const menuItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "users",     label: "Manage Users",  icon: Users },
  { id: "movies",    label: "Manage Movies", icon: Clapperboard },
  { id: "analytics", label: "Analytics",     icon: BarChart3 },

];

export function AdminSidebar({ activeView, onViewChange, isOpen = true, isDark }: AdminSidebarProps) {
  return (
    <aside
      className={cn(
        "sticky top-16 z-40 h-[calc(100vh-4rem)] flex-shrink-0 overflow-hidden transition-[width,opacity,background-color] duration-300 ease-in-out",
        isDark ? "bg-zinc-900" : "bg-white",
        isOpen
          ? isDark
            ? "w-64 border-r border-zinc-800/70"
            : "w-64 border-r border-zinc-200/70"
          : "w-0 border-r-0 opacity-0 pointer-events-none"
      )}
      aria-hidden={!isOpen}
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      <div
        className={cn(
          "w-64 p-6 transition-all duration-300 ease-in-out",
          isOpen ? "translate-x-0 opacity-100" : "-translate-x-full opacity-0 pointer-events-none",
        )}
      >
        {/* Nav Items */}
        <nav className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;

            return (
              <button
                key={item.id}
                onClick={() => onViewChange(item.id as any)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left transition-all duration-200",
                  isActive
                    ? isDark
                      ? "bg-red-500/15 text-red-400 shadow-sm"
                      : "bg-red-50 text-red-600 shadow-sm"
                    : isDark
                      ? "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
                      : "text-zinc-700 hover:bg-zinc-100 hover:text-zinc-900"
                )}
              >
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
