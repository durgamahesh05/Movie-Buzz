import { Bell, ShieldCheck, LogOut, Menu, Sun, Moon } from "lucide-react";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";

interface AdminNavbarProps {
  onLogout?: () => void;
  onToggleSidebar?: () => void;
  darkMode: boolean;
  isSidebarOpen?: boolean;
  onToggleDark: () => void;
  userName?: string;
  userEmail?: string;
}

export function AdminNavbar({
  onLogout,
  onToggleSidebar,
  darkMode,
  isSidebarOpen = true,
  onToggleDark,
  userName = "Admin",
  userEmail = "admin@moviebuzz.com",
}: AdminNavbarProps) {
  const displayName = userName.trim() || "Admin";
  const displayEmail = userEmail.trim() || "admin@moviebuzz.com";

  return (
    <nav className="sticky top-0 z-50 h-16 border-b border-zinc-200 bg-white shadow-sm transition-colors dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center justify-between h-full px-6">

        {/* LEFT: Logo + Menu toggle */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-0.5 select-none">
            <span className="text-xl font-black italic tracking-tight bg-gradient-to-r from-red-600 to-red-400 bg-clip-text text-transparent leading-none">
              MOVIE
            </span>
            <span className="text-xl font-black italic tracking-tight text-zinc-900 dark:text-white leading-none">
              BUZZ
            </span>
          </div>
          <button
            onClick={onToggleSidebar}
            aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
            aria-pressed={isSidebarOpen}
            className="rounded-lg p-1.5 text-zinc-500 transition-all duration-300 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-white"
          >
            <Menu
              className={`w-5 h-5 transition-transform duration-300 ${isSidebarOpen ? "rotate-0" : "-rotate-90"}`}
            />
          </button>
        </div>

        {/* RIGHT */}
        <div className="flex items-center gap-2">

          {/* Bell notification */}
          <Button
            variant="ghost"
            size="icon"
            className="relative text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-800"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-red-500 rounded-full" />
          </Button>

          {/* Avatar + dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-2 rounded-xl border border-zinc-200 bg-white py-1 pl-1 pr-3 outline-none transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:bg-zinc-800">
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-red-500 via-red-600 to-amber-500 text-white shadow-sm ring-2 ring-white dark:ring-zinc-900">
                  <ShieldCheck className="h-4 w-4" strokeWidth={2.2} />
                </span>
                <span className="hidden max-w-28 truncate text-xs font-semibold text-zinc-700 dark:text-zinc-300 sm:block">
                  {displayName}
                </span>
              </button>
            </DropdownMenuTrigger>

            <DropdownMenuContent
              align="end"
              className="w-52 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl shadow-xl p-1.5"
            >
              {/* User info header */}
              <DropdownMenuLabel className="px-2 py-2">
                <div className="flex flex-col gap-0.5">
                  <p className="text-sm font-bold text-zinc-900 dark:text-white">
                    {displayName}
                  </p>
                  <p className="text-xs text-zinc-400 dark:text-zinc-500 font-normal">
                    {displayEmail}
                  </p>
                </div>
              </DropdownMenuLabel>

              <DropdownMenuSeparator className="bg-zinc-100 dark:bg-zinc-800 my-1" />

              {/* Profile */}
              <DropdownMenuItem className="rounded-lg px-3 py-2 text-xs font-medium text-zinc-700 dark:text-zinc-300 cursor-pointer focus:bg-zinc-50 dark:focus:bg-zinc-800 focus:text-zinc-900 dark:focus:text-white">
                <ShieldCheck className="mr-2 w-3.5 h-3.5 text-red-500" />
                Profile
              </DropdownMenuItem>

              <DropdownMenuItem
                onClick={onToggleDark}
                className="rounded-lg px-3 py-2 text-xs font-medium cursor-pointer focus:bg-zinc-50 dark:focus:bg-zinc-800 text-zinc-700 dark:text-zinc-300 focus:text-zinc-900 dark:focus:text-white"
              >
                {darkMode ? (
                  <>
                    <Sun className="mr-2 w-3.5 h-3.5 text-amber-500" />
                    <span className="text-amber-500">Light Mode</span>
                  </>
                ) : (
                  <>
                    <Moon className="mr-2 w-3.5 h-3.5 text-indigo-500" />
                    <span className="text-indigo-500">Dark Mode</span>
                  </>
                )}
              </DropdownMenuItem>

              <DropdownMenuSeparator className="bg-zinc-100 dark:bg-zinc-800 my-1" />

              {/* Logout */}
              <DropdownMenuItem
                onClick={onLogout}
                className="rounded-lg px-3 py-2 text-xs font-medium text-red-500 cursor-pointer focus:bg-red-50 dark:focus:bg-red-900/20 focus:text-red-600"
              >
                <LogOut className="mr-2 w-3.5 h-3.5" />
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </nav>
  );
}
