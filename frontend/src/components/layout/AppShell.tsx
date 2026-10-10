import { Monitor, Moon, Settings as SettingsIcon, Sun } from "lucide-react";
import { useState } from "react";
import { Link, NavLink, Outlet } from "react-router";
import { Toaster } from "sonner";

import { useSettings } from "@/api/hooks";
import { cn } from "@/lib/cn";
import { saveTheme, storedTheme, type Theme } from "@/lib/theme";

import { Badge } from "../ui/badge";
import { Tooltip } from "../ui/tooltip";

const NEXT_THEME: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
const THEME_ICON = { system: Monitor, light: Sun, dark: Moon };

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(storedTheme());
  const Icon = THEME_ICON[theme];
  return (
    <Tooltip content={`Theme: ${theme} (click to change)`}>
      <button
        type="button"
        className="rounded-md p-2 text-ink-soft hover:bg-muted hover:text-ink"
        aria-label={`Theme: ${theme}`}
        onClick={() => {
          const next = NEXT_THEME[theme];
          saveTheme(next);
          setTheme(next);
        }}
      >
        <Icon className="size-4" />
      </button>
    </Tooltip>
  );
}

const navLink = ({ isActive }: { isActive: boolean }) =>
  cn(
    "rounded-md px-2.5 py-1.5 text-sm font-medium whitespace-nowrap sm:px-3",
    isActive ? "bg-muted text-ink" : "text-ink-soft hover:bg-muted hover:text-ink",
  );

export function AppShell() {
  const settings = useSettings();
  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50">
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-line bg-surface/90 backdrop-blur print:hidden">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-1 px-3 sm:gap-2 sm:px-4">
          <Link to="/" className="mr-1 flex shrink-0 items-center gap-2 font-semibold sm:mr-3" aria-label="Research Tool">
            <img src="/favicon.svg" alt="" className="size-6" />
            <span className="hidden sm:inline">Research Tool</span>
          </Link>
          <nav className="flex items-center gap-1" aria-label="Main">
            <NavLink to="/" end className={navLink}>
              Projects
            </NavLink>
            <NavLink to="/new" className={navLink}>
              New research
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-1">
            {settings.data?.runs_locally && (
              <Tooltip content="Papers, the AI model and the cluster are simulated on this computer.">
                <span className="hidden sm:inline">
                  <Badge tone="info">Demo mode</Badge>
                </span>
              </Tooltip>
            )}
            <ThemeToggle />
            <Tooltip content="Settings">
              <Link to="/settings" className="rounded-md p-2 text-ink-soft hover:bg-muted hover:text-ink" aria-label="Settings">
                <SettingsIcon className="size-4" />
              </Link>
            </Tooltip>
          </div>
        </div>
      </header>
      <main id="main">
        <Outlet />
      </main>
      <Toaster position="bottom-right" richColors closeButton theme="system" />
    </div>
  );
}
