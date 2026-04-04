import { cn } from "../lib/utils";

interface MovieBuzzLogoProps {
  className?: string;
  imageClassName?: string;
  showWordmark?: boolean;
  size?: number;
  subtitle?: string;
  theme?: "light" | "dark";
  wordmarkClassName?: string;
}

export function MovieBuzzLogo({
  className,
  imageClassName,
  showWordmark = true,
  size = 44,
  subtitle,
  theme = "light",
  wordmarkClassName,
}: MovieBuzzLogoProps) {
  const titleClassName =
    theme === "dark" ? "text-white" : "text-zinc-950";
  const subtitleClassName =
    theme === "dark" ? "text-zinc-400" : "text-zinc-500";

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <img
        src="/favicon.svg"
        alt="MovieBuzz logo"
        width={size}
        height={size}
        className={cn(
          "rounded-2xl border border-white/10 bg-black/60 object-cover shadow-lg",
          imageClassName,
        )}
      />
      {showWordmark ? (
        <div className="min-w-0">
          <div
            className={cn(
              "truncate text-base font-black uppercase tracking-[0.24em]",
              titleClassName,
              wordmarkClassName,
            )}
          >
            MovieBuzz
          </div>
          {subtitle ? (
            <div className={cn("text-xs", subtitleClassName)}>{subtitle}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
