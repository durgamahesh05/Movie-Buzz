import { useEffect, useState } from "react";
import { Film, Loader2, UserCheck, Users, Heart } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ApiError, getAdminOverview, type AdminOverview } from "../lib/api";

type StatConfig = {
  title: string;
  value: string;
  subtitle: string;
  icon: typeof Users;
  color: string;
  bgColor: string;
};

const EMPTY_OVERVIEW: AdminOverview = {
  total_users: 0,
  verified_users: 0,
  catalog_movies: 0,
  wishlist_items: 0,
};

interface StatsCardsProps {
  refreshToken?: number;
}

export function StatsCards({ refreshToken = 0 }: StatsCardsProps) {
  const [overview, setOverview] = useState<AdminOverview>(EMPTY_OVERVIEW);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    const loadOverview = async () => {
      setLoading(true);
      try {
        const data = await getAdminOverview();
        if (!cancelled) {
          setOverview(data);
          setError("");
        }
      } catch (loadError) {
        if (!cancelled) {
          setOverview(EMPTY_OVERVIEW);
          setError(
            loadError instanceof ApiError
              ? loadError.message
              : "Unable to load admin overview",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadOverview();

    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  const stats: StatConfig[] = [
    {
      title: "Total Users",
      value: overview.total_users.toLocaleString(),
      subtitle: "Registered user accounts",
      icon: Users,
      color: "text-blue-600 dark:text-blue-400",
      bgColor: "bg-blue-50 dark:bg-blue-500/15",
    },
    {
      title: "Verified Users",
      value: overview.verified_users.toLocaleString(),
      subtitle: "Users who completed OTP verification",
      icon: UserCheck,
      color: "text-green-600 dark:text-green-400",
      bgColor: "bg-green-50 dark:bg-green-500/15",
    },
    {
      title: "Catalog Movies",
      value: overview.catalog_movies.toLocaleString(),
      subtitle: "Movie rows stored in the movie catalog DB",
      icon: Film,
      color: "text-violet-600 dark:text-violet-400",
      bgColor: "bg-violet-50 dark:bg-violet-500/15",
    },
    {
      title: "Wishlist Saves",
      value: overview.wishlist_items.toLocaleString(),
      subtitle: "Saved movie entries across users",
      icon: Heart,
      color: "text-rose-600 dark:text-rose-400",
      bgColor: "bg-rose-50 dark:bg-rose-500/15",
    },
  ];

  return (
    <div className="space-y-4">
      {error && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-950/60 dark:bg-rose-950/30 dark:text-rose-200">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card
              key={stat.title}
              className="border-zinc-200/60 bg-white shadow-sm dark:border-zinc-800/60 dark:bg-zinc-900"
            >
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm text-zinc-600 dark:text-zinc-400">
                  {stat.title}
                </CardTitle>
                <div className={`rounded-lg p-2 ${stat.bgColor}`}>
                  <Icon className={`h-4 w-4 ${stat.color}`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2 text-2xl font-semibold text-zinc-950 dark:text-zinc-50">
                  {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : null}
                  <span>{stat.value}</span>
                </div>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  {stat.subtitle}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
