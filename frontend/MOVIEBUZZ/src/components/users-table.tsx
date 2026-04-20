import { useEffect, useMemo, useState } from "react";
import { Loader2, Trash2 } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  ApiError,
  deleteAdminUser,
  getAdminUsers,
  type AdminUser,
  updateAdminUserRole,
} from "../lib/api";

interface UsersTableProps {
  limit?: number;
  refreshToken?: number;
  onUsersChange?: () => void;
}

function maskDashboardEmail(email: string, role?: string) {
  if (role !== "admin") {
    return email;
  }

  if (!email.includes("@")) {
    return `${email.slice(0, 2)}${"*".repeat(Math.max(email.length - 2, 4))}`;
  }

  const [localPart = "", domainPart = ""] = email.split("@");
  const maskedLocal =
    localPart.length <= 2
      ? `${localPart.slice(0, 1)}***`
      : `${localPart.slice(0, 2)}${"*".repeat(Math.max(localPart.length - 2, 3))}`;
  const [domainName = "", ...domainSuffix] = domainPart.split(".");
  const maskedDomain = domainName
    ? `${domainName.slice(0, 1)}${"*".repeat(Math.max(domainName.length - 1, 2))}`
    : "***";
  const suffix = domainSuffix.length ? `.${domainSuffix.join(".")}` : "";

  return `${maskedLocal}@${maskedDomain}${suffix}`;
}

function formatDate(value?: string) {
  if (!value) {
    return "N/A";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleDateString();
}

export function UsersTable({
  limit,
  refreshToken = 0,
  onUsersChange,
}: UsersTableProps) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deletingEmail, setDeletingEmail] = useState("");
  const [updatingRoleEmail, setUpdatingRoleEmail] = useState("");

  useEffect(() => {
    let cancelled = false;

    const loadUsers = async () => {
      setLoading(true);
      try {
        const data = await getAdminUsers();
        if (!cancelled) {
          setUsers(data);
          setError("");
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof ApiError
              ? loadError.message
              : "Unable to load users right now",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    loadUsers();

    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  const displayUsers = useMemo(
    () => (limit ? users.slice(0, limit) : users),
    [limit, users],
  );

  const handleDelete = async (email: string) => {
    if (!window.confirm(`Delete ${email} from MovieBuzz?`)) {
      return;
    }

    setDeletingEmail(email);
    try {
      await deleteAdminUser(email);
      setUsers((current) => current.filter((user) => user.email !== email));
      onUsersChange?.();
    } catch (deleteError) {
      alert(
        deleteError instanceof ApiError
          ? deleteError.message
          : "Unable to delete user right now",
      );
    } finally {
      setDeletingEmail("");
    }
  };

  const handleRoleChange = async (
    email: string,
    role: "user" | "mod" | "admin",
  ) => {
    setUpdatingRoleEmail(email);
    try {
      await updateAdminUserRole(email, role);
      setUsers((current) =>
        current.map((user) => (user.email === email ? { ...user, role } : user)),
      );
      onUsersChange?.();
    } catch (updateError) {
      alert(
        updateError instanceof ApiError
          ? updateError.message
          : "Unable to update the user role right now",
      );
    } finally {
      setUpdatingRoleEmail("");
    }
  };

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-200/60 bg-white text-zinc-900 shadow-sm dark:border-zinc-800/60 dark:bg-zinc-900 dark:text-zinc-100">
      {!limit && (
        <div className="border-b border-zinc-200/60 bg-zinc-50/80 p-4 dark:border-zinc-800/60 dark:bg-zinc-900/60">
          <h3 className="text-sm text-zinc-500 dark:text-zinc-400">User Accounts</h3>
          <p className="mt-1 text-2xl font-semibold text-zinc-950 dark:text-white">
            {loading ? "..." : users.length}
          </p>
        </div>
      )}

      {error && (
        <div className="border-b border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-950/60 dark:bg-rose-950/30 dark:text-rose-200">
          {error}
        </div>
      )}

      <Table>
        <TableHeader>
          <TableRow className="border-zinc-200/60 bg-zinc-50/80 hover:bg-zinc-50 dark:border-zinc-800/60 dark:bg-zinc-900/60 dark:hover:bg-zinc-900/60">
            <TableHead className="text-zinc-600 dark:text-zinc-400">Name</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Email</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Role</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Status</TableHead>
            <TableHead className="text-zinc-600 dark:text-zinc-400">Joined</TableHead>
            <TableHead className="text-right text-zinc-600 dark:text-zinc-400">
              Actions
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell
                colSpan={6}
                className="py-8 text-center text-zinc-500 dark:text-zinc-400"
              >
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading real users...
                </span>
              </TableCell>
            </TableRow>
          ) : displayUsers.length ? (
            displayUsers.map((user) => {
              const isVerified = Boolean(user.verified);
              const isDeleting = deletingEmail === user.email;
              const isUpdatingRole = updatingRoleEmail === user.email;
              const allowRoleManagement = !limit;

              return (
                <TableRow
                  key={user.email}
                  className="border-zinc-200/60 hover:bg-zinc-50 dark:border-zinc-800/60 dark:hover:bg-zinc-800/50"
                >
                  <TableCell className="font-medium text-zinc-900 dark:text-zinc-300">
                    {user.name}
                  </TableCell>
                  <TableCell className="text-zinc-700 dark:text-zinc-300">
                    {maskDashboardEmail(user.email, user.role)}
                  </TableCell>
                  <TableCell>
                    {allowRoleManagement ? (
                      <select
                        value={user.role || "user"}
                        disabled={isUpdatingRole || isDeleting}
                        onChange={(event) =>
                          void handleRoleChange(
                            user.email,
                            event.target.value as "user" | "mod" | "admin",
                          )
                        }
                        className="flex h-9 min-w-[112px] rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm capitalize text-zinc-900 focus:outline-none focus:ring-2 focus:ring-zinc-900 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white dark:focus:ring-zinc-100"
                      >
                        <option value="user">User</option>
                        <option value="mod">Mod</option>
                        <option value="admin">Admin</option>
                      </select>
                    ) : (
                      <Badge
                        variant={user.role === "admin" ? "default" : "outline"}
                        className="capitalize"
                      >
                        {user.role || "user"}
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    <Badge variant={isVerified ? "default" : "secondary"}>
                      {isVerified ? "Verified" : "Pending"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-zinc-700 dark:text-zinc-300">
                    {formatDate(user.created_at)}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      disabled={isDeleting || isUpdatingRole}
                      className="text-red-500 hover:bg-red-50 hover:text-red-600 disabled:opacity-50 dark:hover:bg-red-950/30"
                      onClick={() => handleDelete(user.email)}
                    >
                      {isDeleting || isUpdatingRole ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })
          ) : (
            <TableRow>
              <TableCell
                colSpan={6}
                className="py-8 text-center text-zinc-500 dark:text-zinc-400"
              >
                No users found yet.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
