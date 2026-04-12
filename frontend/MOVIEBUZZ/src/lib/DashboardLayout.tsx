import { Navigate, Outlet } from "react-router-dom";
import { useAppStore } from "../store/appStore";

export default function DashboardLayout() {
  const user = useAppStore((state) => state.user);

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  // Prevent admins from accessing user dashboard
  if (user.role === "admin") {
    return <Navigate to="/admin" replace />;
  }

  return <Outlet />;
}