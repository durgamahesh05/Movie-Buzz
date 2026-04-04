import { Navigate, Outlet } from "react-router-dom";
import { useAppStore } from "../store/appStore";

export default function AdminLayout() {
  const user = useAppStore((state) => state.user);

  if (!user || user.role !== "admin") {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}