import { Navigate, Outlet } from "react-router-dom";
import { useAppStore } from "../store/appStore";

export default function DashboardLayout() {
  const user = useAppStore((state) => state.user);

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}