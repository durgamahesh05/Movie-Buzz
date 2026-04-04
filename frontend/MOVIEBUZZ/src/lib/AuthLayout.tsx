import { Navigate, Outlet } from "react-router-dom";
import { useAppStore } from "../store/appStore";

export default function AuthLayout() {
  const user = useAppStore((state) => state.user);

  if (user) {
    return <Navigate to={user.role === "admin" ? "/admin" : "/dashboard"} replace />;
  }

  return <Outlet />;
}