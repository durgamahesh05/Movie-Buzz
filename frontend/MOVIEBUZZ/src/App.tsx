import { Suspense, lazy } from "react";
import { createBrowserRouter, RouterProvider, Navigate } from "react-router-dom";
import AuthLayout from "./layouts/AuthLayout";
import DashboardLayout from "./layouts/DashboardLayout";
import AdminLayout from "./layouts/AdminLayout";
import { Loader2 } from "lucide-react";

const LoginPage = lazy(() => import("./components/login-page"));
const RegisterPage = lazy(() => import("./components/register-page").then(m => ({ default: m.RegisterPage })));
const ForgotPasswordPage = lazy(() => import("./components/forgot-password").then(m => ({ default: m.ForgotPasswordPage })));
const PreferencesSetupPage = lazy(() => import("./components/preferences-setup-page").then(m => ({ default: m.PreferencesSetupPage })));
const Dashboard = lazy(() => import("./components/Dashboard"));
const AdminDashboard = lazy(() => import("./components/admin-dashboard").then(m => ({ default: m.AdminDashboard })));

const LoadingSpinner = () => (
  <div className="flex h-screen w-screen items-center justify-center bg-background">
    <Loader2 className="h-8 w-8 animate-spin text-primary" />
  </div>
);

const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate to="/login" replace />,
  },
  {
    element: <AuthLayout />,
    children: [
      { path: "login", element: <Suspense fallback={<LoadingSpinner />}><LoginPage /></Suspense> },
      { path: "register", element: <Suspense fallback={<LoadingSpinner />}><RegisterPage /></Suspense> },
      { path: "forgot-password", element: <Suspense fallback={<LoadingSpinner />}><ForgotPasswordPage /></Suspense> },
    ],
  },
  {
    element: <DashboardLayout />,
    children: [
      { path: "dashboard", element: <Suspense fallback={<LoadingSpinner />}><Dashboard /></Suspense> },
      { path: "preferences-setup", element: <Suspense fallback={<LoadingSpinner />}><PreferencesSetupPage /></Suspense> },
    ],
  },
  {
    element: <AdminLayout />,
    children: [
      { path: "admin", element: <Suspense fallback={<LoadingSpinner />}><AdminDashboard /></Suspense> },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
