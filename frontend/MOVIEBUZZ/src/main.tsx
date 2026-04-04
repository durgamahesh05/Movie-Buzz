
  import { createRoot } from "react-dom/client";
import { ErrorBoundary } from "react-error-boundary";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
  import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const GlobalErrorFallback = ({ error }: { error: Error }) => (
  <div className="flex h-screen w-screen flex-col items-center justify-center p-4 text-center">
    <h1 className="text-2xl font-bold text-red-600">Something went wrong</h1>
    <pre className="mt-4 max-w-lg overflow-auto rounded bg-gray-100 p-4 text-left text-sm text-red-500">
      {error.message}
    </pre>
  </div>
);

createRoot(document.getElementById("root")!).render(
  <ErrorBoundary FallbackComponent={GlobalErrorFallback}>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </ErrorBoundary>
);
  
