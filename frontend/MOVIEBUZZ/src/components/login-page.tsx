import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./ui/card";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { Eye, EyeOff } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, loginUser } from "../lib/api";
import { useAppStore } from "../store/appStore";
import { MovieBuzzLogo } from "./moviebuzz-logo";

export default function LoginPage() {
  const navigate = useNavigate();
  const setUser = useAppStore((state) => state.setUser);
  const [loginIdentifier, setLoginIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const normalizedIdentifier = loginIdentifier.trim().toLowerCase();
    const fallbackName = normalizedIdentifier.split("@")[0] || "MovieBuzz User";

    setLoading(true);

    try {
      const data = await loginUser(normalizedIdentifier, password);
      const role = data.role === "admin" || data.role === "mod" ? "admin" : "user";
      setUser({
        name: data.name || fallbackName,
        email: data.email || normalizedIdentifier,
        role,
        age: data.age ?? null,
        preferredGenres: data.preferred_genres ?? [],
        preferredMoods: data.preferred_moods ?? [],
      });
      navigate(role === "admin" ? "/admin" : "/dashboard");
    } catch (error) {
      if (error instanceof ApiError && !error.isNetworkError) {
        alert(error.message);
        return;
      }

      if (error instanceof ApiError) {
        alert(error.message);
        return;
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen relative flex items-center justify-center p-4">
      <div className="absolute inset-0 z-0">
        <ImageWithFallback
          src="https://images.unsplash.com/photo-1739433437912-cca661ba902f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080"
          alt="Cinema background"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm"></div>
      </div>

      <Card className="w-full max-w-md relative z-10 shadow-2xl border-border/50 bg-background/95">
        <CardHeader className="space-y-3 text-center">
          <div className="flex justify-center mb-2">
            <MovieBuzzLogo
              size={52}
              theme="light"
              showWordmark={false}
              imageClassName="rounded-full border-border/60 bg-background p-1.5"
            />
          </div>
          <CardTitle className="text-2xl">Welcome Back</CardTitle>
          <CardDescription>
            Sign in to MOVIEBUZZ - Your movie recommendation hub
          </CardDescription>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label>Email</Label>
              <Input
                type="text"
                autoComplete="username"
                placeholder="Enter your email"
                value={loginIdentifier}
                onChange={(e) => setLoginIdentifier(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Password</Label>
                <button
                  type="button"
                  onClick={() => navigate("/forgot-password")}
                  className="text-sm text-primary hover:underline"
                >
                  Forgot password?
                </button>
              </div>
              <div className="relative">
                <Input
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Signing In..." : "Sign In"}
            </Button>
          </form>
        </CardContent>

        <CardFooter className="flex justify-center border-t pt-6">
          <p className="text-sm text-muted-foreground">
            Don't have an account?{" "}
            <button
              className="text-primary hover:underline"
              onClick={() => navigate("/register")}
            >
              Create an account
            </button>
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}
