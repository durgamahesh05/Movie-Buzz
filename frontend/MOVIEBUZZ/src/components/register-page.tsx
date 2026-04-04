import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { Check, Eye, EyeOff, X } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, resendUserOtp, signupUser, verifyUserOtp } from "../lib/api";
import { getPasswordPolicyError } from "../lib/password-policy";
import { useAppStore } from "../store/appStore";
import { MovieBuzzLogo } from "./moviebuzz-logo";

export function RegisterPage() {
  const navigate = useNavigate();
  const setUser = useAppStore((state) => state.setUser);
  const [passwordMatch, setPasswordMatch] = useState(true);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [showOtp, setShowOtp] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const passwordChecks = [
    {
      label: "At least 6 characters",
      passed: password.length >= 6,
    },
    {
      label: "1 uppercase letter",
      passed: /[A-Z]/.test(password),
    },
    {
      label: "1 number",
      passed: /\d/.test(password),
    },
    {
      label: "1 special character",
      passed: /[^A-Za-z0-9\s]/.test(password),
    },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const normalizedName = fullName.trim();
    const normalizedEmail = email.trim().toLowerCase();

    const passwordPolicyError = getPasswordPolicyError(password);
    if (passwordPolicyError) {
      setPasswordError(passwordPolicyError);
      setPasswordMatch(true);
      return;
    }

    if (password !== confirmPassword) {
      setPasswordMatch(false);
      setPasswordError("");
      return;
    }

    setPasswordMatch(true);
    setPasswordError("");
    setLoading(true);

    try {
      const data = await signupUser(normalizedName, normalizedEmail, password);
      alert(data.msg || "OTP sent to your email");
      setShowOtp(true);
    } catch (error) {
      if (error instanceof ApiError) {
        alert(error.message);
      } else {
        alert("Unable to create account right now");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOtp = async () => {
    const normalizedEmail = email.trim().toLowerCase();
    if (!otp.trim()) {
      alert("Please enter the OTP");
      return;
    }

    setLoading(true);

    try {
      const data = await verifyUserOtp(normalizedEmail, otp.trim());
      setUser({
        name: fullName.trim(),
        email: normalizedEmail,
        role: "user",
        age: null,
        preferredGenres: [],
        preferredMoods: [],
      });
      const nextTarget = data.next_target || "/preferences-setup";
      const alertMessage = data.welcome_email_sent === false
        ? `${data.msg || "Account verified"} Welcome email could not be sent, but your account is active.`
        : data.msg || "Account verified";
      alert(alertMessage);
      window.open(nextTarget, "_self");
    } catch (error) {
      if (error instanceof ApiError) {
        alert(error.message);
      } else {
        alert("Unable to verify OTP right now");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResendOtp = async () => {
    if (!email.trim()) {
      alert("Please enter your email first");
      return;
    }

    setLoading(true);

    try {
      const data = await resendUserOtp(email.trim().toLowerCase());
      alert(data.msg || "New OTP sent");
    } catch (error) {
      if (error instanceof ApiError) {
        alert(error.message);
      } else {
        alert("Unable to resend OTP right now");
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
          <CardTitle className="text-2xl">Create Account</CardTitle>
          <CardDescription>
            Join MOVIEBUZZ and set your preferences right after account creation.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label>Full Name</Label>
              <Input
                placeholder="Enter your full name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label>Email</Label>
              <Input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label>Password</Label>
              <div className="relative">
                <Input
                  name="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    setPasswordError("");
                  }}
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
              {password.length > 0 && (
                <div className="space-y-2 rounded-lg border border-border/60 bg-muted/20 p-3">
                  {passwordChecks.map((check) => (
                    <div
                      key={check.label}
                      className={`flex items-center gap-2 text-xs ${
                        check.passed ? "text-green-600" : "text-muted-foreground"
                      }`}
                    >
                      {check.passed ? (
                        <Check className="h-3.5 w-3.5" />
                      ) : (
                        <X className="h-3.5 w-3.5" />
                      )}
                      <span>{check.label}</span>
                    </div>
                  ))}
                </div>
              )}
              {passwordError && <p className="text-sm text-red-500">{passwordError}</p>}
            </div>

            <div className="space-y-2">
              <Label>Confirm Password</Label>
              <div className="relative">
                <Input
                  name="confirmPassword"
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="Confirm your password"
                  value={confirmPassword}
                  onChange={(e) => {
                    setConfirmPassword(e.target.value);
                    setPasswordMatch(true);
                  }}
                  required
                  className="pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {!passwordMatch && (
                <p className="text-sm text-red-500">Passwords do not match</p>
              )}
            </div>

            <Button type="submit" className="w-full mt-6" disabled={loading}>
              {loading ? "Creating..." : "Create Account"}
            </Button>
          </form>

          {showOtp && (
            <div className="mt-6 space-y-3">
              <Label>Enter OTP (check your email)</Label>
              <Input
                placeholder="Enter OTP"
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
              />
              <Button onClick={handleVerifyOtp} className="w-full" disabled={loading}>
                {loading ? "Verifying..." : "Verify OTP"}
              </Button>
              <button
                type="button"
                onClick={handleResendOtp}
                className="w-full text-sm text-primary hover:underline"
                disabled={loading}
              >
                Resend OTP
              </button>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex justify-center border-t pt-6">
          <p className="text-sm text-muted-foreground">
            Already have an account?{" "}
            <button className="text-primary hover:underline" onClick={() => navigate("/login")}>
              Sign in
            </button>
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}
