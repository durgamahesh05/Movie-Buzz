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
import { Eye, EyeOff, ArrowLeft } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ApiError,
  requestPasswordResetOtp,
  resetPassword,
  verifyPasswordResetOtp,
} from "../lib/api";
import {
  getPasswordPolicyError,
  PASSWORD_POLICY_MESSAGE,
} from "../lib/password-policy";
import { MovieBuzzLogo } from "./moviebuzz-logo";

type Step = "email" | "otp" | "newPassword";

export function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>("email");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [passwordError, setPasswordError] = useState("");

  // Step 1: Request OTP
  const handleRequestOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await requestPasswordResetOtp(email.trim().toLowerCase());
      alert(data.msg || "OTP sent to your email");
      setStep("otp");
    } catch (error) {
      if (error instanceof ApiError) {
        alert(error.message);
      } else {
        alert("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Verify OTP
  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await verifyPasswordResetOtp(
        email.trim().toLowerCase(),
        otp.trim(),
      );
      alert(data.msg || "OTP verified");
      setStep("newPassword");
    } catch (error) {
      if (error instanceof ApiError) {
        alert(error.message);
      } else {
        alert("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Step 3: Reset Password
  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    const policyError = getPasswordPolicyError(newPassword);
    if (policyError) {
      setPasswordError(policyError);
      return;
    }
    if (newPassword !== confirmPassword) {
      setPasswordError("");
      alert("Passwords do not match");
      return;
    }
    setPasswordError("");
    setLoading(true);
    try {
      const data = await resetPassword(
        email.trim().toLowerCase(),
        otp.trim(),
        newPassword,
      );
      alert(data.msg || "Password reset successful");
      navigate("/login");
    } catch (error) {
      if (error instanceof ApiError) {
        alert(error.message);
      } else {
        alert("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const stepTitles: Record<Step, { title: string; description: string }> = {
    email: {
      title: "Forgot Password",
      description: "Enter your email and we'll send you a reset OTP",
    },
    otp: {
      title: "Verify OTP",
      description: `We've sent an OTP to ${email}`,
    },
    newPassword: {
      title: "Reset Password",
      description: "Enter your new password below",
    },
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
          <CardTitle className="text-2xl">{stepTitles[step].title}</CardTitle>
          <CardDescription>{stepTitles[step].description}</CardDescription>
        </CardHeader>

        <CardContent>
          {/* STEP 1: Email */}
          {step === "email" && (
            <form onSubmit={handleRequestOtp} className="space-y-4">
              <div className="space-y-2">
                <Label>Email</Label>
                <Input
                  type="email"
                  placeholder="Enter your registered email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Sending..." : "Send OTP"}
              </Button>
            </form>
          )}

          {/* STEP 2: OTP */}
          {step === "otp" && (
            <form onSubmit={handleVerifyOtp} className="space-y-4">
              <div className="space-y-2">
                <Label>OTP</Label>
                <Input
                  placeholder="Enter the OTP from your email"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  required
                />
              </div>
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Verifying..." : "Verify OTP"}
              </Button>
              <button
                type="button"
                onClick={() => setStep("email")}
                className="w-full text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Didn't receive it? Go back and retry
              </button>
            </form>
          )}

          {/* STEP 3: New Password */}
          {step === "newPassword" && (
            <form onSubmit={handleResetPassword} className="space-y-4">
              <div className="space-y-2">
                <Label>New Password</Label>
                <div className="relative">
                  <Input
                    type={showNewPassword ? "text" : "password"}
                    placeholder="Enter new password"
                    value={newPassword}
                    onChange={(e) => {
                      setNewPassword(e.target.value);
                      setPasswordError("");
                    }}
                    required
                    className="pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showNewPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-xs text-muted-foreground">{PASSWORD_POLICY_MESSAGE}</p>
                {passwordError ? <p className="text-sm text-red-500">{passwordError}</p> : null}
              </div>

              <div className="space-y-2">
                <Label>Confirm New Password</Label>
                <div className="relative">
                  <Input
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="Confirm new password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
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
              </div>

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? "Resetting..." : "Reset Password"}
              </Button>
            </form>
          )}
        </CardContent>

        <CardFooter className="flex justify-center border-t pt-6">
          <button
            onClick={() => navigate("/login")}
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-3 h-3" />
            Back to Sign In
          </button>
        </CardFooter>
      </Card>
    </div>
  );
}
