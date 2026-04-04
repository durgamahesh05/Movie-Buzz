import React, { useEffect } from "react";
import {
  UserPreferences,
  type LoadedUserProfile,
  type SavedUserPreferences,
} from "./user-preferences";
import { MovieBuzzLogo } from "./moviebuzz-logo";
import { useAppStore } from "../store/appStore";

const pageStyles = {
  shell: {
    position: "relative",
    display: "flex",
    minHeight: "100vh",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px",
    backgroundColor: "#030303",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    WebkitFontSmoothing: "antialiased",
  } satisfies React.CSSProperties,
  backgroundImage: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    opacity: 0.3,
    transform: "scale(1.05)",
    filter: "blur(80px)",
  } satisfies React.CSSProperties,
  backgroundOverlay: {
    position: "absolute",
    inset: 0,
    background:
      "linear-gradient(to top right, rgba(0,0,0,1), rgba(0,0,0,0.2), rgba(0,0,0,1))",
  } satisfies React.CSSProperties,
  card: {
    position: "relative",
    zIndex: 10,
    display: "flex",
    width: "100%",
    maxWidth: "500px",
    flexDirection: "column",
    overflow: "hidden",
    borderRadius: "48px",
    border: "1px solid rgba(255,255,255,0.2)",
    backgroundColor: "#f5f5f7",
    boxShadow: "0 40px 80px -20px rgba(0, 0, 0, 0.6)",
  } satisfies React.CSSProperties,
  header: {
    padding: "64px 56px 48px",
    textAlign: "center",
  } satisfies React.CSSProperties,
  logoWrap: {
    display: "flex",
    justifyContent: "center",
    marginBottom: "16px",
  } satisfies React.CSSProperties,
  title: {
    marginBottom: "16px",
    color: "#111827",
    fontSize: "34px",
    fontWeight: 900,
    lineHeight: 1.1,
    letterSpacing: "-0.025em",
  } satisfies React.CSSProperties,
  subtitle: {
    maxWidth: "320px",
    margin: "0 auto",
    color: "#6b7280",
    fontSize: "15px",
    fontWeight: 500,
    lineHeight: 1.7,
  } satisfies React.CSSProperties,
  content: {
    padding: "0 56px 64px",
  } satisfies React.CSSProperties,
} as const;

export function PreferencesSetupPage() {
  const user = useAppStore((state) => state.user);
  const setUser = useAppStore((state) => state.setUser);

  useEffect(() => {
    if (!user) {
      window.open("/login", "_self");
    }
  }, [user]);

  if (!user) return null;

  return (
    <div className="relative" style={pageStyles.shell}>
      {/* Immersive Blurred Background */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        <img
          src="https://images.unsplash.com/photo-1739433437912-cca661ba902f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080"
          alt="Cinema background"
          style={pageStyles.backgroundImage}
        />
        <div style={pageStyles.backgroundOverlay} />
      </div>

      {/* Premium Onboarding Card */}
      <div style={pageStyles.card}>
        {/* Simplified Header */}
        <div style={pageStyles.header}>
          <div style={pageStyles.logoWrap}>
            <MovieBuzzLogo
              size={52}
              theme="light"
              showWordmark={false}
              imageClassName="rounded-full border-border/60 bg-background p-1.5"
            />
          </div>
          <h1 style={pageStyles.title}>
            Personalize Your Feed
          </h1>
          <p style={pageStyles.subtitle}>
            Tell us what you like to help MovieBuzz find the perfect matches for your night.
          </p>
        </div>

        {/* Content Section */}
        <div style={pageStyles.content}>
          <UserPreferences
            user={user}
            onProfileLoaded={(profile: LoadedUserProfile) => {
              const nextUser = {
                ...user,
                name: profile.name || user.name,
                email: profile.email || user.email,
                age: profile.age ?? user.age ?? null,
                preferredGenres: profile.preferredGenres,
                preferredMoods: profile.preferredMoods,
              };

              if (
                nextUser.name === user.name &&
                nextUser.email === user.email &&
                nextUser.age === user.age &&
                JSON.stringify(nextUser.preferredGenres ?? []) === JSON.stringify(user.preferredGenres ?? []) &&
                JSON.stringify(nextUser.preferredMoods ?? []) === JSON.stringify(user.preferredMoods ?? [])
              ) {
                return;
              }

              setUser({
                ...nextUser,
              });
            }}
            onSaved={(preferences: SavedUserPreferences) => {
              setUser({
                ...user,
                age: preferences.age,
                preferredGenres: preferences.preferredGenres,
                preferredMoods: preferences.preferredMoods,
              });
            }}
            onClose={() => window.open("/dashboard", "_self")}
          />
        </div>
      </div>
    </div>
  );
}

export default PreferencesSetupPage;
