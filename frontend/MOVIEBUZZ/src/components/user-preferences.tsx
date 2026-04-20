import React, { useEffect, useState } from "react";
import {
  AlertCircle,
  Check,
  Ghost,
  Heart,
  Loader2,
  Save,
  Smile,
  Tv,
  Zap
} from "lucide-react";
import {
  getUserPreferences,
  saveUserPreferences,
  type UserPreferences as ApiUserPreferences,
} from "../lib/api";
import { invalidateHomeMovieCache } from "../lib/movie-cache";
import type { User } from "../store/appStore";

const GENRES = [
  { id: "action", label: "Action", icon: Zap },
  { id: "comedy", label: "Comedy", icon: Smile },
  { id: "drama", label: "Drama", icon: Tv },
  { id: "sci-fi", label: "Sci-Fi", icon: Zap },
  { id: "thriller", label: "Thriller", icon: Ghost },
  { id: "horror", label: "Horror", icon: Ghost },
  { id: "romance", label: "Romance", icon: Heart },
  { id: "animation", label: "Animation", icon: Smile },
];

const MOODS = [
  { label: "Happy", value: "happy" },
  { label: "Excited", value: "excited" },
  { label: "Sad", value: "sad" },
  { label: "Relaxed", value: "relaxed" },
  { label: "Scary", value: "scared" },
  { label: "Thoughtful", value: "thoughtful" },
];

const formStyles = {
  alert: {
    display: "flex",
    alignItems: "flex-start",
    gap: "12px",
    borderRadius: "24px",
    border: "1px solid transparent",
    padding: "16px",
    fontSize: "14px",
    fontWeight: 600,
    transition: "all 0.3s ease",
  } satisfies React.CSSProperties,
  label: {
    marginLeft: "4px",
    color: "#9ca3af",
    fontSize: "13px",
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.1em",
  } satisfies React.CSSProperties,
  input: {
    width: "100%",
    height: "54px",
    padding: "0 24px",
    borderRadius: "24px",
    border: "1px solid #e5e7eb",
    backgroundColor: "#ffffff",
    color: "#111827",
    boxShadow: "0 1px 2px rgba(0, 0, 0, 0.04)",
    outline: "none",
  } satisfies React.CSSProperties,
  genreButton: {
    display: "flex",
    height: "50px",
    alignItems: "center",
    padding: "0 20px",
    borderRadius: "24px",
    border: "1px solid #e5e7eb",
    backgroundColor: "#ffffff",
    color: "#4b5563",
    fontSize: "14px",
    fontWeight: 700,
    transition: "all 0.2s ease",
  } satisfies React.CSSProperties,
  genreButtonSelected: {
    transform: "scale(1.02)",
    border: "1px solid #08080c",
    backgroundColor: "#08080c",
    color: "#ffffff",
    boxShadow: "0 12px 30px rgba(0, 0, 0, 0.12)",
  } satisfies React.CSSProperties,
  moodButton: {
    borderRadius: "999px",
    border: "1px solid #e5e7eb",
    backgroundColor: "#ffffff",
    padding: "12px 24px",
    color: "#6b7280",
    fontSize: "11px",
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.18em",
    transition: "all 0.2s ease",
  } satisfies React.CSSProperties,
  moodButtonSelected: {
    border: "1px solid #08080c",
    backgroundColor: "#08080c",
    color: "#ffffff",
  } satisfies React.CSSProperties,
  submitButton: {
    display: "flex",
    width: "100%",
    height: "62px",
    alignItems: "center",
    justifyContent: "center",
    gap: "12px",
    borderRadius: "24px",
    border: "none",
    backgroundColor: "#08080c",
    color: "#ffffff",
    fontSize: "16px",
    fontWeight: 700,
    boxShadow: "0 20px 36px rgba(0, 0, 0, 0.2)",
    transition: "all 0.2s ease",
  } satisfies React.CSSProperties,
} as const;

type PreferencesFormData = {
  age: string;
  genres: string[];
  mood: string;
};

type PreferencesMessage = {
  type: "" | "error" | "success";
  text: string;
};

export type LoadedUserProfile = {
  name?: string;
  email?: string;
  age?: number | null;
  preferredGenres: string[];
  preferredMoods: string[];
};

export type SavedUserPreferences = {
  age: number | null;
  preferredGenres: string[];
  preferredMoods: string[];
};

type UserPreferencesProps = {
  onClose: () => void;
  onSaved?: (preferences: SavedUserPreferences) => void;
  onProfileLoaded?: (profile: LoadedUserProfile) => void;
  user: User;
};

function buildInitialFormData(user: User): PreferencesFormData {
  return {
    age: typeof user.age === "number" ? String(user.age) : "18",
    genres: user.preferredGenres || [],
    mood: user.preferredMoods?.[0] || "",
  };
}

export function UserPreferences({
  onClose,
  onSaved,
  onProfileLoaded,
  user,
}: UserPreferencesProps) {
  const initialFormData = buildInitialFormData(user);
  const [formData, setFormData] = useState<PreferencesFormData>(() => initialFormData);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingProfile, setIsLoadingProfile] = useState(true);
  const [message, setMessage] = useState<PreferencesMessage>({ type: "", text: "" });

  useEffect(() => {
    let ignore = false;

    async function loadProfile() {
      setIsLoadingProfile(true);
      try {
        const profile = await getUserPreferences(user.email);
        if (ignore) {
          return;
        }

        const preferredGenres = profile.preferred_genres ?? [];
        const preferredMoods = profile.preferred_moods ?? [];
        setFormData({
          age: typeof profile.age === "number" ? String(profile.age) : initialFormData.age,
          genres: preferredGenres,
          mood: preferredMoods[0] ?? "",
        });
        onProfileLoaded?.({
          name: profile.name,
          email: profile.email,
          age: profile.age,
          preferredGenres,
          preferredMoods,
        });
      } catch (err) {
        if (!ignore) {
          setMessage({
            type: "error",
            text: err instanceof Error ? err.message : "Unable to load saved preferences.",
          });
        }
      } finally {
        if (!ignore) {
          setIsLoadingProfile(false);
        }
      }
    }

    void loadProfile();
    return () => {
      ignore = true;
    };
  }, [user.email]);

  const toggleGenre = (genreLabel: string) => {
    setFormData((p) => ({
      ...p,
      genres: p.genres.includes(genreLabel)
        ? p.genres.filter((g: string) => g !== genreLabel)
        : [...p.genres, genreLabel],
    }));
  };

  const selectMood = (val: string) => setFormData((p) => ({ ...p, mood: p.mood === val ? "" : val }));

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage({ type: "", text: "" });

    try {
      const parsedAge = Number(formData.age);
      if (isNaN(parsedAge) || parsedAge < 1 || parsedAge > 120) 
        throw new Error("Please enter a valid age.");
      if (formData.genres.length === 0) 
        throw new Error("Please select at least one genre.");

      const payload: ApiUserPreferences = {
        age: parsedAge,
        preferred_genres: formData.genres,
        preferred_moods: formData.mood ? [formData.mood] : [],
      };

      const saved = await saveUserPreferences(user.email, payload);
      invalidateHomeMovieCache();

      const savedAge =
        typeof saved.age === "number" ? saved.age : parsedAge;
      const savedGenres =
        saved.preferred_genres ?? formData.genres;
      const savedMoods =
        saved.preferred_moods ?? (formData.mood ? [formData.mood] : []);
      
      onSaved?.({
        age: savedAge,
        preferredGenres: savedGenres,
        preferredMoods: savedMoods,
      });

      setMessage({ type: "success", text: "Preferences saved! Curating your feed..." });
      setTimeout(onClose, 1500);
    } catch (err) {
      setMessage({ type: "error", text: err instanceof Error ? err.message : "Error saving preferences." });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {message.text && (
        <div
          style={{
            ...formStyles.alert,
            ...(message.type === "error"
              ? {
                  borderColor: "#fee2e2",
                  backgroundColor: "#fef2f2",
                  color: "#dc2626",
                }
              : {
                  borderColor: "#dcfce7",
                  backgroundColor: "#f0fdf4",
                  color: "#16a34a",
                }),
          }}
        >
          {message.type === "error" ? (
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
          ) : (
            <Check size={18} className="mt-0.5 shrink-0" />
          )}
          <p>{message.text}</p>
        </div>
      )}

      {/* Age Section */}
      <div className="space-y-2.5">
        <label style={formStyles.label}>
          Age
        </label>
        <input
          type="number"
          value={formData.age}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setFormData((p) => ({ ...p, age: e.target.value }))
          }
          style={{
            ...formStyles.input,
            opacity: isLoadingProfile ? 0.7 : 1,
            cursor: isLoadingProfile ? "not-allowed" : "text",
          }}
          placeholder="How old are you?"
          disabled={isLoadingProfile}
        />
      </div>

      {/* Genres Section */}
      <div className="space-y-3.5">
        <label style={formStyles.label}>
          Select Genres
        </label>
        <div className="grid grid-cols-2 gap-3">
          {GENRES.map((g) => {
            const isSelected = formData.genres.includes(g.label);
            const Icon = g.icon;
            return (
              <button
                key={g.id}
                type="button"
                onClick={() => toggleGenre(g.label)}
                disabled={isLoadingProfile}
                style={{
                  ...formStyles.genreButton,
                  ...(isSelected ? formStyles.genreButtonSelected : null),
                  ...(isLoadingProfile
                    ? { opacity: 0.7, cursor: "not-allowed" }
                    : { cursor: "pointer" }),
                }}
              >
                <Icon size={16} className={`mr-3 ${isSelected ? "text-white" : "text-gray-400"}`} />
                {g.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Mood Section */}
      <div className="space-y-3.5">
        <label style={formStyles.label}>
          Vibe
        </label>
        <div className="flex flex-wrap gap-2.5">
          {MOODS.map((m) => {
            const isSelected = formData.mood === m.value;
            return (
              <button
                key={m.value}
                type="button"
                onClick={() => selectMood(m.value)}
                disabled={isLoadingProfile}
                style={{
                  ...formStyles.moodButton,
                  ...(isSelected ? formStyles.moodButtonSelected : null),
                  ...(isLoadingProfile
                    ? { opacity: 0.7, cursor: "not-allowed" }
                    : { cursor: "pointer" }),
                }}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Submit */}
      <div className="pt-4">
        <button
          type="submit"
          disabled={isSubmitting || isLoadingProfile}
          style={{
            ...formStyles.submitButton,
            opacity: isSubmitting || isLoadingProfile ? 0.7 : 1,
            cursor: isSubmitting || isLoadingProfile ? "not-allowed" : "pointer",
          }}
        >
          {isSubmitting || isLoadingProfile ? (
            <Loader2 className="animate-spin" size={22} />
          ) : (
            <>
              <Save size={20} />
              <span>Save Preferences</span>
            </>
          )}
        </button>
        
        <button
          type="button"
          onClick={onClose}
          disabled={isSubmitting}
          className="mt-6 w-full text-center text-xs font-bold uppercase tracking-[0.2em] text-gray-400 transition-colors hover:text-gray-900"
        >
          Skip for now
        </button>
      </div>
    </form>
  );
}
