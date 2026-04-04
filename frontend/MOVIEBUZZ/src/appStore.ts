import { create } from "zustand";
import { persist } from "zustand/middleware";

export type User = {
  name: string;
  email: string;
  role: "user" | "admin";
  age?: number | null;
  preferredGenres?: string[];
  preferredMoods?: string[];
};

interface AppState {
  user: User | null;
  isDark: boolean;
  setUser: (user: User | null) => void;
  logout: () => void;
  toggleTheme: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      user: null,
      isDark: true,
      setUser: (user) => set({ user }),
      logout: () => set({ user: null }),
      toggleTheme: () => set((state) => ({ isDark: !state.isDark })),
    }),
    {
      name: "moviebuzz-app-store",
      partialize: (state) => ({
        user: state.user,
        isDark: state.isDark,
      }),
    },
  ),
);
