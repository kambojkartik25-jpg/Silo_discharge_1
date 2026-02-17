import { createContext, useContext, useEffect, useMemo, useState } from "react";

import defaultPayload from "../data/defaultPayload";

const AppContext = createContext(null);

function AppProvider({ children }) {
  const [payload, setPayload] = useState(defaultPayload);
  const [optimizeAction, setOptimizeAction] = useState(null);
  const [isDark, setIsDark] = useState(() => {
    const saved = window.localStorage.getItem("silo-blend-theme");
    if (saved === "dark") {
      return true;
    }
    if (saved === "light") {
      return false;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    const saved = window.localStorage.getItem("silo-blend-theme");
    if (saved === "dark" || saved === "light") {
      return undefined;
    }

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event) => {
      setIsDark(event.matches);
    };

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", onChange);
      return () => mediaQuery.removeEventListener("change", onChange);
    }

    mediaQuery.addListener(onChange);
    return () => mediaQuery.removeListener(onChange);
  }, []);

  const value = useMemo(
    () => ({
      payload,
      setPayload,
      optimizeAction,
      setOptimizeAction,
      isDark,
      setIsDark,
    }),
    [isDark, optimizeAction, payload],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return context;
}

export { AppProvider, useAppContext };
