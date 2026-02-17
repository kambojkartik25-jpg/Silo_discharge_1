import { useEffect } from 'react';
import { Navigate, NavLink, Route, Routes } from 'react-router-dom';
import { useLocation } from 'react-router-dom';

import { useAppContext } from './context/AppContext';
import DashboardPage from './pages/DashboardPage';
import InputFormPage from './pages/InputFormPage';

function App() {
  const { isDark, setIsDark, optimizeAction } = useAppContext();
  const location = useLocation();
  const isDashboardRoute = location.pathname === '/dashboard' || location.pathname === '/';

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add('dark');
      window.localStorage.setItem('silo-blend-theme', 'dark');
    } else {
      root.classList.remove('dark');
      window.localStorage.setItem('silo-blend-theme', 'light');
    }
  }, [isDark]);

  return (
    <div className="min-h-screen bg-stone-100 text-stone-900 dark:bg-zinc-900 dark:text-stone-100">
      <div className="mx-auto max-w-7xl p-6">
        <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold">Silo Blend Dashboard</h1>
            <p className="text-sm text-stone-600 dark:text-stone-300">Navigate between dashboard and form editor using shared global state.</p>
          </div>
          <div className="flex items-center gap-2">
            <nav className="flex items-center gap-2">
              <NavLink
                to="/dashboard"
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm ${
                    isActive
                      ? 'bg-amber-800 text-white'
                      : 'border border-stone-300 bg-stone-50 text-stone-700 hover:bg-stone-200 dark:border-zinc-700 dark:bg-zinc-800 dark:text-stone-200 dark:hover:bg-zinc-700'
                  }`
                }
              >
                Dashboard
              </NavLink>
              <NavLink
                to="/form"
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm ${
                    isActive
                      ? 'bg-amber-800 text-white'
                      : 'border border-stone-300 bg-stone-50 text-stone-700 hover:bg-stone-200 dark:border-zinc-700 dark:bg-zinc-800 dark:text-stone-200 dark:hover:bg-zinc-700'
                  }`
                }
              >
                Input Form
              </NavLink>
            </nav>
            {isDashboardRoute ? (
              <button
                type="button"
                onClick={() => optimizeAction?.()}
                disabled={!optimizeAction}
                className="rounded-lg bg-amber-800 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:cursor-not-allowed disabled:bg-stone-400"
              >
                Run Optimize
              </button>
            ) : null}
            <button
              type="button"
              onClick={() => setIsDark((prev) => !prev)}
              className="rounded-lg border border-stone-300 bg-stone-50 px-3 py-2 text-sm text-stone-700 hover:bg-stone-200 dark:border-zinc-700 dark:bg-zinc-800 dark:text-stone-200 dark:hover:bg-zinc-700"
            >
              {isDark ? 'Light Mode' : 'Dark Mode'}
            </button>
          </div>
        </header>

        <Routes>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/form" element={<InputFormPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
