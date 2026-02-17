import { useCallback, useEffect, useMemo, useState } from 'react';

import SiloFigure from '../components/SiloFigure';
import { useAppContext } from '../context/AppContext';

function round(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '-';
  }
  return value.toFixed(4);
}

function MiniPairBar({ label, leftValue, rightValue, deltaValue }) {
  const denominator = Math.max(Math.abs(leftValue || 0), Math.abs(rightValue || 0), 1e-9);
  const leftPct = (Math.abs(leftValue || 0) / denominator) * 100;
  const rightPct = (Math.abs(rightValue || 0) / denominator) * 100;

  return (
    <div className="rounded-xl border border-stone-200 bg-stone-100/70 p-3 dark:border-zinc-700 dark:bg-zinc-800/60">
      <div className="mb-2 flex items-center justify-between gap-3 text-lg font-semibold">
        <span className="text-stone-800 dark:text-stone-200">{label}</span>
        <span className={`${Number(deltaValue) > 0 ? 'text-amber-700' : 'text-emerald-700'} text-base`}>
          Δ {round(deltaValue)}
        </span>
      </div>
      <div className="space-y-2">
        <div>
          <div className="mb-1 flex justify-between text-lg text-stone-700 dark:text-stone-300">
            <span>Target</span>
            <span>{round(leftValue)}</span>
          </div>
          <div className="h-3.5 rounded-md bg-stone-200 dark:bg-zinc-700">
            <div className="h-3.5 rounded-md bg-sky-500" style={{ width: `${leftPct}%` }} />
          </div>
        </div>
        <div>
          <div className="mb-1 flex justify-between text-lg text-stone-700 dark:text-stone-300">
            <span>Pred</span>
            <span>{round(rightValue)}</span>
          </div>
          <div className="h-3.5 rounded-md bg-stone-200 dark:bg-zinc-700">
            <div className="h-3.5 rounded-md bg-amber-500" style={{ width: `${rightPct}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

function SimpleSeriesBars({ title, unitSuffix = '', values }) {
  const maxValue = Math.max(...values.map((entry) => Math.abs(entry.value)), 1e-9);
  return (
    <div className="rounded-lg border border-stone-200 bg-stone-100/70 p-3 dark:border-zinc-700 dark:bg-zinc-800/60">
      <p className="mb-2 text-sm font-semibold">{title}</p>
      <div className="space-y-2">
        {values.map((entry) => (
          <div key={entry.label}>
            <div className="mb-0.5 flex justify-between text-xs text-stone-700 dark:text-stone-300">
              <span>{entry.label}</span>
              <span>{round(entry.value)}{unitSuffix}</span>
            </div>
            <div className="h-2.5 rounded bg-stone-200 dark:bg-zinc-700">
              <div
                className="h-2.5 rounded bg-emerald-500"
                style={{ width: `${(Math.abs(entry.value) / maxValue) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DashboardPage() {
  const { payload, setOptimizeAction } = useAppContext();
  const [responseData, setResponseData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const targetRows = useMemo(() => {
    if (!responseData) {
      return [];
    }
    const target = payload?.target_params || {};
    const predicted = responseData?.predicted_total_blended_params || {};
    const deltas = responseData?.deltas_vs_target || {};

    return Object.keys(target).map((key) => ({
      key,
      target: target[key],
      predicted: predicted[key],
      delta: deltas[key],
    }));
  }, [payload, responseData]);

  const onOptimize = useCallback(async () => {
    setError('');
    setResponseData(null);

    setLoading(true);
    try {
      const res = await fetch(`${apiBaseUrl}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        const detail = typeof data?.detail === 'string' ? data.detail : JSON.stringify(data);
        throw new Error(detail || `Request failed with status ${res.status}`);
      }
      setResponseData(data);
    } catch (requestError) {
      setError(requestError.message || 'Request failed.');
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, payload]);

  useEffect(() => {
    setOptimizeAction(() => onOptimize);
    return () => setOptimizeAction(null);
  }, [onOptimize, setOptimizeAction]);

  const bestTimesEntries = useMemo(
    () =>
      Object.entries(responseData?.best_times_s || {}).map(([label, value]) => ({
        label,
        value: Number(value || 0),
      })),
    [responseData]
  );

  const bestMassEntries = useMemo(
    () =>
      Object.entries(responseData?.best_masses_kg || {}).map(([label, value]) => ({
        label,
        value: Number(value || 0),
      })),
    [responseData]
  );

  return (
    <>
      {loading ? (
        <p className="mb-4 rounded-md bg-amber-100 p-2 text-sm text-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
          Optimizing...
        </p>
      ) : null}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800 lg:col-span-8">
          <SiloFigure silos={payload.silos} layers={payload.layers} />
        </section>

        <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800 lg:col-span-4">
          <h3 className="mb-3 text-lg font-semibold">Target Parameters</h3>
          <div className="space-y-2">
            {Object.entries(payload.target_params || {}).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between rounded-md bg-stone-200/70 px-3 py-2 text-sm dark:bg-zinc-700/60">
                <span className="text-stone-700 dark:text-stone-300">{key}</span>
                <span className="font-semibold">{round(value)}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800 lg:col-span-4">
          <h3 className="mb-3 text-lg font-semibold">Optimization Summary</h3>
          {!responseData ? (
            <p className="text-sm text-stone-500 dark:text-stone-400">Run optimize to view results.</p>
          ) : (
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="rounded-md bg-stone-200/70 p-2 dark:bg-zinc-700/60">
                <p className="text-stone-500">Optimizer</p>
                <p className="font-semibold">{responseData.optimizer_name}</p>
              </div>
              <div className="rounded-md bg-stone-200/70 p-2 dark:bg-zinc-700/60">
                <p className="text-stone-500">Mode</p>
                <p className="font-semibold">{responseData.mode}</p>
              </div>
              <div className="rounded-md bg-stone-200/70 p-2 dark:bg-zinc-700/60">
                <p className="text-stone-500">Success</p>
                <p className="font-semibold">{String(responseData.success)}</p>
              </div>
              <div className="rounded-md bg-stone-200/70 p-2 dark:bg-zinc-700/60">
                <p className="text-stone-500">Final Error</p>
                <p className="font-semibold">{round(responseData.final_error)}</p>
              </div>
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800 lg:col-span-8">
          <h3 className="mb-3 text-lg font-semibold">Targeted vs Predicted</h3>
          {!targetRows.length ? (
            <p className="text-sm text-stone-500 dark:text-stone-400">No comparison available yet.</p>
          ) : (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {targetRows.map((row) => (
                <MiniPairBar
                  key={`chart-${row.key}`}
                  label={row.key}
                  leftValue={Number(row.target || 0)}
                  rightValue={Number(row.predicted || 0)}
                  deltaValue={Number(row.delta || 0)}
                />
              ))}
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800 lg:col-span-12">
          <h3 className="mb-3 text-lg font-semibold">Best Discharge</h3>
          {!responseData ? (
            <p className="text-sm text-stone-500 dark:text-stone-400">No optimization output yet.</p>
          ) : (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <SimpleSeriesBars title="Best Times" unitSuffix=" s" values={bestTimesEntries} />
              <SimpleSeriesBars title="Best Masses" unitSuffix=" kg" values={bestMassEntries} />
            </div>
          )}
        </section>
      </div>

      {error ? <p className="mt-4 rounded-md bg-red-100 p-2 text-sm text-red-800 dark:bg-red-950/50 dark:text-red-300">{error}</p> : null}
    </>
  );
}

export default DashboardPage;
