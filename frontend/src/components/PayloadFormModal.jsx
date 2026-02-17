import { useEffect, useState } from "react";

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function PayloadFormModal({ isOpen, onClose, initialPayload, onSubmit }) {
  const [formState, setFormState] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isOpen || !initialPayload) {
      return;
    }
    setFormState({
      silos: pretty(initialPayload.silos),
      layers: pretty(initialPayload.layers),
      suppliers: pretty(initialPayload.suppliers),
      target_params: pretty(initialPayload.target_params),
      weights: pretty(initialPayload.weights),
      material: pretty(initialPayload.material),
      beverloo: pretty(initialPayload.beverloo),
      optimizer_settings: pretty(initialPayload.optimizer_settings),
      sigma_m: String(initialPayload.sigma_m),
      steps: String(initialPayload.steps),
      auto_adjust: Boolean(initialPayload.auto_adjust),
      mode: initialPayload.mode,
      fixed_total_mass_kg:
        initialPayload.fixed_total_mass_kg === null ||
        initialPayload.fixed_total_mass_kg === undefined
          ? ""
          : String(initialPayload.fixed_total_mass_kg),
      optimizer: initialPayload.optimizer,
      return_debug: Boolean(initialPayload.return_debug),
    });
    setError("");
  }, [initialPayload, isOpen]);

  if (!isOpen || !formState) {
    return null;
  }

  const onChange = (name, value) => {
    setFormState((prev) => ({ ...prev, [name]: value }));
  };

  const parseJsonField = (rawValue, label) => {
    try {
      return JSON.parse(rawValue);
    } catch (parseFieldError) {
      const baseMessage =
        parseFieldError && parseFieldError.message
          ? parseFieldError.message
          : "Unable to parse JSON.";
      throw new Error(`Invalid JSON in ${label}: ${baseMessage}`);
    }
  };

  const onSave = () => {
    try {
      const nextPayload = {
        silos: parseJsonField(formState.silos, "Silos"),
        layers: parseJsonField(formState.layers, "Layers"),
        suppliers: parseJsonField(formState.suppliers, "Suppliers"),
        material: parseJsonField(formState.material, "Material"),
        beverloo: parseJsonField(formState.beverloo, "Beverloo"),
        sigma_m: Number(formState.sigma_m),
        steps: Number(formState.steps),
        auto_adjust: Boolean(formState.auto_adjust),
        target_params: parseJsonField(formState.target_params, "Target Params"),
        weights: parseJsonField(formState.weights, "Weights"),
        mode: formState.mode,
        fixed_total_mass_kg:
          formState.fixed_total_mass_kg === ""
            ? null
            : Number(formState.fixed_total_mass_kg),
        optimizer: formState.optimizer,
        optimizer_settings: parseJsonField(
          formState.optimizer_settings,
          "Optimizer Settings",
        ),
        return_debug: Boolean(formState.return_debug),
      };
      onSubmit(nextPayload);
      onClose();
    } catch (parseError) {
      setError(parseError?.message || "Invalid JSON in one or more fields.");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-950/45 p-4">
      <div
        className="max-h-[92vh] w-full max-w-6xl overflow-auto rounded-2xl border border-stone-300 bg-stone-50 p-5 shadow-2xl dark:border-zinc-700 dark:bg-zinc-800"
        role="dialog"
        aria-modal="true"
        aria-labelledby="payload-form-modal-title"
      >
        <div className="mb-4 flex items-center justify-between">
          <h2
            id="payload-form-modal-title"
            className="text-xl font-semibold text-stone-900 dark:text-stone-100"
          >
            Input Form
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-stone-300 bg-stone-100 px-3 py-1 text-sm text-stone-700 hover:bg-stone-200 dark:border-zinc-600 dark:bg-zinc-700 dark:text-stone-200 dark:hover:bg-zinc-600"
          >
            Close
          </button>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Silos (JSON)
            </label>
            <textarea
              className="h-48 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.silos}
              onChange={(e) => onChange("silos", e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Layers (JSON)
            </label>
            <textarea
              className="h-48 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.layers}
              onChange={(e) => onChange("layers", e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Suppliers (JSON)
            </label>
            <textarea
              className="h-48 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.suppliers}
              onChange={(e) => onChange("suppliers", e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Target Params (JSON)
            </label>
            <textarea
              className="h-48 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.target_params}
              onChange={(e) => onChange("target_params", e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Weights (JSON)
            </label>
            <textarea
              className="h-40 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.weights}
              onChange={(e) => onChange("weights", e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Optimizer Settings (JSON)
            </label>
            <textarea
              className="h-40 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.optimizer_settings}
              onChange={(e) => onChange("optimizer_settings", e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Material (JSON)
            </label>
            <textarea
              className="h-28 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.material}
              onChange={(e) => onChange("material", e.target.value)}
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
              Beverloo (JSON)
            </label>
            <textarea
              className="h-28 w-full rounded-md border border-stone-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.beverloo}
              onChange={(e) => onChange("beverloo", e.target.value)}
            />
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <label className="text-sm text-stone-700 dark:text-stone-300">
            sigma_m
            <input
              className="mt-1 w-full rounded-md border border-stone-300 bg-white p-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.sigma_m}
              onChange={(e) => onChange("sigma_m", e.target.value)}
            />
          </label>
          <label className="text-sm text-stone-700 dark:text-stone-300">
            steps
            <input
              className="mt-1 w-full rounded-md border border-stone-300 bg-white p-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.steps}
              onChange={(e) => onChange("steps", e.target.value)}
            />
          </label>
          <label className="text-sm text-stone-700 dark:text-stone-300">
            mode
            <select
              className="mt-1 w-full rounded-md border border-stone-300 bg-white p-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.mode}
              onChange={(e) => onChange("mode", e.target.value)}
            >
              <option value="A">A</option>
              <option value="B">B</option>
            </select>
          </label>
          <label className="text-sm text-stone-700 dark:text-stone-300">
            fixed_total_mass_kg
            <input
              className="mt-1 w-full rounded-md border border-stone-300 bg-white p-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.fixed_total_mass_kg}
              onChange={(e) => onChange("fixed_total_mass_kg", e.target.value)}
              placeholder="null for mode B"
            />
          </label>
          <label className="text-sm text-stone-700 dark:text-stone-300">
            optimizer
            <select
              className="mt-1 w-full rounded-md border border-stone-300 bg-white p-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
              value={formState.optimizer}
              onChange={(e) => onChange("optimizer", e.target.value)}
            >
              <option value="slsqp">slsqp</option>
              <option value="least_squares">least_squares</option>
              <option value="trust_constr">trust_constr</option>
              <option value="bayes">bayes</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-stone-700 dark:text-stone-300">
            <input
              type="checkbox"
              checked={formState.auto_adjust}
              onChange={(e) => onChange("auto_adjust", e.target.checked)}
            />
            auto_adjust
          </label>
          <label className="flex items-center gap-2 text-sm text-stone-700 dark:text-stone-300">
            <input
              type="checkbox"
              checked={formState.return_debug}
              onChange={(e) => onChange("return_debug", e.target.checked)}
            />
            return_debug
          </label>
        </div>

        {error ? (
          <p className="mt-4 rounded-md bg-red-100 p-2 text-sm text-red-800 dark:bg-red-950/50 dark:text-red-300">
            {error}
          </p>
        ) : null}

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-stone-300 bg-stone-100 px-3 py-2 text-sm text-stone-700 hover:bg-stone-200 dark:border-zinc-600 dark:bg-zinc-700 dark:text-stone-200 dark:hover:bg-zinc-600"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSave}
            className="rounded-md bg-amber-800 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700"
          >
            Save Input
          </button>
        </div>
      </div>
    </div>
  );
}

export default PayloadFormModal;
