import { Link } from 'react-router-dom';

import { useAppContext } from '../context/AppContext';

function InputFormPage() {
  const { payload, setPayload } = useAppContext();

  const silos = payload.silos || [];
  const suppliers = payload.suppliers || [];

  const updatePayload = (key, value) => {
    setPayload((prev) => ({ ...prev, [key]: value }));
  };

  const normalizeSilos = (silosInput) => silosInput.map((silo, index) => ({ ...silo, silo_id: `S${index + 1}` }));

  const updateArrayField = (section, index, field, rawValue) => {
    const next = [...payload[section]];
    next[index] = {
      ...next[index],
      [field]: rawValue,
    };
    if (section === 'silos') {
      updatePayload(section, normalizeSilos(next));
      return;
    }
    updatePayload(section, next);
  };

  const removeArrayRow = (section, index) => {
    const next = payload[section].filter((_, rowIndex) => rowIndex !== index);
    if (section === 'silos') {
      const normalized = normalizeSilos(next);
      const siloIds = new Set(normalized.map((silo) => silo.silo_id));
      const nextLayers = payload.layers.filter((layer) => siloIds.has(layer.silo_id));
      setPayload((prev) => ({ ...prev, silos: normalized, layers: nextLayers }));
      return;
    }
    updatePayload(section, next);
  };

  const addSiloRow = () => {
    const next = normalizeSilos([
      ...silos,
      {
        silo_id: '',
        capacity_kg: 4000,
        body_diameter_m: 3,
        outlet_diameter_m: 0.2,
        initial_mass_kg: 0,
      },
    ]);
    updatePayload('silos', next);
  };

  const addLayerRow = () => {
    updatePayload('layers', [
      ...(payload.layers || []),
      {
        silo_id: silos[0]?.silo_id || 'S1',
        layer_index: 1,
        lot_id: `L${Date.now().toString().slice(-4)}`,
        supplier: suppliers[0]?.supplier || '',
        segment_mass_kg: 0,
      },
    ]);
  };

  const addSupplierRow = () => {
    updatePayload('suppliers', [
      ...(payload.suppliers || []),
      {
        supplier: `Supplier_${(payload.suppliers || []).length + 1}`,
        moisture_pct: 0,
        fine_extract_db_pct: 0,
        wort_pH: 0,
        diastatic_power_WK: 0,
        total_protein_pct: 0,
        wort_colour_EBC: 0,
      },
    ]);
  };

  const updateObjectEntry = (section, oldKey, newKey, value) => {
    const entries = Object.entries(payload[section] || {}).map(([key, val]) =>
      key === oldKey ? [newKey, value] : [key, val]
    );
    updatePayload(section, Object.fromEntries(entries));
  };

  const removeObjectEntry = (section, keyToRemove) => {
    const entries = Object.entries(payload[section] || {}).filter(([key]) => key !== keyToRemove);
    updatePayload(section, Object.fromEntries(entries));
  };

  const addObjectEntry = (section, defaultKey) => {
    const existing = payload[section] || {};
    let nextKey = defaultKey;
    let count = 1;
    while (Object.prototype.hasOwnProperty.call(existing, nextKey)) {
      nextKey = `${defaultKey}_${count}`;
      count += 1;
    }
    updatePayload(section, { ...existing, [nextKey]: 0 });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-xl font-semibold">Input Form</h2>
        <Link
          to="/dashboard"
          className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 hover:bg-amber-100 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200 dark:hover:bg-amber-900/50"
        >
          Back to Dashboard
        </Link>
      </div>

      <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Silos</h3>
          <button type="button" onClick={addSiloRow} className="rounded-md bg-amber-800 px-3 py-1 text-sm text-white hover:bg-amber-700">+ Add Row</button>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-stone-300 dark:border-zinc-700">
                <th className="px-2 py-2 text-left">silo_id</th>
                <th className="px-2 py-2 text-left">capacity_kg</th>
                <th className="px-2 py-2 text-left">body_diameter_m</th>
                <th className="px-2 py-2 text-left">outlet_diameter_m</th>
                <th className="px-2 py-2 text-left">initial_mass_kg</th>
                <th className="px-2 py-2 text-left">Action</th>
              </tr>
            </thead>
            <tbody>
              {silos.map((silo, index) => (
                <tr key={`silo-${index}`} className="border-b border-stone-200 dark:border-zinc-800">
                  <td className="px-2 py-2"><input value={silo.silo_id} readOnly className="w-20 rounded border border-stone-300 bg-stone-200 px-2 py-1 dark:border-zinc-600 dark:bg-zinc-700" /></td>
                  <td className="px-2 py-2"><input type="number" value={silo.capacity_kg} onChange={(e) => updateArrayField('silos', index, 'capacity_kg', Number(e.target.value))} className="w-32 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" step="0.01" value={silo.body_diameter_m} onChange={(e) => updateArrayField('silos', index, 'body_diameter_m', Number(e.target.value))} className="w-32 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" step="0.01" value={silo.outlet_diameter_m} onChange={(e) => updateArrayField('silos', index, 'outlet_diameter_m', Number(e.target.value))} className="w-32 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={silo.initial_mass_kg} onChange={(e) => updateArrayField('silos', index, 'initial_mass_kg', Number(e.target.value))} className="w-32 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><button type="button" onClick={() => removeArrayRow('silos', index)} className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 dark:border-red-900 dark:text-red-300">Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Layers</h3>
          <button type="button" onClick={addLayerRow} className="rounded-md bg-amber-800 px-3 py-1 text-sm text-white hover:bg-amber-700">+ Add Row</button>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-stone-300 dark:border-zinc-700">
                <th className="px-2 py-2 text-left">silo_id</th>
                <th className="px-2 py-2 text-left">layer_index</th>
                <th className="px-2 py-2 text-left">lot_id</th>
                <th className="px-2 py-2 text-left">supplier</th>
                <th className="px-2 py-2 text-left">segment_mass_kg</th>
                <th className="px-2 py-2 text-left">Action</th>
              </tr>
            </thead>
            <tbody>
              {(payload.layers || []).map((layer, index) => (
                <tr key={`layer-${index}`} className="border-b border-stone-200 dark:border-zinc-800">
                  <td className="px-2 py-2">
                    <select value={layer.silo_id} onChange={(e) => updateArrayField('layers', index, 'silo_id', e.target.value)} className="w-24 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900">
                      {silos.map((silo) => <option key={silo.silo_id} value={silo.silo_id}>{silo.silo_id}</option>)}
                    </select>
                  </td>
                  <td className="px-2 py-2"><input type="number" value={layer.layer_index} onChange={(e) => updateArrayField('layers', index, 'layer_index', Number(e.target.value))} className="w-24 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input value={layer.lot_id} onChange={(e) => updateArrayField('layers', index, 'lot_id', e.target.value)} className="w-28 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input value={layer.supplier} onChange={(e) => updateArrayField('layers', index, 'supplier', e.target.value)} className="w-32 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={layer.segment_mass_kg} onChange={(e) => updateArrayField('layers', index, 'segment_mass_kg', Number(e.target.value))} className="w-32 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><button type="button" onClick={() => removeArrayRow('layers', index)} className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 dark:border-red-900 dark:text-red-300">Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-lg font-semibold">Suppliers</h3>
          <button type="button" onClick={addSupplierRow} className="rounded-md bg-amber-800 px-3 py-1 text-sm text-white hover:bg-amber-700">+ Add Row</button>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-stone-300 dark:border-zinc-700">
                <th className="px-2 py-2 text-left">supplier</th>
                <th className="px-2 py-2 text-left">moisture_pct</th>
                <th className="px-2 py-2 text-left">fine_extract_db_pct</th>
                <th className="px-2 py-2 text-left">wort_pH</th>
                <th className="px-2 py-2 text-left">diastatic_power_WK</th>
                <th className="px-2 py-2 text-left">total_protein_pct</th>
                <th className="px-2 py-2 text-left">wort_colour_EBC</th>
                <th className="px-2 py-2 text-left">Action</th>
              </tr>
            </thead>
            <tbody>
              {(payload.suppliers || []).map((supplier, index) => (
                <tr key={`supplier-${index}`} className="border-b border-stone-200 dark:border-zinc-800">
                  <td className="px-2 py-2"><input value={supplier.supplier} onChange={(e) => updateArrayField('suppliers', index, 'supplier', e.target.value)} className="w-32 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={supplier.moisture_pct} onChange={(e) => updateArrayField('suppliers', index, 'moisture_pct', Number(e.target.value))} className="w-24 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={supplier.fine_extract_db_pct} onChange={(e) => updateArrayField('suppliers', index, 'fine_extract_db_pct', Number(e.target.value))} className="w-24 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={supplier.wort_pH} onChange={(e) => updateArrayField('suppliers', index, 'wort_pH', Number(e.target.value))} className="w-20 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={supplier.diastatic_power_WK} onChange={(e) => updateArrayField('suppliers', index, 'diastatic_power_WK', Number(e.target.value))} className="w-24 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={supplier.total_protein_pct} onChange={(e) => updateArrayField('suppliers', index, 'total_protein_pct', Number(e.target.value))} className="w-24 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><input type="number" value={supplier.wort_colour_EBC} onChange={(e) => updateArrayField('suppliers', index, 'wort_colour_EBC', Number(e.target.value))} className="w-24 rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" /></td>
                  <td className="px-2 py-2"><button type="button" onClick={() => removeArrayRow('suppliers', index)} className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 dark:border-red-900 dark:text-red-300">Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-lg font-semibold">Target Parameters</h3>
            <button type="button" onClick={() => addObjectEntry('target_params', 'new_param')} className="rounded-md bg-amber-800 px-3 py-1 text-sm text-white hover:bg-amber-700">+ Add Row</button>
          </div>
          <div className="space-y-2">
            {Object.entries(payload.target_params || {}).map(([key, value]) => (
              <div key={key} className="grid grid-cols-[1fr_140px_auto] gap-2">
                <input value={key} onChange={(e) => updateObjectEntry('target_params', key, e.target.value, value)} className="rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
                <input type="number" value={value} onChange={(e) => updateObjectEntry('target_params', key, key, Number(e.target.value))} className="rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
                <button type="button" onClick={() => removeObjectEntry('target_params', key)} className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 dark:border-red-900 dark:text-red-300">Remove</button>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-lg font-semibold">Weights</h3>
            <button type="button" onClick={() => addObjectEntry('weights', 'new_weight')} className="rounded-md bg-amber-800 px-3 py-1 text-sm text-white hover:bg-amber-700">+ Add Row</button>
          </div>
          <div className="space-y-2">
            {Object.entries(payload.weights || {}).map(([key, value]) => (
              <div key={key} className="grid grid-cols-[1fr_140px_auto] gap-2">
                <input value={key} onChange={(e) => updateObjectEntry('weights', key, e.target.value, value)} className="rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
                <input type="number" value={value} onChange={(e) => updateObjectEntry('weights', key, key, Number(e.target.value))} className="rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
                <button type="button" onClick={() => removeObjectEntry('weights', key)} className="rounded border border-red-300 px-2 py-1 text-xs text-red-700 dark:border-red-900 dark:text-red-300">Remove</button>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        <label className="rounded-2xl border border-stone-300 bg-stone-50 p-4 text-sm shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          sigma_m
          <input type="number" value={payload.sigma_m} onChange={(e) => updatePayload('sigma_m', Number(e.target.value))} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
        </label>
        <label className="rounded-2xl border border-stone-300 bg-stone-50 p-4 text-sm shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          steps
          <input type="number" value={payload.steps} onChange={(e) => updatePayload('steps', Number(e.target.value))} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
        </label>
        <label className="rounded-2xl border border-stone-300 bg-stone-50 p-4 text-sm shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          mode
          <select value={payload.mode} onChange={(e) => updatePayload('mode', e.target.value)} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900">
            <option value="A">A</option>
            <option value="B">B</option>
          </select>
        </label>
        <label className="rounded-2xl border border-stone-300 bg-stone-50 p-4 text-sm shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          fixed_total_mass_kg
          <input
            type="number"
            value={payload.fixed_total_mass_kg ?? ''}
            onChange={(e) => updatePayload('fixed_total_mass_kg', e.target.value === '' ? null : Number(e.target.value))}
            className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900"
          />
        </label>
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          <h3 className="mb-3 text-lg font-semibold">Material</h3>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-sm">rho_bulk_kg_m3
              <input type="number" value={payload.material.rho_bulk_kg_m3} onChange={(e) => updatePayload('material', { ...payload.material, rho_bulk_kg_m3: Number(e.target.value) })} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
            </label>
            <label className="text-sm">grain_diameter_m
              <input type="number" value={payload.material.grain_diameter_m} onChange={(e) => updatePayload('material', { ...payload.material, grain_diameter_m: Number(e.target.value) })} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
            </label>
          </div>
        </div>

        <div className="rounded-2xl border border-stone-300 bg-stone-50 p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
          <h3 className="mb-3 text-lg font-semibold">Beverloo</h3>
          <div className="grid grid-cols-3 gap-3">
            <label className="text-sm">C
              <input type="number" value={payload.beverloo.C} onChange={(e) => updatePayload('beverloo', { ...payload.beverloo, C: Number(e.target.value) })} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
            </label>
            <label className="text-sm">k
              <input type="number" value={payload.beverloo.k} onChange={(e) => updatePayload('beverloo', { ...payload.beverloo, k: Number(e.target.value) })} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
            </label>
            <label className="text-sm">g_m_s2
              <input type="number" value={payload.beverloo.g_m_s2} onChange={(e) => updatePayload('beverloo', { ...payload.beverloo, g_m_s2: Number(e.target.value) })} className="mt-1 w-full rounded border border-stone-300 bg-white px-2 py-1 dark:border-zinc-600 dark:bg-zinc-900" />
            </label>
          </div>
        </div>
      </section>
    </div>
  );
}

export default InputFormPage;
