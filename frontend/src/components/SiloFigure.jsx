import { useMemo } from "react";

function colorForIndex(index, total) {
  const goldenAngle = 137.508;
  const hue = Math.round((index * goldenAngle) % 360);
  const lightness = total > 1 ? 52 + ((index % 3) - 1) * 4 : 52;
  return `hsl(${hue} 70% ${lightness}%)`;
}

function SiloTank({ silo, layers, layerColorMap }) {
  const totalLayerMass = layers.reduce(
    (sum, layer) => sum + Number(layer.segment_mass_kg || 0),
    0,
  );
  const shapeTopY = 20;
  const shapeBottomY = 265;
  const shapeHeight = shapeBottomY - shapeTopY;
  const leftX = 18;
  const rightX = 122;
  const apexX = 70;
  const wallBottomY = 230;
  const shapePath = `M${leftX} ${shapeTopY} Q${apexX} -8 ${rightX} ${shapeTopY} L${rightX} ${wallBottomY} L${apexX} ${shapeBottomY} L${leftX} ${wallBottomY} Z`;

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        viewBox="0 0 140 280"
        className="h-64 w-40"
        role="img"
        aria-label={`Silo ${silo.silo_id}`}
      >
        <defs>
          <clipPath id={`silo-shape-${silo.silo_id}`}>
            <path d={shapePath} />
          </clipPath>
        </defs>

        <path
          d={shapePath}
          fill="rgb(245 245 244)"
          className="dark:fill-zinc-900"
        />

        {(() => {
          let cursorY = shapeBottomY;
          return layers.map((layer) => {
            const segmentMass = Number(layer.segment_mass_kg || 0);
            const height =
              totalLayerMass > 0
                ? (segmentMass / totalLayerMass) * shapeHeight
                : 0;
            const renderHeight = Math.max(height, 10);
            const y = cursorY - renderHeight;
            cursorY = y;

            const colorKey = `${layer.silo_id}-${layer.layer_index}-${layer.lot_id}`;
            const colorValue = layerColorMap.get(colorKey) || "#94a3b8";
            return (
              <rect
                key={`${layer.silo_id}-${layer.layer_index}-${layer.lot_id}`}
                x="0"
                y={y}
                width="140"
                height={renderHeight}
                clipPath={`url(#silo-shape-${silo.silo_id})`}
                fill={colorValue}
              >
                <title>{`${layer.lot_id} (${layer.segment_mass_kg} kg)`}</title>
              </rect>
            );
          });
        })()}

        <path
          d={shapePath}
          fill="none"
          stroke="rgb(120 113 108)"
          strokeWidth="6"
        />
      </svg>
      <div className="text-center text-xs">
        <p className="font-semibold text-stone-800 dark:text-stone-100">
          {silo.silo_id}
        </p>
        <p className="text-stone-500 dark:text-stone-400">
          {layers.length} layers
        </p>
      </div>
    </div>
  );
}

function SiloFigure({ silos, layers }) {
  const layerColorMap = useMemo(() => {
    const map = new Map();
    layers.forEach((layer, index) => {
      const key = `${layer.silo_id}-${layer.layer_index}-${layer.lot_id}`;
      if (!map.has(key)) {
        map.set(key, colorForIndex(index, layers.length || 1));
      }
    });
    return map;
  }, [layers]);

  const layersBySilo = useMemo(() => {
    const groups = new Map();
    silos.forEach((silo) => groups.set(silo.silo_id, []));
    layers.forEach((layer) => {
      const key = layer.silo_id;
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key).push(layer);
    });
    groups.forEach((groupLayers) => {
      groupLayers.sort(
        (left, right) => Number(left.layer_index) - Number(right.layer_index),
      );
    });
    return groups;
  }, [layers, silos]);

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-stone-900 dark:text-stone-100">
          Silo Figure
        </h3>
        <p className="text-sm text-stone-500 dark:text-stone-400">
          {silos.length} silos
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {silos.map((silo) => (
          <SiloTank
            key={silo.silo_id}
            silo={silo}
            layers={layersBySilo.get(silo.silo_id) || []}
            layerColorMap={layerColorMap}
          />
        ))}
      </div>
    </div>
  );
}

export default SiloFigure;
