import { useMemo, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  XAxis,
  YAxis,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  CartesianGrid,
  Customized,
} from "recharts";
import { getProviderColor, PROVIDER_LABELS } from "../providerColors";
import ProviderLogo from "./ProviderLogo";

const LOGO_MAP = {
  openai: "/logos/openai.svg",
  anthropic: "/logos/anthropic.svg",
  google: "/logos/google.png",
  qwen: "/logos/qwen.png",
  moonshot: "/logos/moonshot.png",
};

const INVERT_ON_DARK = new Set(["openai", "moonshot"]);

function getModelBase(model) {
  const match = model.match(/^(.*?\d+(?:\.\d+)?)/);
  return match ? match[1] : model;
}

function FamilyLines({ points, xAxisMap, yAxisMap }) {
  const xAxis = xAxisMap && Object.values(xAxisMap)[0];
  const yAxis = yAxisMap && Object.values(yAxisMap)[0];
  if (!xAxis?.scale || !yAxis?.scale || !points?.length) return null;

  const groups = {};
  for (const pt of points) {
    const base = getModelBase(pt.model);
    if (!groups[base]) groups[base] = [];
    groups[base].push(pt);
  }

  const lines = [];
  for (const [base, pts] of Object.entries(groups)) {
    if (pts.length < 2) continue;
    const sorted = [...pts].sort((a, b) => a.x - b.x);
    for (let i = 0; i < sorted.length - 1; i++) {
      const x1 = xAxis.scale(sorted[i].x);
      const y1 = yAxis.scale(sorted[i].y);
      const x2 = xAxis.scale(sorted[i + 1].x);
      const y2 = yAxis.scale(sorted[i + 1].y);
      if ([x1, y1, x2, y2].some((v) => v == null || isNaN(v))) continue;
      lines.push(
        <line
          key={`${base}-${i}`}
          x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={getProviderColor(sorted[i].provider)}
          strokeWidth={1.5}
          strokeOpacity={0.35}
          strokeDasharray="4 3"
        />
      );
    }
  }

  return <g>{lines}</g>;
}

function ChartCard({ title, children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-panel p-6"
    >
      <h3 className="text-sm font-semibold uppercase tracking-wider text-text-muted mb-6">
        {title}
      </h3>
      {children}
    </motion.div>
  );
}

function LogoShape({ cx, cy, payload }) {
  const provider = payload?.provider;
  const src = LOGO_MAP[provider];
  const size = 18;
  if (!src) {
    return (
      <circle cx={cx} cy={cy} r={5} fill={getProviderColor(provider)} style={{ cursor: "pointer" }} />
    );
  }
  const needsInvert = INVERT_ON_DARK.has(provider);
  return (
    <image
      href={src}
      x={cx - size / 2}
      y={cy - size / 2}
      width={size}
      height={size}
      className={needsInvert ? "dark-invert" : ""}
      style={{ cursor: "pointer", pointerEvents: "all" }}
    />
  );
}

function FloatingTooltip({ hoveredPoint, mousePos, containerRef }) {
  if (!hoveredPoint || !containerRef?.current) return null;

  const rect = containerRef.current.getBoundingClientRect();
  const left = mousePos.x - rect.left + 16;
  const top = mousePos.y - rect.top - 10;

  return (
    <div
      className="absolute z-50 pointer-events-none"
      style={{ left, top, transform: "translateY(-100%)" }}
    >
      <div className="bg-surface-raised border border-border rounded-xl overflow-hidden min-w-[160px]">
        <div className="h-0.5 w-full" style={{ background: getProviderColor(hoveredPoint.provider) }} />
        <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <ProviderLogo provider={hoveredPoint.provider} size={16} />
          <span className="font-semibold text-sm">
            {hoveredPoint.model}
            {!hoveredPoint.reasoning && (
              <span className="text-text-muted ml-0.5">*</span>
            )}
          </span>
        </div>
        <div className="space-y-1 text-xs text-text-secondary">
          <div className="flex justify-between gap-4">
            <span>Accuracy</span>
            <span className="font-medium text-text-primary">{hoveredPoint.accuracy}%</span>
          </div>
          {hoveredPoint.cost != null && (
            <div className="flex justify-between gap-4">
              <span>Cost</span>
              <span className="font-medium text-text-primary">${hoveredPoint.cost.toFixed(2)}</span>
            </div>
          )}
          {hoveredPoint.totalTime != null && (
            <div className="flex justify-between gap-4">
              <span>Time</span>
              <span className="font-medium text-text-primary">{(hoveredPoint.totalTime / 60).toFixed(0)}m</span>
            </div>
          )}
        </div>
        </div>
      </div>
    </div>
  );
}

function ProviderFilter({ providers, enabled, onToggle }) {
  return (
    <div className="flex flex-wrap gap-3 justify-center mt-5">
      {providers.map((p) => {
        const active = enabled.has(p);
        return (
          <button
            key={p}
            onClick={() => onToggle(p)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border transition-all duration-200 cursor-pointer ${
              active
                ? "border-border-subtle text-text-secondary hover:border-text-muted"
                : "border-border-subtle/50 text-text-muted/40 opacity-40 hover:opacity-60"
            }`}
          >
            <ProviderLogo provider={p} size={14} className={active ? "" : "grayscale"} />
            <span>{PROVIDER_LABELS[p] ?? p}</span>
          </button>
        );
      })}
    </div>
  );
}

function CostVsAccuracyScatter({ runs, enabledProviders }) {
  const [hovered, setHovered] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [containerRef, setContainerRef] = useState({ current: null });

  const refCallback = useCallback((node) => {
    setContainerRef({ current: node });
  }, []);

  const visibleProviders = useMemo(
    () => [...new Set(runs.map((r) => r.provider))].filter((p) => enabledProviders.has(p)),
    [runs, enabledProviders]
  );

  const dataByProvider = useMemo(() => {
    const map = {};
    for (const r of runs) {
      if (r.cost <= 0 || !enabledProviders.has(r.provider)) continue;
      if (!map[r.provider]) map[r.provider] = [];
      map[r.provider].push({
        x: r.cost,
        y: r.accuracy,
        model: r.model,
        provider: r.provider,
        reasoning: r.reasoning,
        cost: r.cost,
        accuracy: r.accuracy,
      });
    }
    return map;
  }, [runs, enabledProviders]);

  return (
    <div ref={refCallback} className="relative" onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}>
      <ResponsiveContainer width="100%" height={560}>
        <ScatterChart margin={{ top: 24, right: 30, bottom: 24, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
          <XAxis
            type="number"
            dataKey="x"
            name="Cost ($)"
            scale="log"
            domain={["auto", "auto"]}
            tick={{ fontSize: 11, fill: "var(--color-text-muted)" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `$${v < 1 ? v.toFixed(2) : v.toFixed(0)}`}
            label={{
              value: "Estimated Cost ($, log scale)",
              position: "bottom",
              offset: -5,
              style: { fontSize: 11, fill: "var(--color-text-muted)" },
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Accuracy (%)"
            domain={[0, 100]}
            tick={{ fontSize: 11, fill: "var(--color-text-muted)" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `${v}%`}
            label={{
              value: "Accuracy (%)",
              angle: -90,
              position: "insideLeft",
              offset: 10,
              style: { fontSize: 11, fill: "var(--color-text-muted)" },
            }}
          />
          <ZAxis range={[60, 60]} />
          <Customized component={(props) => (
            <FamilyLines {...props} points={visibleProviders.flatMap((p) => dataByProvider[p] || [])} />
          )} />
          {visibleProviders.map((p) =>
            dataByProvider[p] ? (
              <Scatter
                key={p}
                name={p}
                data={dataByProvider[p]}
                shape={<LogoShape />}
                onMouseEnter={(entry) => setHovered(entry)}
                onMouseLeave={() => setHovered(null)}
              />
            ) : null
          )}
        </ScatterChart>
      </ResponsiveContainer>
      <FloatingTooltip hoveredPoint={hovered} mousePos={mousePos} containerRef={containerRef} />
    </div>
  );
}

function ReasoningTimeScatter({ runs, enabledProviders }) {
  const [hovered, setHovered] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [containerRef, setContainerRef] = useState({ current: null });

  const refCallback = useCallback((node) => {
    setContainerRef({ current: node });
  }, []);

  const visibleProviders = useMemo(
    () => [...new Set(runs.filter((r) => r.reasoning).map((r) => r.provider))].filter((p) => enabledProviders.has(p)),
    [runs, enabledProviders]
  );

  const dataByProvider = useMemo(() => {
    const map = {};
    for (const r of runs) {
      if (!r.reasoning || r.totalTime <= 0 || !enabledProviders.has(r.provider)) continue;
      if (!map[r.provider]) map[r.provider] = [];
      map[r.provider].push({
        x: r.totalTime / 60,
        y: r.accuracy,
        model: r.model,
        provider: r.provider,
        reasoning: r.reasoning,
        accuracy: r.accuracy,
        totalTime: r.totalTime,
        cost: r.cost,
      });
    }
    return map;
  }, [runs, enabledProviders]);

  return (
    <div ref={refCallback} className="relative" onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}>
      <ResponsiveContainer width="100%" height={560}>
        <ScatterChart margin={{ top: 24, right: 30, bottom: 24, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
          <XAxis
            type="number"
            dataKey="x"
            name="Time (min)"
            scale="log"
            domain={["auto", "auto"]}
            tick={{ fontSize: 11, fill: "var(--color-text-muted)" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `${v < 1 ? v.toFixed(1) : v.toFixed(0)}m`}
            label={{
              value: "Total Run Time (minutes, log scale)",
              position: "bottom",
              offset: -5,
              style: { fontSize: 11, fill: "var(--color-text-muted)" },
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Accuracy (%)"
            domain={[0, 100]}
            tick={{ fontSize: 11, fill: "var(--color-text-muted)" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <ZAxis range={[60, 60]} />
          <Customized component={(props) => (
            <FamilyLines {...props} points={visibleProviders.flatMap((p) => dataByProvider[p] || [])} />
          )} />
          {visibleProviders.map((p) =>
            dataByProvider[p] ? (
              <Scatter
                key={p}
                name={p}
                data={dataByProvider[p]}
                shape={<LogoShape />}
                onMouseEnter={(entry) => setHovered(entry)}
                onMouseLeave={() => setHovered(null)}
              />
            ) : null
          )}
        </ScatterChart>
      </ResponsiveContainer>
      <FloatingTooltip hoveredPoint={hovered} mousePos={mousePos} containerRef={containerRef} />
    </div>
  );
}

export default function Charts({ runs }) {
  const [activeChart, setActiveChart] = useState("cost");
  const charts = [
    { key: "cost", label: "Cost vs Accuracy" },
    { key: "time", label: "Time vs Accuracy" },
  ];

  const allProviders = useMemo(
    () => [...new Set(runs.map((r) => r.provider))],
    [runs]
  );

  const [enabledProviders, setEnabledProviders] = useState(
    () => new Set(allProviders)
  );

  function toggleProvider(p) {
    setEnabledProviders((prev) => {
      const next = new Set(prev);
      if (next.has(p)) {
        if (next.size > 1) next.delete(p);
      } else {
        next.add(p);
      }
      return next;
    });
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <h2 className="text-2xl font-semibold">Efficiency</h2>
        <div className="flex gap-2">
          {charts.map((c) => (
            <button
              key={c.key}
              onClick={() => setActiveChart(c.key)}
              className={`px-4 py-2 rounded-full text-sm font-medium border transition-all duration-200 cursor-pointer ${
                activeChart === c.key
                  ? "bg-text-primary text-surface border-text-primary"
                  : "border-border text-text-secondary hover:border-text-secondary hover:text-text-primary"
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>

      {activeChart === "cost" && (
        <ChartCard title="Cost vs Accuracy">
          <CostVsAccuracyScatter runs={runs} enabledProviders={enabledProviders} />
        </ChartCard>
      )}
      {activeChart === "time" && (
        <ChartCard title="Reasoning Time vs Accuracy">
          <ReasoningTimeScatter runs={runs} enabledProviders={enabledProviders} />
        </ChartCard>
      )}

      <ProviderFilter
        providers={allProviders}
        enabled={enabledProviders}
        onToggle={toggleProvider}
      />
    </div>
  );
}
