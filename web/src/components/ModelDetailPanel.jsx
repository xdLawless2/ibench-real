import { useMemo, useState, useEffect, useRef } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { getProviderColor } from "../providerColors";
import ProviderLogo from "./ProviderLogo";

function StatCard({ label, value, sub, accent }) {
  return (
    <div
      className="bg-surface-overlay/50 rounded-lg p-4 border-l-2"
      style={{ borderLeftColor: accent || "transparent" }}
    >
      <div className="text-xs font-medium uppercase tracking-wider text-text-muted mb-1">
        {label}
      </div>
      <div className="text-2xl font-bold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-text-muted mt-0.5">{sub}</div>}
    </div>
  );
}

function TokenBreakdown({ tokens, providerColor }) {
  const data = [
    { name: "Prompt", value: tokens.prompt },
    { name: "Reasoning", value: tokens.reasoning },
    { name: "Output", value: tokens.completion - tokens.reasoning },
  ].filter((d) => d.value > 0);

  const total = data.reduce((s, d) => s + d.value, 0);
  if (total === 0) return null;

  const colors = [
    "var(--color-text-muted)",
    providerColor,
    "var(--color-text-secondary)",
  ];

  return (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-3">
        Token Usage
      </h4>
      <div className="flex h-3 rounded-full overflow-hidden gap-px">
        {data.map((d, i) => (
          <div
            key={d.name}
            style={{
              width: `${(d.value / total) * 100}%`,
              background: colors[i],
            }}
            className="first:rounded-l-full last:rounded-r-full"
            title={`${d.name}: ${d.value.toLocaleString()}`}
          />
        ))}
      </div>
      <div className="flex gap-4 mt-2">
        {data.map((d, i) => (
          <div key={d.name} className="flex items-center gap-1.5 text-xs text-text-muted">
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: colors[i] }}
            />
            {d.name}: {(d.value / 1000).toFixed(0)}k
          </div>
        ))}
      </div>
    </div>
  );
}

function LatencyHistogram({ items }) {
  const buckets = useMemo(() => {
    const latencies = items
      .map((it) => it.latency)
      .filter((l) => l != null && l > 0);
    if (latencies.length === 0) return [];

    const max = Math.max(...latencies);
    const numBuckets = Math.min(12, Math.ceil(max));
    const size = max / numBuckets;
    const bins = Array.from({ length: numBuckets }, (_, i) => ({
      label: `${(i * size).toFixed(0)}`,
      count: 0,
    }));
    for (const l of latencies) {
      const idx = Math.min(Math.floor(l / size), numBuckets - 1);
      bins[idx].count++;
    }
    return bins;
  }, [items]);

  if (buckets.length === 0) return null;

  return (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-3">
        Latency Distribution (seconds)
      </h4>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={buckets} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="label"
            tick={{ fontSize: 10, fill: "var(--color-text-muted)" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis hide />
          <Tooltip
            cursor={{ fill: "var(--color-surface-overlay)", opacity: 0.5 }}
            contentStyle={{
              background: "var(--color-surface-raised)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {buckets.map((_, i) => (
              <Cell key={i} fill="var(--color-text-muted)" />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ImageGrid({ items }) {
  const [selected, setSelected] = useState(null);

  return (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-3">
        Per-Image Results
      </h4>
      <div className="grid grid-cols-5 sm:grid-cols-10 gap-1">
        {items.map((item) => (
          <div
            key={item.index}
            onClick={() => setSelected(item)}
            className="rounded cursor-pointer hover:brightness-125 px-1 py-1.5 flex flex-col items-center gap-0.5"
            style={{
              background: item.correct
                ? "color-mix(in srgb, var(--color-correct) 20%, transparent)"
                : "color-mix(in srgb, var(--color-incorrect) 20%, transparent)",
              border: `1px solid ${item.correct ? "var(--color-correct)" : "var(--color-incorrect)"}`,
              borderLeftWidth: 2,
            }}
            title={`#${item.index}: truth=${item.truth}, pred=${item.pred}${item.correct ? " ✓" : " ✗"}`}
          >
            <span className="text-[8px] text-text-muted leading-none">#{item.index}</span>
            <span className="text-[10px] font-bold tabular-nums leading-none" style={{ color: item.correct ? "var(--color-correct)" : "var(--color-incorrect)" }}>
              {item.pred}<span className="text-text-muted font-normal">/{item.truth}</span>
            </span>
          </div>
        ))}
      </div>
      <div className="flex gap-4 mt-2 text-xs text-text-muted">
        <span className="flex items-center gap-1">
          <span className="w-3 h-1.5 rounded-sm bg-correct" /> Correct
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-1.5 rounded-sm bg-incorrect" /> Incorrect
        </span>
        <span className="ml-auto text-text-muted/60">pred/truth — click to view</span>
      </div>

      {selected && (
        <>
          <div
            onClick={() => setSelected(null)}
            className="fixed inset-0 bg-black/70 z-[60]"
          />
          <div
            className="fixed inset-0 z-[70] flex items-center justify-center p-8"
            onClick={() => setSelected(null)}
          >
            <div
              onClick={(e) => e.stopPropagation()}
              className="bg-surface-raised border border-border rounded-2xl overflow-hidden max-w-lg w-full"
            >
              <img
                src={`/imgs/${selected.index}.png`}
                alt={`Image ${selected.index}`}
                className="w-full"
              />
              <div className="p-4 flex items-center justify-between">
                <div>
                  <p className="font-semibold">Image #{selected.index}</p>
                  <p className="text-sm text-text-secondary">
                    Truth: <span className="font-bold text-text-primary tabular-nums">{selected.truth}</span>
                    &nbsp;&middot;&nbsp;
                    Predicted: <span className={`font-bold tabular-nums ${selected.correct ? "text-correct" : "text-incorrect"}`}>{selected.pred}</span>
                  </p>
                </div>
                <span className={`text-xs font-bold px-3 py-1 rounded-full ${selected.correct ? "bg-correct/15 text-correct" : "bg-incorrect/15 text-incorrect"}`}>
                  {selected.correct ? "Correct" : "Wrong"}
                </span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function ModelDetailPanel({ run, onClose }) {
  const [visible, setVisible] = useState(false);
  const [shouldRenderContent, setShouldRenderContent] = useState(false);
  const prevRunRef = useRef(null);
  const isOpen = run != null;

  useEffect(() => {
    if (isOpen) {
      prevRunRef.current = run;
      requestAnimationFrame(() => {
        requestAnimationFrame(() => setVisible(true));
      });
      const timer = setTimeout(() => setShouldRenderContent(true), 300);
      return () => clearTimeout(timer);
    } else {
      setVisible(false);
      setShouldRenderContent(false);
    }
  }, [isOpen, run]);

  const displayRun = run ?? prevRunRef.current;
  if (!displayRun) return null;

  const providerColor = getProviderColor(displayRun.provider);
  const costPerImage = displayRun.cost / (displayRun.items?.length || 100);

  function handleTransitionEnd(e) {
    if (e.target === e.currentTarget && !visible) {
      prevRunRef.current = null;
    }
  }

  const showPanel = isOpen || visible;
  if (!showPanel && !prevRunRef.current) return null;

  return (
    <>
      <div
        onClick={onClose}
        className="fixed inset-0 bg-black/60 z-40"
        style={{
          opacity: visible ? 1 : 0,
          transition: "opacity 0.2s ease-out",
          pointerEvents: visible ? "auto" : "none",
        }}
      />

      <aside
        onTransitionEnd={handleTransitionEnd}
        className="fixed top-0 right-0 h-full w-full max-w-2xl bg-surface-raised border-l border-border z-50 overflow-y-auto scrollbar-thin will-change-transform"
        style={{
          transform: visible ? "translateX(0)" : "translateX(100%)",
          transition: "transform 0.25s ease-out",
        }}
      >
        <div className="h-1 w-full" style={{ background: providerColor }} />
        <div className="sticky top-0 bg-surface-raised border-b border-border-subtle px-8 py-5 flex items-center justify-between z-10">
          <div>
            <div className="flex items-center gap-3">
              <ProviderLogo provider={displayRun.provider} size={24} />
              <h2 className="text-xl font-bold">
                {displayRun.model}
                {!displayRun.reasoning && <span className="text-text-muted ml-0.5">*</span>}
              </h2>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-9 h-9 rounded-lg flex items-center justify-center hover:bg-surface-overlay transition-colors cursor-pointer"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="p-8 space-y-8">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard label="Accuracy" value={`${displayRun.accuracy}%`} sub={`${displayRun.numCorrect}/100 correct`} accent={providerColor} />
            <StatCard label="MAE" value={displayRun.mae.toFixed(2)} accent={providerColor} />
            <StatCard label="Avg Latency" value={`${displayRun.latencyAvg < 60 ? displayRun.latencyAvg.toFixed(1) + "s" : (displayRun.latencyAvg / 60).toFixed(1) + "m"}`} sub={`P95: ${displayRun.latencyP95 < 60 ? displayRun.latencyP95.toFixed(1) + "s" : (displayRun.latencyP95 / 60).toFixed(1) + "m"}`} accent={providerColor} />
            <StatCard label="Total Cost" value={`$${displayRun.cost < 1 ? displayRun.cost.toFixed(2) : displayRun.cost.toFixed(0)}`} sub={`$${costPerImage.toFixed(4)}/img`} accent={providerColor} />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Total Time" value={`${(displayRun.totalTime / 60).toFixed(0)}m`} />
            <StatCard label="Failures" value={displayRun.failures} />
          </div>

          <TokenBreakdown tokens={displayRun.tokens} providerColor={providerColor} />

          {shouldRenderContent ? (
            <LatencyHistogram items={displayRun.items || []} />
          ) : (
            <div className="h-[152px]" />
          )}

          {shouldRenderContent && displayRun.items && displayRun.items.length > 0 && (
            <ImageGrid items={displayRun.items} />
          )}

          <div className="text-xs text-text-muted pt-4 border-t border-border-subtle">
            <p>
              Run: <code className="font-mono text-text-secondary">{displayRun.slug}</code>
            </p>
            <p>Timestamp: {new Date(displayRun.timestamp).toLocaleString()}</p>
            <p>
              Pricing: ${displayRun.price?.inputPerMillion}/M in, ${displayRun.price?.outputPerMillion}/M out
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}
