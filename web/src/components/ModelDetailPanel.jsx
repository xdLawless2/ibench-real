import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
            className="transition-all duration-500 first:rounded-l-full last:rounded-r-full"
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
      <div className="grid grid-cols-10 gap-1">
        {items.map((item) => (
          <div
            key={item.index}
            onClick={() => setSelected(item)}
            className="relative aspect-square rounded-md overflow-hidden border-2 transition-transform hover:scale-110 hover:z-10 cursor-pointer"
            style={{
              borderColor: item.correct
                ? "var(--color-correct)"
                : "var(--color-incorrect)",
            }}
            title={`#${item.index}: truth=${item.truth}, pred=${item.pred}${item.correct ? " ✓" : " ✗"}`}
          >
            <img
              src={`/imgs/${item.index}.png`}
              alt={`Image ${item.index}`}
              className="w-full h-full object-cover"
              loading="lazy"
            />
            <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
              <span className="text-[10px] text-white font-bold tabular-nums">
                {item.pred}/{item.truth}
              </span>
            </div>
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
      </div>

      <AnimatePresence>
        {selected && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelected(null)}
              className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[60]"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ type: "spring", damping: 28, stiffness: 300 }}
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
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function ModelDetailPanel({ run, onClose }) {
  const providerColor = getProviderColor(run.provider);

  const costPerImage = run.cost / (run.items?.length || 100);

  return (
    <>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        onClick={onClose}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
      />

      <motion.aside
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
        className="fixed top-0 right-0 h-full w-full max-w-2xl bg-surface-raised border-l border-border z-50 overflow-y-auto scrollbar-thin"
      >
        <div className="h-1 w-full" style={{ background: providerColor }} />
        <div className="sticky top-0 bg-surface-raised/90 backdrop-blur-xl border-b border-border-subtle px-8 py-5 flex items-center justify-between z-10">
          <div>
            <div className="flex items-center gap-3">
              <ProviderLogo provider={run.provider} size={24} />
              <h2 className="text-xl font-bold">
                {run.model}
                {!run.reasoning && <span className="text-text-muted ml-0.5">*</span>}
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
            <StatCard label="Accuracy" value={`${run.accuracy}%`} sub={`${run.numCorrect}/100 correct`} accent={providerColor} />
            <StatCard label="MAE" value={run.mae.toFixed(2)} accent={providerColor} />
            <StatCard label="Avg Latency" value={`${run.latencyAvg < 60 ? run.latencyAvg.toFixed(1) + "s" : (run.latencyAvg / 60).toFixed(1) + "m"}`} sub={`P95: ${run.latencyP95 < 60 ? run.latencyP95.toFixed(1) + "s" : (run.latencyP95 / 60).toFixed(1) + "m"}`} accent={providerColor} />
            <StatCard label="Total Cost" value={`$${run.cost < 1 ? run.cost.toFixed(2) : run.cost.toFixed(0)}`} sub={`$${costPerImage.toFixed(4)}/img`} accent={providerColor} />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Total Time" value={`${(run.totalTime / 60).toFixed(0)}m`} />
            <StatCard label="Failures" value={run.failures} />
          </div>

          <TokenBreakdown tokens={run.tokens} providerColor={providerColor} />

          <LatencyHistogram items={run.items || []} />

          {run.items && run.items.length > 0 && (
            <ImageGrid items={run.items} />
          )}

          <div className="text-xs text-text-muted pt-4 border-t border-border-subtle">
            <p>
              Run: <code className="font-mono text-text-secondary">{run.slug}</code>
            </p>
            <p>Timestamp: {new Date(run.timestamp).toLocaleString()}</p>
            <p>
              Pricing: ${run.price?.inputPerMillion}/M in, ${run.price?.outputPerMillion}/M out
            </p>
          </div>
        </div>
      </motion.aside>
    </>
  );
}
