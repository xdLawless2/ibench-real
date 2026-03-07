import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import ProviderLogo from "./ProviderLogo";
import { getProviderColor } from "../providerColors";

const FILTERS = [
  { key: "all", label: "All Models" },
  { key: "base", label: "Base Only" },
  { key: "reasoning", label: "With Reasoning" },
];

const SORT_KEYS = [
  { key: "accuracy", label: "Accuracy", desc: true },
  { key: "mae", label: "MAE", desc: false },
  { key: "latencyAvg", label: "Avg Latency", desc: false },
  { key: "cost", label: "Est. Cost", desc: false },
];

function formatLatency(s) {
  if (s == null) return "—";
  if (s < 10) return s.toFixed(2) + "s";
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem.toFixed(0)}s`;
}

function formatCost(c) {
  if (c == null) return "—";
  if (c < 0.01) return "<$0.01";
  if (c < 1) return "$" + c.toFixed(2);
  if (c < 100) return "$" + c.toFixed(2);
  return "$" + c.toFixed(0);
}

export default function Leaderboard({ runs, onSelectModel }) {
  const [filter, setFilter] = useState("all");
  const [sortKey, setSortKey] = useState("accuracy");
  const [sortAsc, setSortAsc] = useState(false);

  const filtered = useMemo(() => {
    let data = [...runs];
    if (filter === "base") data = data.filter((r) => !r.reasoning);
    else if (filter === "reasoning") data = data.filter((r) => r.reasoning);
    return data;
  }, [runs, filter]);

  const sorted = useMemo(() => {
    const list = [...filtered];
    list.sort((a, b) => {
      const aVal = a[sortKey] ?? Infinity;
      const bVal = b[sortKey] ?? Infinity;
      const cmp = aVal - bVal;
      if (Math.abs(cmp) < 1e-9) {
        const tieA = a.accuracy ?? 0;
        const tieB = b.accuracy ?? 0;
        return tieB - tieA;
      }
      return sortAsc ? cmp : -cmp;
    });
    return list;
  }, [filtered, sortKey, sortAsc]);

  function handleSort(key) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key !== "accuracy");
    }
  }

  function getRank(index) {
    if (index === 0) return { rank: 1, cls: "text-amber-400" };
    const prev = sorted[index - 1];
    const curr = sorted[index];
    const same = Math.abs((prev[sortKey] ?? 0) - (curr[sortKey] ?? 0)) < 1e-9;
    const prevRank = index;
    const rank = same ? prevRank : index + 1;
    if (rank === 1) return { rank, cls: "text-amber-400" };
    if (rank === 2) return { rank, cls: "text-sky-300" };
    if (rank === 3) return { rank, cls: "text-orange-400" };
    return { rank, cls: "text-text-muted" };
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <h2 className="text-2xl font-semibold">Leaderboard</h2>
        <div className="flex gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`px-4 py-2 rounded-full text-sm font-medium border transition-all duration-200 cursor-pointer ${
                filter === f.key
                  ? "bg-text-primary text-surface border-text-primary"
                  : "border-border text-text-secondary hover:border-text-secondary hover:text-text-primary"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-panel overflow-hidden">
        <div className="overflow-x-auto scrollbar-thin">
          <table className="w-full">
            <thead>
              <tr className="bg-surface-overlay/50">
                <th className="text-left px-5 py-4 text-xs font-semibold uppercase tracking-wider text-text-muted w-14">
                  #
                </th>
                <th className="text-left px-5 py-4 text-xs font-semibold uppercase tracking-wider text-text-muted">
                  Model
                </th>
                {SORT_KEYS.map((sk) => (
                  <th
                    key={sk.key}
                    onClick={() => handleSort(sk.key)}
                    className="text-left px-5 py-4 text-xs font-semibold uppercase tracking-wider text-text-muted cursor-pointer select-none hover:text-text-primary transition-colors whitespace-nowrap"
                  >
                    {sk.label}
                    {sortKey === sk.key && (
                      <span className="ml-1">{sortAsc ? "↑" : "↓"}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((run, i) => {
                const { rank, cls } = getRank(i);
                return (
                  <motion.tr
                    key={run.slug}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, delay: Math.min(i * 0.02, 0.6) }}
                    onClick={() => onSelectModel(run.slug)}
                    className="border-t border-border-subtle hover:bg-surface-overlay/40 cursor-pointer transition-colors group leaderboard-row"
                    style={{ "--row-accent": getProviderColor(run.provider) }}
                  >
                    <td className={`px-5 py-4 font-semibold text-sm tabular-nums ${cls}`}>
                      {rank}
                    </td>
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-3">
                        <ProviderLogo provider={run.provider} size={18} />
                        <span className="font-medium text-sm transition-colors model-name">
                          {run.model}
                          {!run.reasoning && (
                            <span className="text-text-muted ml-0.5" title="No reasoning">*</span>
                          )}
                        </span>
                      </div>
                    </td>
                    <td className="px-5 py-4">
                      <div className="flex items-center gap-3 min-w-[140px]">
                        <span className="font-semibold text-sm tabular-nums w-12">
                          {run.accuracy}%
                        </span>
                        <div className="flex-1 h-1.5 bg-border-subtle rounded-full overflow-hidden">
                          <motion.div
                            className="h-full rounded-full"
                            style={{ background: getProviderColor(run.provider) }}
                            initial={{ width: 0 }}
                            animate={{ width: `${run.accuracy}%` }}
                            transition={{ duration: 0.6, delay: Math.min(i * 0.02, 0.6) + 0.2, ease: "easeOut" }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-5 py-4 text-sm tabular-nums text-text-secondary">
                      {run.mae.toFixed(2)}
                    </td>
                    <td className="px-5 py-4 text-sm tabular-nums text-text-secondary whitespace-nowrap">
                      {formatLatency(run.latencyAvg)}
                    </td>
                    <td className="px-5 py-4 text-sm tabular-nums text-text-secondary">
                      {formatCost(run.cost)}
                    </td>
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-text-muted mt-4 text-center">
        Click any row to view detailed results &middot; <span className="text-text-muted">* = no reasoning</span>
      </p>
    </div>
  );
}
