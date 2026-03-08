import { useState, useMemo, memo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ProviderLogo from "./ProviderLogo";
import { getProviderColor } from "../providerColors";

function ImageModal({ imageIndex, truth, runs, onClose }) {
  const predictions = useMemo(() => {
    return runs
      .map((r) => {
        const item = r.items?.find((it) => it.index === imageIndex);
        if (!item) return null;
        return {
          model: r.model,
          reasoning: r.reasoning,
          provider: r.provider,
          pred: item.pred,
          correct: item.correct,
          latency: item.latency,
        };
      })
      .filter(Boolean)
      .sort((a, b) => {
        if (a.correct !== b.correct) return a.correct ? -1 : 1;
        return Math.abs(a.pred - truth) - Math.abs(b.pred - truth);
      });
  }, [imageIndex, runs, truth]);

  const correctCount = predictions.filter((p) => p.correct).length;

  return (
    <>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40"
      />
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
        className="fixed inset-4 sm:inset-10 md:inset-16 z-50 bg-surface-raised border border-border rounded-2xl overflow-hidden flex flex-col md:flex-row"
      >
        <div className="md:w-1/2 bg-surface flex items-center justify-center p-8 border-b md:border-b-0 md:border-r border-border-subtle">
          <div className="text-center">
            <img
              src={`/imgs/${imageIndex}.png`}
              alt={`Image ${imageIndex}`}
              className="max-w-full max-h-[50vh] rounded-lg"
            />
            <div className="mt-4 space-y-1">
              <p className="text-lg font-bold">Image #{imageIndex}</p>
              <p className="text-sm text-text-secondary">
                Ground truth:{" "}
                <span className="font-bold text-text-primary tabular-nums">
                  {truth} intersections
                </span>
              </p>
              <p className="text-xs text-text-muted">
                {correctCount}/{predictions.length} models correct
              </p>
            </div>
          </div>
        </div>

        <div className="md:w-1/2 overflow-y-auto scrollbar-thin">
          <div className="sticky top-0 bg-surface-raised/90 backdrop-blur-xl border-b border-border-subtle px-6 py-4 flex items-center justify-between z-10">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-text-muted">
              Model Predictions
            </h3>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-surface-overlay transition-colors cursor-pointer"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>

          <div className="divide-y divide-border-subtle">
            {predictions.map((p, i) => (
              <div
                key={i}
                className="px-6 py-3 flex items-center gap-3 hover:bg-surface-overlay/30 transition-colors border-l-2"
                style={{ borderLeftColor: getProviderColor(p.provider) }}
              >
                <ProviderLogo provider={p.provider} size={16} />
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium truncate block">
                    {p.model}
                    {!p.reasoning && <span className="text-text-muted ml-0.5">*</span>}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-sm tabular-nums">
                  <span
                    className={`font-bold ${p.correct ? "text-correct" : "text-incorrect"}`}
                  >
                    {p.pred}
                  </span>
                  <span
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                      p.correct
                        ? "bg-correct/20 text-correct"
                        : "bg-incorrect/20 text-incorrect"
                    }`}
                  >
                    {p.correct ? "✓" : "✗"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </>
  );
}

const SORT_OPTIONS = [
  { key: "default", label: "#1 — #100" },
  { key: "hardest", label: "Least Solved" },
  { key: "easiest", label: "Most Solved" },
];

export default memo(function ImageViewer({ runs, truth }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [sortBy, setSortBy] = useState("default");

  const imageStats = useMemo(() => {
    return truth.map((t, i) => {
      const idx = i + 1;
      let correct = 0;
      let total = 0;
      for (const r of runs) {
        const item = r.items?.find((it) => it.index === idx);
        if (item) {
          total++;
          if (item.correct) correct++;
        }
      }
      return { index: idx, truth: t, correct, total, ratio: total > 0 ? correct / total : 0 };
    });
  }, [runs, truth]);

  const sortedImages = useMemo(() => {
    const list = [...imageStats];
    if (sortBy === "hardest") list.sort((a, b) => a.ratio - b.ratio || a.index - b.index);
    else if (sortBy === "easiest") list.sort((a, b) => b.ratio - a.ratio || a.index - b.index);
    return list;
  }, [imageStats, sortBy]);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <h2 className="text-2xl font-semibold">Image Viewer</h2>
        <div className="flex gap-2">
          {SORT_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              onClick={() => setSortBy(opt.key)}
              className={`px-4 py-2 rounded-full text-sm font-medium border transition-all duration-200 cursor-pointer ${
                sortBy === opt.key
                  ? "bg-text-primary text-surface border-text-primary"
                  : "border-border text-text-secondary hover:border-text-secondary hover:text-text-primary"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-5 sm:grid-cols-8 md:grid-cols-10 gap-2">
        {sortedImages.map((img) => {
          const hue = img.ratio * 120;
          return (
            <button
              key={img.index}
              onClick={() => setSelectedImage(img.index)}
              className="relative aspect-square rounded-lg overflow-hidden border-2 cursor-pointer group img-tile"
              style={{
                borderColor: `hsl(${hue}, 60%, 50%)`,
              }}
            >
              <img
                src={`/imgs/${img.index}.png`}
                alt={`Image ${img.index}`}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-white text-xs font-bold">#{img.index}</span>
                <span className="text-white/80 text-[10px] tabular-nums">
                  {img.correct}/{img.total}
                </span>
              </div>
              <div
                className="absolute bottom-0 left-0 right-0 h-1"
                style={{
                  background: `hsl(${hue}, 60%, 50%)`,
                }}
              />
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-6 mt-4 text-xs text-text-muted justify-center">
        <span className="flex items-center gap-1.5">
          <span className="w-6 h-1.5 rounded-sm" style={{ background: "hsl(0, 60%, 50%)" }} />
          Few models correct
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-6 h-1.5 rounded-sm" style={{ background: "hsl(60, 60%, 50%)" }} />
          Some correct
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-6 h-1.5 rounded-sm" style={{ background: "hsl(120, 60%, 50%)" }} />
          Most correct
        </span>
      </div>

      <AnimatePresence>
        {selectedImage && (
          <ImageModal
            imageIndex={selectedImage}
            truth={truth[selectedImage - 1]}
            runs={runs}
            onClose={() => setSelectedImage(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
});
