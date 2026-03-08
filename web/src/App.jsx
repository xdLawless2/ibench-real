import { useState, useMemo, useCallback, useEffect, lazy, Suspense } from "react";
import { useTheme } from "./hooks/useTheme";
import Header from "./components/Header";
import ThemeToggle from "./components/ThemeToggle";
import Leaderboard from "./components/Leaderboard";
import ModelDetailPanel from "./components/ModelDetailPanel";
import Footer from "./components/Footer";

const Charts = lazy(() => import("./components/Charts"));
const ImageViewer = lazy(() => import("./components/ImageViewer"));

const SECTIONS = ["leaderboard", "charts", "images", "methodology"];

function SectionSpinner() {
  return (
    <div className="flex items-center justify-center py-32">
      <div className="w-6 h-6 border-2 border-text-muted border-t-text-primary rounded-full animate-spin" />
    </div>
  );
}

export default function App() {
  const { theme, toggle } = useTheme();
  const [selectedModel, setSelectedModel] = useState(null);
  const [activeSection, setActiveSection] = useState("leaderboard");
  const [data, setData] = useState(null);

  useEffect(() => {
    import("./data/benchmark-data.json").then((mod) => setData(mod.default));
  }, []);

  const handleSelectModel = useCallback((slug) => setSelectedModel(slug), []);
  const handleCloseDetail = useCallback(() => setSelectedModel(null), []);

  const meta = data?.meta;
  const runs = data?.runs;
  const truth = data?.truth;

  const selectedRun = useMemo(
    () => runs?.find((r) => r.slug === selectedModel) ?? null,
    [runs, selectedModel]
  );

  if (!data) {
    return (
      <div className="min-h-screen bg-surface text-text-primary flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-8 h-8 border-2 border-text-muted border-t-text-primary rounded-full animate-spin mx-auto" />
          <p className="text-sm text-text-muted">Loading benchmark data…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface text-text-primary">
      <ThemeToggle theme={theme} onToggle={toggle} />

      <Header meta={meta} />

      <nav className="sticky top-0 z-30 bg-surface/80 backdrop-blur-xl border-b border-border-subtle">
        <div className="max-w-7xl mx-auto px-6 flex gap-1 py-2">
          {SECTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setActiveSection(s)}
              className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 capitalize ${
                activeSection === s
                  ? "bg-text-primary text-surface"
                  : "text-text-secondary hover:text-text-primary hover:bg-surface-overlay"
              }`}
            >
              {s === "images" ? "Image Viewer" : s === "charts" ? "Charts" : s === "methodology" ? "Methodology" : s}
            </button>
          ))}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-10">
        {activeSection === "leaderboard" && (
          <Leaderboard runs={runs} onSelectModel={handleSelectModel} />
        )}
        {activeSection === "charts" && (
          <Suspense fallback={<SectionSpinner />}>
            <Charts runs={runs} theme={theme} />
          </Suspense>
        )}
        {activeSection === "images" && (
          <Suspense fallback={<SectionSpinner />}>
            <ImageViewer runs={runs} truth={truth} />
          </Suspense>
        )}
        {activeSection === "methodology" && (
          <div>
            <h2 className="text-2xl font-semibold mb-8">Methodology</h2>
            <div className="glass-panel p-8 space-y-8 max-w-3xl">
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider text-text-muted mb-3">Task</h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  Each model receives 100 synthetic images containing randomly placed geometric shapes — circles, triangles, and parallelograms. The task is to count the <strong className="text-text-primary">exact number of distinct intersection points</strong> between different shapes in each image.
                </p>
              </div>

              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider text-text-muted mb-3">Dataset Generation</h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  Images are generated programmatically with controlled parameters: shape count, size ranges, spacing constraints, and angle thresholds. Ground-truth intersection counts are computed analytically using segment–segment, segment–circle, and circle–circle intersection math.
                </p>
              </div>

              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider text-text-muted mb-3">Evaluation</h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  All models are queried through the <a href="https://openrouter.ai" target="_blank" rel="noreferrer" className="text-text-primary border-b border-border hover:border-text-primary transition-colors">OpenRouter</a> API with identical prompts and image inputs. Each model sees the same 100 images in the same order. The parser looks for structured answers first (plain numbers, JSON), then scans for "final answer" / "total" patterns near the end of the response, falling back to the last integer found.
                </p>
              </div>

              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider text-text-muted mb-3">Metrics</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="bg-surface-overlay/50 rounded-lg p-4">
                    <p className="text-sm font-medium mb-1">Accuracy</p>
                    <p className="text-xs text-text-muted">Percentage of images where the model's prediction exactly matches the ground truth.</p>
                  </div>
                  <div className="bg-surface-overlay/50 rounded-lg p-4">
                    <p className="text-sm font-medium mb-1">MAE</p>
                    <p className="text-xs text-text-muted">Mean Absolute Error — average distance between prediction and truth across all images.</p>
                  </div>
                  <div className="bg-surface-overlay/50 rounded-lg p-4">
                    <p className="text-sm font-medium mb-1">Latency</p>
                    <p className="text-xs text-text-muted">Average and P95 response time per image, measured end-to-end including network.</p>
                  </div>
                  <div className="bg-surface-overlay/50 rounded-lg p-4">
                    <p className="text-sm font-medium mb-1">Cost</p>
                    <p className="text-xs text-text-muted">Estimated total cost based on token usage and per-model pricing at time of testing.</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold uppercase tracking-wider text-text-muted mb-3">Reasoning Modes</h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  Models marked without an asterisk (*) were tested with reasoning enabled (effort level: medium). Models with * were tested in base/non-reasoning mode.                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      <ModelDetailPanel run={selectedRun} onClose={handleCloseDetail} />

      <Footer />
    </div>
  );
}
