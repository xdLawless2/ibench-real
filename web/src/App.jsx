import { useState, useMemo, useCallback, useEffect, lazy, Suspense } from "react";
import { AnimatePresence } from "framer-motion";
import { useTheme } from "./hooks/useTheme";
import Header from "./components/Header";
import ThemeToggle from "./components/ThemeToggle";
import Leaderboard from "./components/Leaderboard";
import ModelDetailPanel from "./components/ModelDetailPanel";
import Footer from "./components/Footer";

const Charts = lazy(() => import("./components/Charts"));
const ImageViewer = lazy(() => import("./components/ImageViewer"));

const SECTIONS = ["leaderboard", "efficiency", "images"];

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
              {s === "images" ? "Image Viewer" : s === "efficiency" ? "Efficiency" : s}
            </button>
          ))}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-10">
        {activeSection === "leaderboard" && (
          <Leaderboard runs={runs} onSelectModel={handleSelectModel} />
        )}
        {activeSection === "efficiency" && (
          <Suspense fallback={<SectionSpinner />}>
            <Charts runs={runs} theme={theme} />
          </Suspense>
        )}
        {activeSection === "images" && (
          <Suspense fallback={<SectionSpinner />}>
            <ImageViewer runs={runs} truth={truth} />
          </Suspense>
        )}
      </main>

      <AnimatePresence>
        {selectedRun && (
          <ModelDetailPanel run={selectedRun} onClose={handleCloseDetail} />
        )}
      </AnimatePresence>

      <Footer />
    </div>
  );
}
