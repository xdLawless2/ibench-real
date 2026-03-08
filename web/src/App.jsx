import { useState, useMemo } from "react";
import { AnimatePresence } from "framer-motion";
import benchmarkData from "./data/benchmark-data.json";
import { useTheme } from "./hooks/useTheme";
import Header from "./components/Header";
import ThemeToggle from "./components/ThemeToggle";
import Leaderboard from "./components/Leaderboard";
import ModelDetailPanel from "./components/ModelDetailPanel";
import Charts from "./components/Charts";
import ImageViewer from "./components/ImageViewer";
import Footer from "./components/Footer";

const SECTIONS = ["leaderboard", "efficiency", "images"];

export default function App() {
  const { theme, toggle } = useTheme();
  const [selectedModel, setSelectedModel] = useState(null);
  const [activeSection, setActiveSection] = useState("leaderboard");

  const { meta, runs, truth } = benchmarkData;

  const selectedRun = useMemo(
    () => runs.find((r) => r.slug === selectedModel) ?? null,
    [runs, selectedModel]
  );

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
          <Leaderboard
            runs={runs}
            onSelectModel={(slug) => setSelectedModel(slug)}
          />
        )}
        {activeSection === "efficiency" && <Charts runs={runs} theme={theme} />}
        {activeSection === "images" && (
          <ImageViewer runs={runs} truth={truth} />
        )}
      </main>

      <AnimatePresence>
        {selectedRun && (
          <ModelDetailPanel
            run={selectedRun}
            onClose={() => setSelectedModel(null)}
          />
        )}
      </AnimatePresence>

      <Footer />
    </div>
  );
}
