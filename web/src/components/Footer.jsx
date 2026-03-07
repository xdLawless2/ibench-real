export default function Footer() {
  return (
    <footer className="border-t border-border-subtle mt-20">
      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="max-w-lg mx-auto text-center">
          <p className="text-xs font-medium tracking-[3px] uppercase text-text-muted mb-6">
            About
          </p>
          <p className="text-sm text-text-secondary leading-relaxed">
            EyeBench v2 was created by <strong className="text-text-primary">Aditya Singh</strong>.
            All models tested via{" "}
            <a
              href="https://openrouter.ai"
              target="_blank"
              rel="noreferrer"
              className="text-text-primary border-b border-border hover:border-text-primary transition-colors"
            >
              OpenRouter
            </a>
            . Pricing estimates based on token usage at time of testing.
          </p>

          <div className="mt-10 p-6 bg-surface-raised rounded-xl border border-border-subtle text-left">
            <h3 className="text-sm font-semibold mb-3">Methodology</h3>
            <p className="text-sm text-text-secondary leading-relaxed">
              Each model receives 100 synthetic images containing randomly placed
              shapes (circles, triangles, parallelograms, lines). The task: count
              the exact number of intersection points between different shapes.
              Models are queried via the OpenRouter API with identical prompts.
              Accuracy measures exact match rate; MAE shows average prediction
              deviation.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
