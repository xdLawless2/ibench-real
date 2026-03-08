import { memo } from "react";

export default memo(function Footer() {
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

        </div>
      </div>
    </footer>
  );
});
