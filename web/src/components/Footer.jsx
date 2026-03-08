import { memo } from "react";

export default memo(function Footer() {
  return (
    <footer className="mt-16">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="max-w-lg mx-auto text-center">
          <p className="text-xs text-text-secondary leading-relaxed">
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
