export default function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      onClick={onToggle}
      aria-label="Toggle theme"
      className="fixed top-5 right-5 z-50 w-11 h-11 rounded-full border border-border bg-surface-raised flex items-center justify-center transition-all duration-200 hover:border-text-secondary hover:scale-105 active:scale-95 cursor-pointer"
    >
      {theme === "dark" ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="text-text-primary">
          <circle cx="12" cy="12" r="5" stroke="currentColor" strokeWidth="2" />
          {[
            [12, 1, 12, 3], [12, 21, 12, 23],
            [4.22, 4.22, 5.64, 5.64], [18.36, 18.36, 19.78, 19.78],
            [1, 12, 3, 12], [21, 12, 23, 12],
            [4.22, 19.78, 5.64, 18.36], [18.36, 5.64, 19.78, 4.22],
          ].map(([x1, y1, x2, y2], i) => (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          ))}
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" className="text-text-primary">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}
