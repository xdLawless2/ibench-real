const LOGO_MAP = {
  openai: { src: "/logos/openai.svg", invertOnDark: true },
  anthropic: { src: "/logos/anthropic.svg", invertOnDark: false },
  google: { src: "/logos/google.png", invertOnDark: false },
  qwen: { src: "/logos/qwen.png", invertOnDark: false },
  moonshot: { src: "/logos/moonshot.png", invertOnDark: true },
};

export default function ProviderLogo({ provider, size = 18, className = "" }) {
  const logo = LOGO_MAP[provider];

  if (!logo) {
    return (
      <span
        className={`inline-block rounded-full bg-text-muted flex-shrink-0 ${className}`}
        style={{ width: size, height: size }}
      />
    );
  }

  return (
    <img
      src={logo.src}
      alt={provider}
      width={size}
      height={size}
      className={`inline-block flex-shrink-0 object-contain ${logo.invertOnDark ? "dark-invert" : ""} ${className}`}
      draggable={false}
    />
  );
}
