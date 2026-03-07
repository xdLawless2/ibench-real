export const PROVIDER_COLORS = {
  openai: "var(--color-openai)",
  anthropic: "var(--color-anthropic)",
  google: "var(--color-google)",
  qwen: "var(--color-qwen)",
  moonshot: "var(--color-moonshot)",
  xai: "var(--color-xai)",
  opensource: "var(--color-opensource)",
};

export const PROVIDER_CLASS = {
  openai: "bg-openai",
  anthropic: "bg-anthropic",
  google: "bg-google",
  qwen: "bg-qwen",
  moonshot: "bg-moonshot",
  xai: "bg-xai",
  opensource: "bg-opensource",
};

export const PROVIDER_LABELS = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  google: "Google",
  qwen: "Qwen",
  moonshot: "Moonshot",
  xai: "xAI",
  opensource: "Open Source",
};

export function getProviderColor(provider) {
  return PROVIDER_COLORS[provider] ?? PROVIDER_COLORS.opensource;
}

export function getProviderDotClass(provider) {
  return PROVIDER_CLASS[provider] ?? PROVIDER_CLASS.opensource;
}
