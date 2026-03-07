import { motion } from "framer-motion";

const statVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.3 + i * 0.1, duration: 0.5, ease: "easeOut" },
  }),
};

export default function Header({ meta }) {
  const stats = [
    { value: meta.totalModels, label: "Models Tested" },
    { value: meta.totalImages, label: "Test Images" },
    { value: `${meta.bestAccuracy}%`, label: "Best Accuracy" },
  ];

  return (
    <header className="relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-openai/5 rounded-full blur-[120px]" />
      </div>

      <div className="relative max-w-7xl mx-auto px-6 pt-24 pb-16 text-center">
        <motion.p
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-xs font-medium tracking-[3px] uppercase text-text-muted mb-5"
        >
          Vision AI Benchmark
        </motion.p>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-6xl sm:text-7xl font-bold tracking-tight mb-5"
        >
          EyeBench v2
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-lg text-text-secondary max-w-xl mx-auto text-balance"
        >
          Evaluating vision models on counting shape intersections in synthetic
          images.
        </motion.p>

        <div className="flex justify-center gap-12 sm:gap-20 mt-14">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              custom={i}
              initial="hidden"
              animate="visible"
              variants={statVariants}
              className="text-center"
            >
              <div className="text-4xl sm:text-5xl font-bold tabular-nums">
                {s.value}
              </div>
              <div className="text-xs font-medium tracking-[2px] uppercase text-text-muted mt-2">
                {s.label}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="border-b border-border-subtle" />
    </header>
  );
}
