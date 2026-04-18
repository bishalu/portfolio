import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ConsoleCartridge } from './ConsoleCartridge'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

interface StudioData {
  id: string
  data: {
    name: string
    tagline: string
    description: string
    status?: string
    accent?: string
    icon?: string
  }
}

interface VibesetConsoleProps {
  studios: StudioData[]
}

export const VibesetConsole: React.FC<VibesetConsoleProps> = ({ studios }) => {
  const [isUnlocked, setIsUnlocked] = useState(false)
  const [isBooting, setIsBooting] = useState(false)
  const [activeStudioId, setActiveStudioId] = useState<string | null>(null)
  const constraintsRef = useRef(null)

  const handleUnlock = () => {
    setIsBooting(true)
    setTimeout(() => {
      setIsUnlocked(true)
      setIsBooting(false)
      // Auto-select first studio after a delay for better UX
      setTimeout(() => setActiveStudioId(studios[0]?.id), 800)
    }, 800)
  }

  return (
    <div
      className="perspective-1000 relative flex min-h-[700px] w-full flex-col items-center justify-center p-4 md:p-8"
      ref={constraintsRef}
    >
      {/* ─────────────────────────────────────────────────────────────
          LOCKED STATE: The "Sleep Mode" Cover
         ───────────────────────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        {!isUnlocked && (
          <motion.div
            key="lock-screen"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{
              y: -100,
              opacity: 0,
              scale: 1.1,
              filter: 'blur(20px)',
              transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] },
            }}
            className="relative z-50 flex aspect-[3/4] w-full max-w-md flex-col justify-between overflow-hidden rounded-[32px] border border-white/10 bg-[#0a0a0a] p-8 shadow-2xl md:aspect-[4/3]"
          >
            {/* Ambient Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#1a1a1a] to-black" />
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(255,255,255,0.03),transparent_70%)]" />

            {/* Status Bar */}
            <div className="relative z-10 flex w-full items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-orange-500 shadow-[0_0_8px_orange]" />
                <span className="font-mono text-[10px] font-bold tracking-widest text-white/40 uppercase">
                  Standby Mode
                </span>
              </div>
              <div className="flex gap-1">
                <div className="h-1.5 w-4 rounded-sm bg-white/20" />
                <div className="h-1.5 w-4 rounded-sm bg-white/20" />
                <div className="h-1.5 w-4 rounded-sm bg-white/60" />
              </div>
            </div>

            {/* Center Visual */}
            <div className="relative z-10 flex flex-1 flex-col items-center justify-center text-center">
              <motion.div
                animate={{ opacity: [0.4, 0.7, 0.4] }}
                transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
                className="relative mb-6"
              >
                <div className="flex h-24 w-24 items-center justify-center rounded-full border border-white/5 bg-gradient-to-tr from-white/5 to-white/10 backdrop-blur-sm">
                  <svg
                    width="40"
                    height="40"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1"
                    className="text-white/30"
                  >
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                  </svg>
                </div>
              </motion.div>

              <h2 className="mb-2 font-[Outfit] text-3xl font-bold tracking-tight text-white">Vibeset</h2>
              <p className="max-w-[200px] text-sm text-gray-500">
                System is sleeping. Slide to initialized studio modules.
              </p>
            </div>

            {/* The Slide "Latch" */}
            <div className="relative z-10 w-full rounded-full border border-white/5 bg-white/5 p-1.5 backdrop-blur-md">
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                <motion.span
                  animate={{ x: [0, 5, 0], opacity: [0.3, 0.6, 0.3] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="font-mono text-[10px] tracking-[0.2em] text-white/30 uppercase"
                >
                  Slide to Power On &gt;&gt;
                </motion.span>
              </div>

              <motion.div
                drag="x"
                dragConstraints={{ left: 0, right: 260 }} // Adjusted for typical width
                dragElastic={0.05}
                dragSnapToOrigin={!isBooting}
                onDragEnd={(e, info) => {
                  // If dragged past threshold
                  if (info.offset.x > 100) {
                    handleUnlock()
                  }
                }}
                animate={isBooting ? { x: 260, scale: 1.1 } : { scale: 1 }}
                className={twMerge(
                  'relative z-20 flex h-12 w-12 cursor-grab items-center justify-center rounded-full shadow-lg transition-colors duration-300 active:cursor-grabbing',
                  isBooting
                    ? 'bg-white text-black shadow-[0_0_30px_rgba(255,255,255,0.5)]'
                    : 'bg-[#1f1f1f] text-white hover:bg-[#2a2a2a]',
                )}
              >
                {isBooting ? (
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    className="animate-spin"
                  >
                    <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm0-4v-4" />
                  </svg>
                ) : (
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M13 5l7 7-7 7M5 12h14" />
                  </svg>
                )}
              </motion.div>
            </div>

            {/* Botttom Branding */}
            <div className="absolute right-0 bottom-2 left-0 pb-3 text-center">
              <p className="font-mono text-[9px] tracking-wider text-white/20">POWERED BY VIBESET.AI</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─────────────────────────────────────────────────────────────
          UNLOCKED STATE: The Cartridge Grid
         ───────────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {isUnlocked && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
            className="relative z-10 w-full max-w-6xl"
          >
            {/* Header Status Bar (In-System) */}
            <div className="mb-8 flex items-center justify-between border-b border-white/5 px-2 pb-4">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="flex items-center gap-3"
              >
                <div className="h-2 w-2 animate-pulse rounded-full bg-[#1AE8CE] shadow-[0_0_8px_#1AE8CE]" />
                <span className="font-mono text-xs font-bold tracking-widest text-[#1AE8CE]">
                  SYSTEM ONLINE // VIBESET.OS
                </span>
              </motion.div>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
                className="font-mono text-[10px] tracking-wider text-gray-500 uppercase"
              >
                Connected via Vibeset.ai
              </motion.div>
            </div>

            {/* The Grid */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 md:gap-6 lg:grid-cols-3">
              {studios.map((studio, idx) => (
                <motion.div
                  key={studio.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + idx * 0.1, duration: 0.5 }}
                >
                  <ConsoleCartridge
                    index={idx}
                    studio={studio}
                    isActive={activeStudioId === studio.id}
                    onClick={() => setActiveStudioId(studio.id === activeStudioId ? null : studio.id)}
                  />
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
