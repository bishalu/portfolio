import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ConsoleCartridge } from './ConsoleCartridge'
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

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
        <div className="relative w-full min-h-[700px] flex flex-col items-center justify-center p-4 md:p-8 perspective-1000" ref={constraintsRef}>

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
                            transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] }
                        }}
                        className="relative z-50 w-full max-w-md aspect-[3/4] md:aspect-[4/3] bg-[#0a0a0a] rounded-[32px] border border-white/10 shadow-2xl overflow-hidden flex flex-col justify-between p-8"
                    >
                        {/* Ambient Background */}
                        <div className="absolute inset-0 bg-gradient-to-br from-[#1a1a1a] to-black" />
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(255,255,255,0.03),transparent_70%)]" />

                        {/* Status Bar */}
                        <div className="relative z-10 flex justify-between items-center w-full">
                            <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 bg-orange-500 rounded-full animate-pulse shadow-[0_0_8px_orange]" />
                                <span className="text-[10px] font-mono font-bold tracking-widest text-white/40 uppercase">Standby Mode</span>
                            </div>
                            <div className="flex gap-1">
                                <div className="w-4 h-1.5 bg-white/20 rounded-sm" />
                                <div className="w-4 h-1.5 bg-white/20 rounded-sm" />
                                <div className="w-4 h-1.5 bg-white/60 rounded-sm" />
                            </div>
                        </div>

                        {/* Center Visual */}
                        <div className="relative z-10 flex-1 flex flex-col items-center justify-center text-center">
                            <motion.div
                                animate={{ opacity: [0.4, 0.7, 0.4] }}
                                transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
                                className="mb-6 relative"
                            >
                                <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-white/5 to-white/10 border border-white/5 backdrop-blur-sm flex items-center justify-center">
                                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-white/30">
                                        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                                    </svg>
                                </div>
                            </motion.div>

                            <h2 className="text-3xl font-bold text-white font-[Outfit] tracking-tight mb-2">Vibeset</h2>
                            <p className="text-sm text-gray-500 max-w-[200px]">System is sleeping. Slide to initialized studio modules.</p>
                        </div>

                        {/* The Slide "Latch" */}
                        <div className="relative z-10 w-full bg-white/5 rounded-full p-1.5 backdrop-blur-md border border-white/5">
                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                <motion.span
                                    animate={{ x: [0, 5, 0], opacity: [0.3, 0.6, 0.3] }}
                                    transition={{ repeat: Infinity, duration: 2 }}
                                    className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase"
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
                                    "relative z-20 w-12 h-12 rounded-full shadow-lg flex items-center justify-center cursor-grab active:cursor-grabbing transition-colors duration-300",
                                    isBooting ? "bg-white text-black shadow-[0_0_30px_rgba(255,255,255,0.5)]" : "bg-[#1f1f1f] text-white hover:bg-[#2a2a2a]"
                                )}
                            >
                                {isBooting ? (
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="animate-spin">
                                        <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm0-4v-4" />
                                    </svg>
                                ) : (
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M13 5l7 7-7 7M5 12h14" />
                                    </svg>
                                )}
                            </motion.div>
                        </div>

                        {/* Botttom Branding */}
                        <div className="absolute bottom-2 left-0 right-0 text-center pb-3">
                            <p className="text-[9px] text-white/20 font-mono tracking-wider">POWERED BY VIBESET.AI</p>
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
                        transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 }}
                        className="w-full max-w-6xl relative z-10"
                    >
                        {/* Header Status Bar (In-System) */}
                        <div className="flex items-center justify-between mb-8 border-b border-white/5 pb-4 px-2">
                            <motion.div
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.4 }}
                                className="flex items-center gap-3"
                            >
                                <div className="w-2 h-2 bg-[#1AE8CE] rounded-full animate-pulse shadow-[0_0_8px_#1AE8CE]" />
                                <span className="text-xs font-mono text-[#1AE8CE] tracking-widest font-bold">SYSTEM ONLINE // VIBESET.OS</span>
                            </motion.div>
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: 0.6 }}
                                className="text-[10px] font-mono text-gray-500 uppercase tracking-wider"
                            >
                                Connected via Vibeset.ai
                            </motion.div>
                        </div>

                        {/* The Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
                            {studios.map((studio, idx) => (
                                <motion.div
                                    key={studio.id}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.2 + (idx * 0.1), duration: 0.5 }}
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
