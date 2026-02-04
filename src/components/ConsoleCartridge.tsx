import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

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

interface CartridgeProps {
    studio: StudioData
    isActive: boolean
    onClick: () => void
    index: number
}

// Map accents to neon variables for vivid cyber aesthetics
const ACCENT_MAP: Record<string, string> = {
    'living-teal': '#1AE8CE', // Cyan
    'moss-green': '#99FE00', // Neon Green
    'google-blue': '#D717E7', // Magenta
    'purple': '#D717E7',      // Fallback for purple to Magenta
}

export const ConsoleCartridge: React.FC<CartridgeProps> = ({ studio, isActive, onClick, index }) => {
    const neonColor = studio.data.accent && ACCENT_MAP[studio.data.accent]
        ? ACCENT_MAP[studio.data.accent]
        : '#1AE8CE'

    return (
        <motion.button
            layoutId={`cartridge-${studio.id}`}
            onClick={onClick}
            className={cn(
                "group relative flex flex-col items-start justify-between w-full p-5 md:p-6 overflow-hidden transition-all duration-300 text-left",
                "bg-[#141720]/40 border border-white/5 backdrop-blur-md rounded-2xl",
                "hover:border-white/20 hover:bg-[#141720]/60",
                isActive ? "ring-1 ring-offset-2 ring-offset-[#0a0a0a] ring-[var(--neon-color)] bg-[#141720]/80 border-[var(--neon-color)]/30" : ""
            )}
            style={{
                '--neon-color': neonColor,
            } as React.CSSProperties}
            whileHover={{ scale: 1.01, y: -2 }}
            whileTap={{ scale: 0.98 }}
        >
            {/* Ambient Background Glow */}
            <div
                className={cn(
                    "absolute inset-0 bg-[radial-gradient(circle_at_top_right,var(--neon-color),transparent_70%)] opacity-0 transition-opacity duration-500",
                    isActive ? "opacity-10" : "group-hover:opacity-5"
                )}
            />

            {/* Glint Effect on Hover */}
            <div className="absolute inset-0 bg-gradient-to-tr from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

            {/* Top Status Status Bar */}
            <div className="flex items-center justify-between w-full mb-4 relative z-10">
                <span className="text-[10px] uppercase tracking-widest text-white/30 font-mono group-hover:text-[var(--neon-color)] transition-colors">
                    MOD-{String(index + 1).padStart(2, '0')}
                </span>
                <div className="flex items-center gap-2">
                    <span className={cn(
                        "text-[9px] font-mono tracking-wider transition-colors duration-300",
                        isActive ? "text-[var(--neon-color)]" : "text-white/20"
                    )}>
                        {isActive ? 'ONLINE' : 'READY'}
                    </span>
                    <div className={cn(
                        "h-1.5 w-1.5 rounded-full transition-all duration-300",
                        isActive || studio.data.status === 'active'
                            ? "bg-[var(--neon-color)] shadow-[0_0_8px_var(--neon-color)] scale-110"
                            : "bg-white/10"
                    )} />
                </div>
            </div>

            {/* Main Content */}
            <div className="relative z-10 w-full mb-1">
                <h3 className={cn(
                    "text-lg md:text-xl font-bold font-[Outfit] tracking-tight transition-all duration-300",
                    isActive ? "text-white text-shadow-neon" : "text-white/90 group-hover:text-white"
                )}>
                    {studio.data.name}
                </h3>
                <p className="text-sm text-gray-400 mt-1 line-clamp-2 md:line-clamp-none font-light leading-relaxed">
                    {studio.data.tagline}
                </p>
            </div>

            {/* Active Revealer (Description & Details) */}
            <AnimatePresence>
                {isActive && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="w-full relative z-10 overflow-hidden"
                    >
                        <div className="pt-4 mt-4 border-t border-white/10">
                            <p className="text-sm text-gray-300 leading-relaxed font-light">
                                {studio.data.description}
                            </p>

                            {/* Decorative Tech Footer */}
                            <div className="mt-5 flex items-center justify-between">
                                <span className="text-[10px] font-mono text-[var(--neon-color)]/70 flex items-center gap-1">
                                    <span className="inline-block w-2 h-2 border border-[var(--neon-color)] rounded-[1px]"></span>
                                    SYSTEM_ACTIVE
                                </span>
                                <div className="h-px w-12 bg-gradient-to-r from-[var(--neon-color)]/50 to-transparent" />
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Visual Tech Decor - Bottom Right Corner */}
            <div className="absolute bottom-3 right-3 flex gap-0.5 opacity-0 group-hover:opacity-40 transition-opacity duration-300">
                {[1, 2, 3].map(i => (
                    <div
                        key={i}
                        className="w-0.5 h-2 bg-[var(--neon-color)] rounded-full"
                        style={{ opacity: 1 - (i * 0.2) }}
                    />
                ))}
            </div>

            {/* Scanline overlay (subtle) */}
            <div className="absolute inset-0 bg-repeat bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPgo8cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSI0IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDIiLz4KPC9zdmc+')] opacity-[0.03] pointer-events-none" />
        </motion.button>
    )
}
