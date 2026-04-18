import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

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
  purple: '#D717E7', // Fallback for purple to Magenta
}

export const ConsoleCartridge: React.FC<CartridgeProps> = ({ studio, isActive, onClick, index }) => {
  const neonColor = studio.data.accent && ACCENT_MAP[studio.data.accent] ? ACCENT_MAP[studio.data.accent] : '#1AE8CE'

  return (
    <motion.button
      layoutId={`cartridge-${studio.id}`}
      onClick={onClick}
      className={cn(
        'group relative flex w-full flex-col items-start justify-between overflow-hidden p-5 text-left transition-all duration-300 md:p-6',
        'rounded-2xl border border-white/5 bg-[#141720]/40 backdrop-blur-md',
        'hover:border-white/20 hover:bg-[#141720]/60',
        isActive
          ? 'border-[var(--neon-color)]/30 bg-[#141720]/80 ring-1 ring-[var(--neon-color)] ring-offset-2 ring-offset-[#0a0a0a]'
          : '',
      )}
      style={
        {
          '--neon-color': neonColor,
        } as React.CSSProperties
      }
      whileHover={{ scale: 1.01, y: -2 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Ambient Background Glow */}
      <div
        className={cn(
          'absolute inset-0 bg-[radial-gradient(circle_at_top_right,var(--neon-color),transparent_70%)] opacity-0 transition-opacity duration-500',
          isActive ? 'opacity-10' : 'group-hover:opacity-5',
        )}
      />

      {/* Glint Effect on Hover */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-tr from-white/5 to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />

      {/* Top Status Status Bar */}
      <div className="relative z-10 mb-4 flex w-full items-center justify-between">
        <span className="font-mono text-[10px] tracking-widest text-white/30 uppercase transition-colors group-hover:text-[var(--neon-color)]">
          MOD-{String(index + 1).padStart(2, '0')}
        </span>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'font-mono text-[9px] tracking-wider transition-colors duration-300',
              isActive ? 'text-[var(--neon-color)]' : 'text-white/20',
            )}
          >
            {isActive ? 'ONLINE' : 'READY'}
          </span>
          <div
            className={cn(
              'h-1.5 w-1.5 rounded-full transition-all duration-300',
              isActive || studio.data.status === 'active'
                ? 'scale-110 bg-[var(--neon-color)] shadow-[0_0_8px_var(--neon-color)]'
                : 'bg-white/10',
            )}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 mb-1 w-full">
        <h3
          className={cn(
            'font-[Outfit] text-lg font-bold tracking-tight transition-all duration-300 md:text-xl',
            isActive ? 'text-shadow-neon text-white' : 'text-white/90 group-hover:text-white',
          )}
        >
          {studio.data.name}
        </h3>
        <p className="mt-1 line-clamp-2 text-sm leading-relaxed font-light text-gray-400 md:line-clamp-none">
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
            className="relative z-10 w-full overflow-hidden"
          >
            <div className="mt-4 border-t border-white/10 pt-4">
              <p className="text-sm leading-relaxed font-light text-gray-300">{studio.data.description}</p>

              {/* Decorative Tech Footer */}
              <div className="mt-5 flex items-center justify-between">
                <span className="flex items-center gap-1 font-mono text-[10px] text-[var(--neon-color)]/70">
                  <span className="inline-block h-2 w-2 rounded-[1px] border border-[var(--neon-color)]"></span>
                  SYSTEM_ACTIVE
                </span>
                <div className="h-px w-12 bg-gradient-to-r from-[var(--neon-color)]/50 to-transparent" />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Visual Tech Decor - Bottom Right Corner */}
      <div className="absolute right-3 bottom-3 flex gap-0.5 opacity-0 transition-opacity duration-300 group-hover:opacity-40">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-2 w-0.5 rounded-full bg-[var(--neon-color)]" style={{ opacity: 1 - i * 0.2 }} />
        ))}
      </div>

      {/* Scanline overlay (subtle) */}
      <div className="pointer-events-none absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPgo8cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSI0IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDIiLz4KPC9zdmc+')] bg-repeat opacity-[0.03]" />
    </motion.button>
  )
}
