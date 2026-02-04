"use client"

import { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence, useInView, LayoutGroup } from "framer-motion"

// Studio type matching the content collection schema
interface Studio {
    id: string
    data: {
        id: string
        name: string
        tagline: string
        description: string
        accent: string
        icon?: string
        features: string[]
        techStack: string[]
        status: string
        coverImage?: string // Optional cover image for the poster
    }
}

interface VibesetSwitchboardProps {
    studios: Studio[]
}

export default function VibesetSwitchboard({ studios }: VibesetSwitchboardProps) {
    // Select the first studio by default once revealed, or null initially if we want a "choose" state
    // But for a "console" feel, having one active is better. Let's start null and set on reveal.
    const [selectedStudioId, setSelectedStudioId] = useState<string | null>(null)
    const [isReduced, setIsReduced] = useState(false)
    const [hasRevealed, setHasRevealed] = useState(false)

    const containerRef = useRef<HTMLElement>(null)
    const isInView = useInView(containerRef, { amount: 0.3, once: true })

    // Check for reduced motion preference
    useEffect(() => {
        const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)")
        setIsReduced(mediaQuery.matches)

        const handler = (e: MediaQueryListEvent) => setIsReduced(e.matches)
        mediaQuery.addEventListener("change", handler)
        return () => mediaQuery.removeEventListener("change", handler)
    }, [])

    // Auto-reveal logic: When scrolled into view, trigger the "Cover" animation
    // and then settle into the main console state.
    useEffect(() => {
        if (isInView && !hasRevealed) {
            const timer = setTimeout(() => {
                setHasRevealed(true)
                // Auto-select the first studio if none selected
                if (!selectedStudioId && studios.length > 0) {
                    setSelectedStudioId(studios[0].id)
                }
            }, 1200) // Slightly faster reveal
            return () => clearTimeout(timer)
        }
    }, [isInView, hasRevealed, studios, selectedStudioId])

    const activeStudio = studios.find((s) => s.id === selectedStudioId) || studios[0]

    return (
        <section
            ref={containerRef}
            className="vibeset-switchboard"
            id="vibeset"
        >
            <LayoutGroup>
                <AnimatePresence mode="wait">
                    {!hasRevealed ? (
                        /* ══════════════════════════════════════════════════════════
                           COVER STATE — Initial Reveal Animation
                           ══════════════════════════════════════════════════════════ */
                        <motion.div
                            key="cover"
                            className="switchboard-cover"
                            initial={{ opacity: 1 }}
                            exit={{
                                opacity: 0,
                                scale: 1.05,
                                filter: "blur(12px)"
                            }}
                            transition={{ duration: 0.8, ease: [0.33, 1, 0.68, 1] }}
                        >
                            <div className="cover-glow" />
                            <div className="cover-content">
                                <motion.div
                                    className="retina-scan-container"
                                    initial={{ scale: 0.8, opacity: 0 }}
                                    animate={isInView ? { scale: 1, opacity: 1 } : {}}
                                    transition={{ duration: 0.8 }}
                                >
                                    <svg className="retina-scan-ring" viewBox="0 0 100 100">
                                        <circle cx="50" cy="50" r="46" className="ring-track" />
                                        {isInView && (
                                            <motion.circle
                                                cx="50"
                                                cy="50"
                                                r="46"
                                                className="ring-indicator"
                                                initial={{ pathLength: 0, opacity: 0, rotate: -90 }}
                                                animate={{ pathLength: 1, opacity: 1, rotate: -90 }}
                                                transition={{ duration: 1.2, ease: "easeInOut" }}
                                            />
                                        )}
                                    </svg>
                                </motion.div>
                                <motion.h2
                                    className="cover-title"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={isInView ? { opacity: 1, y: 0 } : {}}
                                    transition={{ delay: 0.3, duration: 0.6 }}
                                >
                                    Vibeset
                                </motion.h2>
                                <motion.p
                                    className="cover-tagline"
                                    initial={{ opacity: 0 }}
                                    animate={isInView ? { opacity: 1 } : {}}
                                    transition={{ delay: 0.5, duration: 0.5 }}
                                >
                                    Initializing Studio Interface...
                                </motion.p>
                            </div>
                        </motion.div>
                    ) : (
                        /* ══════════════════════════════════════════════════════════
                           MAIN INTERFACE — Master-Detail View
                           ══════════════════════════════════════════════════════════ */
                        <motion.div
                            key="interface"
                            className="switchboard-interface"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.6 }}
                        >
                            {/* 1. MASTER VIEW: Horizontal "Netflix-style" Reel */}
                            <div className="reel-container">

                                <div className="gallery-track">
                                    {studios.map((studio) => {
                                        const isSelected = selectedStudioId === studio.id
                                        return (
                                            <button
                                                key={studio.id}
                                                className={`studio-poster ${isSelected ? "selected" : ""}`}
                                                onClick={() => setSelectedStudioId(studio.id)}
                                                aria-label={`Select ${studio.data.name}`}
                                                aria-pressed={isSelected}
                                            >
                                                <div className="poster-visual">
                                                    {/* Gradient background based on accent color */}
                                                    <div
                                                        className="poster-gradient"
                                                        style={{
                                                            background: `linear-gradient(to top, ${studio.data.accent}40, transparent)`
                                                        }}
                                                    />
                                                    {studio.data.icon && (
                                                        <img
                                                            src={studio.data.icon}
                                                            alt=""
                                                            className="poster-icon"
                                                        />
                                                    )}
                                                </div>
                                                <div className="poster-info">
                                                    <span className="poster-name">{studio.data.name}</span>
                                                </div>
                                                {/* Active Border Glow */}
                                                {isSelected && (
                                                    <motion.div
                                                        className="poster-glow"
                                                        layoutId="activeGlow"
                                                        transition={{ duration: 0.3 }}
                                                    />
                                                )}
                                            </button>
                                        )
                                    })}
                                </div>
                                {/* Mobile Scroll Indicator */}
                                <div className="mobile-scroll-hint">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M9 18l6-6-6-6" />
                                    </svg>
                                </div>
                            </div>

                            {/* 2. DETAIL VIEW: The Console */}
                            <div className="console-display">
                                <AnimatePresence mode="wait">
                                    {activeStudio && (
                                        <motion.div
                                            key={activeStudio.id}
                                            className="console-content"
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            exit={{ opacity: 0, y: -10 }}
                                            transition={{ duration: 0.3 }}
                                        >
                                            {/* Header Section */}
                                            <div className="console-header-row">
                                                {/* Grid Cell 1: Interactive Preview */}
                                                <fieldset className="hud-quadrant preview-quadrant">
                                                    <legend>INTERACTIVE PREVIEW</legend>
                                                    <div className="action-placeholder" style={{ height: "180px", width: "100%", justifyContent: "flex-start", paddingLeft: "1.5rem", border: "none", background: "transparent" }}>
                                                        <span style={{ opacity: 0.5, textTransform: "uppercase", fontSize: "0.75rem", letterSpacing: "0.05em" }}>Load Content...</span>
                                                    </div>
                                                </fieldset>

                                                {/* Grid Cell 2: Features */}
                                                <fieldset className="hud-quadrant features-quadrant">
                                                    <legend>FEATURES</legend>
                                                    <div className="features-list" style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                                                        {activeStudio.data.features.slice(0, 4).map((feature, i) => (
                                                            <motion.button
                                                                key={i}
                                                                whileHover={{ scale: 1.02, backgroundColor: "rgba(255, 255, 255, 0.08)" }}
                                                                whileTap={{ scale: 0.98 }}
                                                                onClick={() => console.log(`Feature clicked: ${feature}`)}
                                                                style={{
                                                                    display: "flex",
                                                                    alignItems: "center",
                                                                    gap: "1rem",
                                                                    width: "100%",
                                                                    padding: "0.6rem 1rem",
                                                                    background: "rgba(255, 255, 255, 0.02)",
                                                                    border: "1px solid rgba(255, 255, 255, 0.05)",
                                                                    borderRadius: "12px",
                                                                    color: "rgba(253, 245, 227, 0.9)",
                                                                    fontSize: "0.85rem",
                                                                    textAlign: "left",
                                                                    cursor: "pointer",
                                                                    position: "relative",
                                                                    overflow: "hidden"
                                                                }}
                                                            >
                                                                {/* "Cyber-Pill" Indicator */}
                                                                <div style={{
                                                                    width: "4px",
                                                                    height: "24px",
                                                                    borderRadius: "4px",
                                                                    background: activeStudio.data.accent,
                                                                    boxShadow: `0 0 12px ${activeStudio.data.accent}`,
                                                                    flexShrink: 0
                                                                }} />

                                                                <span style={{ fontWeight: 500, letterSpacing: "0.01em" }}>{feature}</span>

                                                                {/* Subtle gradient overlay for 2026 feel */}
                                                                <div style={{
                                                                    position: "absolute",
                                                                    inset: 0,
                                                                    background: `linear-gradient(90deg, transparent, ${activeStudio.data.accent}05)`,
                                                                    pointerEvents: "none"
                                                                }} />
                                                            </motion.button>
                                                        ))}
                                                    </div>
                                                </fieldset>
                                            </div>

                                            {/* Content Grid */}
                                            <div className="console-grid">
                                                {/* Left Col: Overview */}
                                                <fieldset className="hud-quadrant overview-quadrant">
                                                    <legend>OVERVIEW</legend>
                                                    <div className="console-main-col">
                                                        <div className="console-title-group" style={{ marginBottom: "1rem" }}>
                                                            <h2 className="console-studio-name">{activeStudio.data.name}</h2>
                                                            <span className="console-tagline">{activeStudio.data.tagline}</span>
                                                        </div>
                                                        <p className="console-description">{activeStudio.data.description}</p>
                                                    </div>
                                                </fieldset>

                                                {/* Right Col: Architecture */}
                                                <fieldset className="hud-quadrant arch-quadrant">
                                                    <legend>TECHNICALS</legend>
                                                    <div className="tech-tags">
                                                        {activeStudio.data.techStack.map((tech, i) => (
                                                            <span key={i} className="tech-tag">{tech}</span>
                                                        ))}
                                                    </div>
                                                </fieldset>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </LayoutGroup>
        </section >
    )
}

