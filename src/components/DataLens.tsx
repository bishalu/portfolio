"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence, useInView, useSpring, useTransform, type Variants } from "framer-motion"

// ════════════════════════════════════════════════════════════════════════════
// TYPES
// ════════════════════════════════════════════════════════════════════════════

interface Publication {
    id: string
    data: {
        title: string
        abstract: string
        year: number
        journal?: string
        doi?: string
        pdf?: string
        coauthors?: string[]
        tags?: string[]
    }
}

interface DataLensProps {
    publications: Publication[]
}

// ════════════════════════════════════════════════════════════════════════════
// SUB-COMPONENT: React Publication Card
// ════════════════════════════════════════════════════════════════════════════

const PublicationCard = ({ pub }: { pub: Publication["data"] }) => {
    // Flip state for 3D card interaction
    const [isFlipped, setIsFlipped] = useState(false)

    const authorDisplay = pub.coauthors && pub.coauthors.length > 0
        ? `with ${pub.coauthors.join(', ')}`
        : ''

    const handleFlip = () => {
        setIsFlipped(!isFlipped)
    }

    return (
        <motion.article
            className={`pub-card flip-card-container ${isFlipped ? 'is-flipped' : ''}`}
            onClick={handleFlip}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleFlip() }}
            role="button"
            tabIndex={0}
            aria-label={`${pub.title}. Click to ${isFlipped ? 'view summary' : 'read abstract'}`}
            style={{ perspective: 1000 }}
        >
            <motion.div
                className="flip-card-inner"
                animate={{ rotateY: isFlipped ? 180 : 0 }}
                transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
                style={{ transformStyle: 'preserve-3d', position: 'relative', width: '100%', height: '100%' }}
            >
                {/* ─── Front Face ─────────────────────────────────────────────── */}
                <div className="flip-card-face flip-card-front pub-content-wrapper">
                    <header className="pub-header">
                        <span className="pub-year">{pub.year}</span>
                        <h3 className="pub-title">{pub.title}</h3>
                        {pub.journal && <p className="pub-journal">{pub.journal}</p>}
                        {pub.coauthors && pub.coauthors.length > 0 && (
                            <p className="pub-authors">{authorDisplay}</p>
                        )}
                    </header>

                    {pub.tags && pub.tags.length > 0 && (
                        <div className="pub-tags">
                            {pub.tags.map((tag, i) => (
                                <span key={i} className="tag">{tag}</span>
                            ))}
                        </div>
                    )}

                    {/* Flip Hint Icon */}
                    <div className="flip-hint">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M17 1l4 4-4 4" />
                            <path d="M3 11V9a4 4 0 0 1 4-4h14" />
                            <path d="M7 23l-4-4 4-4" />
                            <path d="M21 13v2a4 4 0 0 1-4 4H3" />
                        </svg>
                        <span>Flip for abstract</span>
                    </div>
                </div>

                {/* ─── Back Face ──────────────────────────────────────────────── */}
                <div className="flip-card-face flip-card-back pub-content-wrapper">
                    <header className="pub-header">
                        <span className="pub-year">{pub.year}</span>
                        <h3 className="pub-title" style={{ fontSize: '1rem' }}>{pub.title}</h3>
                    </header>

                    <p className="pub-abstract-full">{pub.abstract}</p>

                    <footer className="pub-links">
                        {pub.doi && (
                            <a href={`https://doi.org/${pub.doi}`} className="pub-link" target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()}>
                                <span>DOI</span>
                            </a>
                        )}
                        {pub.pdf && (
                            <a href={pub.pdf} className="pub-link" target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()}>
                                <span>PDF</span>
                            </a>
                        )}
                    </footer>

                    {/* Close Hint Icon */}
                    <div className="flip-hint">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M7 23l-4-4 4-4" />
                            <path d="M21 13v2a4 4 0 0 1-4 4H3" />
                            <path d="M17 1l4 4-4 4" />
                            <path d="M3 11V9a4 4 0 0 1 4-4h14" />
                        </svg>
                        <span>Flip back</span>
                    </div>
                </div>
            </motion.div>
        </motion.article>
    )
}


// ════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT: Data Lens
// ════════════════════════════════════════════════════════════════════════════

export default function DataLens({ publications }: DataLensProps) {
    const containerRef = useRef<HTMLElement>(null)
    const [isRevealed, setIsRevealed] = useState(false)
    const [isReduced, setIsReduced] = useState(false)
    const isInView = useInView(containerRef, { amount: 0.5, once: true })

    // Check reduced motion
    useEffect(() => {
        const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)")
        setIsReduced(mediaQuery.matches)
        const handler = (e: MediaQueryListEvent) => setIsReduced(e.matches)
        mediaQuery.addEventListener("change", handler)
        return () => mediaQuery.removeEventListener("change", handler)
    }, [])

    // Trigger Reveal Animation
    useEffect(() => {
        if (isInView && !isRevealed) {
            // Delay slightly to allow user to see the "locked" state briefly
            const timer = setTimeout(() => {
                setIsRevealed(true)
            }, 500)
            return () => clearTimeout(timer)
        }
    }, [isInView, isRevealed])

    // ─── Animation Variants ─────────────────────────────────────────────────

    const contentVariants: Variants = {
        hidden: {
            opacity: 0,
            clipPath: "circle(0% at 50% 50%)",
            filter: "blur(20px)",
            scale: 0.9
        },
        visible: {
            opacity: 1,
            clipPath: "circle(150% at 50% 50%)",
            filter: "blur(0px)",
            scale: 1,
            transition: {
                duration: 1.5,
                ease: "easeInOut", // Fluid lens opening
                delay: 0.2
            }
        }
    }

    const coverVariants: Variants = {
        locked: {
            opacity: 1,
            scale: 1,
            display: "flex"
        },
        scanning: {
            opacity: 0,
            scale: 1.5, // Expand out
            transition: {
                duration: 0.8,
                ease: "easeIn",
                delay: 0.1
            },
            transitionEnd: {
                display: "none"
            }
        }
    }

    // Reduced motion fallback
    if (isReduced) {
        return (
            <section className="datalens-section" id="research">
                <div className="datalens-header">
                    <h2 className="section-title">Research</h2>
                </div>
                <div className="publications-grid">
                    {publications.map((pub) => (
                        <PublicationCard key={pub.id} pub={pub.data} />
                    ))}
                </div>
            </section>
        )
    }

    return (
        <section ref={containerRef} className="datalens-section" id="research">

            {/* ─── Content Layer (Revealed by Iris) ───────────────────── */}
            <motion.div
                className="datalens-content"
                initial="hidden"
                animate={isRevealed ? "visible" : "hidden"}
                variants={contentVariants}
            >
                <div className="datalens-header">
                    <h2 className="section-title">Research</h2>
                    <p className="section-subtitle">Systems Architecture & Intelligence Analysis</p>
                </div>

                <div className="publications-grid">
                    {publications.map((pub, index) => (
                        <motion.div
                            key={pub.id}
                            initial={{ opacity: 0, y: 30 }}
                            animate={isRevealed ? { opacity: 1, y: 0 } : {}}
                            transition={{ delay: 1.2 + (index * 0.1), duration: 0.6, ease: "easeOut" }}
                        >
                            <PublicationCard pub={pub.data} />
                        </motion.div>
                    ))}
                </div>
            </motion.div>

            {/* ─── Cover Layer (The 'Shutter') ─────────────────────────── */}
            <motion.div
                className="datalens-cover"
                variants={coverVariants}
                initial="locked"
                animate={isRevealed ? "scanning" : "locked"}
            >
                {/* Lock Icon / Status Text */}
                <motion.div
                    className="cover-status"
                    animate={isRevealed ? { opacity: 0, scale: 0.5 } : { opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="lock-icon-wrapper">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lock-icon">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                        </svg>
                    </div>
                    <span className="status-text">Optical Lens Inactive</span>
                </motion.div>
            </motion.div>

        </section>
    )
}

// ════════════════════════════════════════════════════════════════════════════
// STYLES (injected via CSS Modules or Styled Components usually, but scoped here)
// ════════════════════════════════════════════════════════════════════════════
// Note: In Next.js/React we'd use CSS Modules. In Astro + React, we can ensure
// styles are global or use a separate CSS file. For simplicity in this artifact,
// I'll rely on the global CSS but I'll add specific class definitions in global
// scope via a <style> tag in the wrapping Astro component.
