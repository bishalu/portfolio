'use client'

import { useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react'
import { AnimatePresence, LayoutGroup, motion, useInView } from 'framer-motion'

export type ProductStudioId = 'curation-engine' | 'cue' | 'choon'

export interface ProductConsoleItem {
  id: ProductStudioId
  name: string
  tagline: string
  description: string
  accent: string
  capabilities: string[]
  techStack: string[]
  sources: string[]
  studioModules: { id: string; name: string; href: string }[]
  icon?: string
  media?:
    | {
        type: 'video'
        src: string
        poster?: string
        alt?: string
      }
    | {
        type: 'article'
        href: string
        badge: string
        headline: string
        excerpt: string
        ctaLabel: string
      }
}

interface VibesetSwitchboardProps {
  products?: ProductConsoleItem[]
  studios?: ProductConsoleItem[]
}

const PRODUCT_ORDER: ProductStudioId[] = ['curation-engine', 'cue', 'choon']

const SCREEN_LABELS = ['Media', 'Overview', 'Capabilities', 'Technicals'] as const
const SCREEN_LEGENDS = ['MEDIA', 'OVERVIEW', 'INCLUDES', 'TECHNICALS'] as const

export default function VibesetSwitchboard({ products, studios }: VibesetSwitchboardProps) {
  const safeProducts = products ?? studios ?? []
  const sortedProducts = useMemo(
    () => [...safeProducts].sort((a, b) => PRODUCT_ORDER.indexOf(a.id) - PRODUCT_ORDER.indexOf(b.id)),
    [safeProducts],
  )

  const [selectedProductId, setSelectedProductId] = useState<ProductStudioId | null>(null)
  const [activeScreen, setActiveScreen] = useState(0)
  const [isReduced, setIsReduced] = useState(false)
  const [hasRevealed, setHasRevealed] = useState(false)
  const [isMobileView, setIsMobileView] = useState(false)
  const [isDocumentVisible, setIsDocumentVisible] = useState(true)

  const containerRef = useRef<HTMLElement>(null)
  const isInView = useInView(containerRef, { amount: 0.25 })
  const videoElementsRef = useRef<Map<string, HTMLVideoElement>>(new Map())
  const userPausedProductsRef = useRef<Set<ProductStudioId>>(new Set())
  const programmaticPauseRef = useRef(false)

  // Drag tracking for swipe gesture
  const dragStartX = useRef<number>(0)
  const SWIPE_THRESHOLD = 50

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    setIsReduced(mediaQuery.matches)

    const handler = (event: MediaQueryListEvent) => setIsReduced(event.matches)
    mediaQuery.addEventListener('change', handler)
    return () => mediaQuery.removeEventListener('change', handler)
  }, [])

  useEffect(() => {
    const mobileQuery = window.matchMedia('(max-width: 767px)')
    setIsMobileView(mobileQuery.matches)
    const handler = (event: MediaQueryListEvent) => setIsMobileView(event.matches)
    mobileQuery.addEventListener('change', handler)
    return () => mobileQuery.removeEventListener('change', handler)
  }, [])

  useEffect(() => {
    const onVisibilityChange = () => {
      setIsDocumentVisible(document.visibilityState === 'visible')
    }
    onVisibilityChange()
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [])

  useEffect(() => {
    if (!isInView || hasRevealed || sortedProducts.length === 0) return

    const revealDelay = isReduced ? 0 : 1000
    const timer = window.setTimeout(() => {
      setHasRevealed(true)
      if (!selectedProductId) {
        setSelectedProductId(sortedProducts[0].id)
      }
    }, revealDelay)

    return () => window.clearTimeout(timer)
  }, [hasRevealed, isInView, isReduced, selectedProductId, sortedProducts])

  useEffect(() => {
    if (!selectedProductId && sortedProducts.length > 0) {
      setSelectedProductId(sortedProducts[0].id)
    }
  }, [selectedProductId, sortedProducts])

  // Reset screen to 0 whenever the product changes
  useEffect(() => {
    setActiveScreen(0)
  }, [selectedProductId])

  // Listen for hash changes to deep-link to specific tabs
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash
      if (hash.startsWith('#vibeset-')) {
        const id = hash.replace('#vibeset-', '') as ProductStudioId
        if (PRODUCT_ORDER.includes(id)) {
          setSelectedProductId(id)
        }
      }
    }

    // Check on mount
    handleHashChange()

    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  if (sortedProducts.length === 0) {
    return null
  }

  const activeProduct = sortedProducts.find((item) => item.id === selectedProductId) ?? sortedProducts[0]
  const TOTAL_SCREENS = 4

  const setVideoRef = (key: string) => (node: HTMLVideoElement | null) => {
    if (node) {
      videoElementsRef.current.set(key, node)
      return
    }
    videoElementsRef.current.delete(key)
  }

  const pauseVideoElement = (video: HTMLVideoElement) => {
    if (video.paused) return
    programmaticPauseRef.current = true
    video.pause()
    window.setTimeout(() => {
      programmaticPauseRef.current = false
    }, 0)
  }

  const onVideoPause = () => {
    if (!selectedProductId || programmaticPauseRef.current) return
    userPausedProductsRef.current.add(selectedProductId)
  }

  const onVideoPlay = () => {
    if (!selectedProductId) return
    userPausedProductsRef.current.delete(selectedProductId)
  }

  useEffect(() => {
    const videoItems = Array.from(videoElementsRef.current.values())
    if (videoItems.length === 0 || !selectedProductId) return

    for (const video of videoItems) {
      const productId = video.dataset.productId as ProductStudioId | undefined
      const view = video.dataset.view
      const isActiveProduct = productId === selectedProductId
      const isActiveView = isMobileView ? view === 'mobile' : view === 'desktop'
      const isMobileMediaScreenActive = !isMobileView || activeScreen === 0
      const shouldPlay =
        isInView &&
        isDocumentVisible &&
        isActiveProduct &&
        isActiveView &&
        isMobileMediaScreenActive &&
        !userPausedProductsRef.current.has(selectedProductId)

      if (!shouldPlay) {
        pauseVideoElement(video)
        continue
      }

      const playPromise = video.play()
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => undefined)
      }
    }
  }, [activeScreen, isDocumentVisible, isInView, isMobileView, selectedProductId])

  const goToScreen = (idx: number) => {
    setActiveScreen(Math.max(0, Math.min(TOTAL_SCREENS - 1, idx)))
  }

  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, productId: ProductStudioId) => {
    const currentIndex = sortedProducts.findIndex((item) => item.id === productId)
    if (currentIndex < 0) return

    let nextIndex = currentIndex

    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault()
      nextIndex = (currentIndex + 1) % sortedProducts.length
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault()
      nextIndex = (currentIndex - 1 + sortedProducts.length) % sortedProducts.length
    } else if (event.key === 'Home') {
      event.preventDefault()
      nextIndex = 0
    } else if (event.key === 'End') {
      event.preventDefault()
      nextIndex = sortedProducts.length - 1
    } else {
      return
    }

    const next = sortedProducts[nextIndex]
    setSelectedProductId(next.id)
    window.requestAnimationFrame(() => {
      document.getElementById(`product-tab-${next.id}`)?.focus()
    })
  }

  const onConsoleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowRight') {
      event.preventDefault()
      goToScreen(activeScreen + 1)
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault()
      goToScreen(activeScreen - 1)
    }
  }

  const renderMediaPanel = (options: { panelKey: string; view: 'desktop' | 'mobile' }) => {
    const { panelKey, view } = options
    const media = activeProduct.media
    const showVideo = media?.type === 'video'
    const showArticle = media?.type === 'article'

    if (showVideo) {
      return (
        <div className="media-frame">
          <video
            ref={setVideoRef(panelKey)}
            className="media-video"
            src={media.src}
            poster={media.poster}
            aria-label={media.alt ?? `${activeProduct.name} demo video`}
            data-product-id={activeProduct.id}
            data-view={view}
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
            controls
            onPause={onVideoPause}
            onPlay={onVideoPlay}
          />
        </div>
      )
    }

    if (showArticle) {
      return (
        <a
          href={media.href}
          target="_blank"
          rel="noopener noreferrer"
          className="media-frame media-article"
          style={{ '--media-accent': activeProduct.accent } as CSSProperties}
          aria-label={`${media.ctaLabel}: ${media.headline}`}
          data-view={view}
          data-panel-key={panelKey}
        >
          <div className="media-article-glow" aria-hidden="true" />
          <span className="media-placeholder-badge">{media.badge}</span>
          <div className="media-article-copy">
            <h3>{media.headline}</h3>
            <p>{media.excerpt}</p>
          </div>
          <span className="media-article-cta">
            <span>{media.ctaLabel}</span>
            <span className="media-article-cta-icon" aria-hidden="true">
              ↗
            </span>
          </span>
        </a>
      )
    }

    return (
      <div className="media-placeholder" style={{ '--media-accent': activeProduct.accent } as CSSProperties}>
        <span className="media-placeholder-badge">Media Coming Soon</span>
        <h3>{activeProduct.name}</h3>
        <p>{activeProduct.tagline}</p>
      </div>
    )
  }

  // Screens content factory
  const renderScreen = (screenIndex: number) => {
    const legend = SCREEN_LEGENDS[screenIndex]

    if (screenIndex === 0) {
      return (
        <fieldset className="hud-quadrant screen-quadrant media-quadrant">
          <legend>{legend}</legend>
          {renderMediaPanel({ panelKey: `${activeProduct.id}-mobile-media`, view: 'mobile' })}
        </fieldset>
      )
    }

    if (screenIndex === 1) {
      return (
        <fieldset className="hud-quadrant screen-quadrant overview-screen">
          <legend>{legend}</legend>
          <div className="console-main-col">
            <h2 className="console-studio-name">{activeProduct.name}</h2>
            <span className="console-tagline">{activeProduct.tagline}</span>
            <p className="console-description">{activeProduct.description}</p>
          </div>
        </fieldset>
      )
    }

    if (screenIndex === 2) {
      return (
        <fieldset className="hud-quadrant screen-quadrant capabilities-screen">
          <legend>{legend}</legend>
          <div className="includes-buttons" aria-label={`${activeProduct.name} studio modules`}>
            {activeProduct.studioModules.map((studio) => (
              <a key={studio.id} href={studio.href} className="includes-button">
                {studio.name}
              </a>
            ))}
          </div>
        </fieldset>
      )
    }

    if (screenIndex === 3) {
      return (
        <fieldset className="hud-quadrant screen-quadrant technicals-screen">
          <legend>{legend}</legend>
          <div className="tech-tags">
            {activeProduct.techStack.map((tech) => (
              <span key={tech} className="tech-tag">
                {tech}
              </span>
            ))}
          </div>
        </fieldset>
      )
    }

    return null
  }

  return (
    <section ref={containerRef} className="vibeset-switchboard">
      <LayoutGroup>
        <AnimatePresence mode="wait">
          {!hasRevealed ? (
            <motion.div
              key="cover"
              className="switchboard-cover"
              initial={{ opacity: 1 }}
              exit={{ opacity: 0, scale: isReduced ? 1 : 1.03, filter: isReduced ? 'none' : 'blur(8px)' }}
              transition={{ duration: 0.45, ease: [0.33, 1, 0.68, 1] }}
            >
              <div className="cover-glow" />
              <div className="cover-content">
                <motion.div
                  className="retina-scan-container"
                  initial={{ scale: 0.86, opacity: 0 }}
                  animate={isInView ? { scale: 1, opacity: 1 } : {}}
                  transition={{ duration: 0.35 }}
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
                        transition={{ duration: isReduced ? 0.2 : 0.45, ease: 'easeInOut' }}
                      />
                    )}
                  </svg>
                </motion.div>
                <motion.h2
                  className="cover-title"
                  initial={{ opacity: 0, y: 8 }}
                  animate={isInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ delay: 0.12, duration: 0.25 }}
                >
                  Vibeset
                </motion.h2>
                <motion.p
                  className="cover-tagline"
                  initial={{ opacity: 0 }}
                  animate={isInView ? { opacity: 1 } : {}}
                  transition={{ delay: 0.18, duration: 0.22 }}
                >
                  Initializing product console...
                </motion.p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="interface"
              className="switchboard-interface"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              {/* ── Product selector tab bar ── */}
              <div className="tab-bar-container">
                <div className="studio-tab-bar" role="tablist" aria-label="Vibeset products">
                  {sortedProducts.map((product) => {
                    const isSelected = selectedProductId === product.id
                    return (
                      <button
                        key={product.id}
                        id={`product-tab-${product.id}`}
                        role="tab"
                        aria-controls={`product-panel-${product.id}`}
                        aria-selected={isSelected}
                        tabIndex={isSelected ? 0 : -1}
                        className={`studio-tab ${isSelected ? 'selected' : ''}`}
                        onClick={() => setSelectedProductId(product.id)}
                        onKeyDown={(event) => onTabKeyDown(event, product.id)}
                      >
                        <div className="tab-content">
                          {product.icon && <img src={product.icon} alt="" className="tab-icon" />}
                          <span className="tab-name">{product.name.replace('Vibeset ', '')}</span>
                        </div>
                        {isSelected && (
                          <motion.div
                            className="tab-indicator"
                            layoutId="activeTabIndicator"
                            transition={{ duration: 0.25, ease: 'easeOut' }}
                            initial={false}
                            style={{ backgroundColor: product.accent, boxShadow: `0 0 12px ${product.accent}80` }}
                          />
                        )}
                        {isSelected && (
                          <div
                            className="tab-glow-bg"
                            style={{
                              background: `radial-gradient(circle at bottom, ${product.accent}30, transparent 70%)`,
                            }}
                          />
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* ── Console display ── */}
              <div className="console-display">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeProduct.id}
                    role="tabpanel"
                    id={`product-panel-${activeProduct.id}`}
                    aria-labelledby={`product-tab-${activeProduct.id}`}
                    className="console-content"
                    initial={isReduced ? { opacity: 1 } : { opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={isReduced ? { opacity: 1 } : { opacity: 0, y: -6 }}
                    transition={{ duration: isReduced ? 0.05 : 0.2 }}
                  >
                    {/* ── DESKTOP: Apple Feature Flow (Option 1) ── */}
                    <motion.div
                      className="desktop-grid-layout apple-feature-layout"
                      initial="hidden"
                      animate="visible"
                      variants={{
                        hidden: {},
                        visible: {
                          transition: { staggerChildren: 0.1, delayChildren: 0.1 },
                        },
                      }}
                    >
                      {/* Left Column: Text & Includes */}
                      <div className="feature-col-left">
                        <motion.fieldset
                          className="hud-quadrant overview-quadrant"
                          variants={{
                            hidden: { opacity: 0, x: -20 },
                            visible: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } },
                          }}
                        >
                          <legend>OVERVIEW</legend>
                          <div className="console-main-col">
                            <h2 className="console-studio-name">{activeProduct.name}</h2>
                            <span className="console-tagline">{activeProduct.tagline}</span>
                            <p className="console-description">{activeProduct.description}</p>
                          </div>
                        </motion.fieldset>

                        <motion.fieldset
                          className="hud-quadrant features-quadrant"
                          variants={{
                            hidden: { opacity: 0, x: -20 },
                            visible: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } },
                          }}
                        >
                          <legend>INCLUDES</legend>
                          <div className="includes-buttons" aria-label={`${activeProduct.name} studio modules`}>
                            {activeProduct.studioModules.map((studio) => (
                              <a key={studio.id} href={studio.href} className="includes-button">
                                {studio.name}
                              </a>
                            ))}
                          </div>
                        </motion.fieldset>
                      </div>

                      {/* Right Column: Hero Media & Tech Specs */}
                      <div className="feature-col-right">
                        <motion.fieldset
                          className="hud-quadrant preview-quadrant media-quadrant"
                          variants={{
                            hidden: { opacity: 0, scale: 0.98, filter: 'blur(4px)' },
                            visible: {
                              opacity: 1,
                              scale: 1,
                              filter: 'blur(0px)',
                              transition: { type: 'spring', stiffness: 200, damping: 20 },
                            },
                          }}
                        >
                          <legend>MEDIA</legend>
                          {renderMediaPanel({ panelKey: `${activeProduct.id}-desktop-media`, view: 'desktop' })}
                        </motion.fieldset>

                        <motion.fieldset
                          className="hud-quadrant arch-quadrant"
                          variants={{
                            hidden: { opacity: 0, y: 20 },
                            visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } },
                          }}
                        >
                          <legend>TECHNICALS</legend>
                          <div className="tech-tags">
                            {activeProduct.techStack.map((tech) => (
                              <span key={tech} className="tech-tag">
                                {tech}
                              </span>
                            ))}
                          </div>
                        </motion.fieldset>
                      </div>
                    </motion.div>

                    {/* ── MOBILE: Paged screen deck ── */}
                    <div
                      className="mobile-screen-deck"
                      role="region"
                      aria-label="Product information screens"
                      onKeyDown={onConsoleKeyDown}
                      tabIndex={0}
                    >
                      {/* Drag / swipe container */}
                      <div
                        className="screen-track-outer"
                        onPointerDown={(e) => {
                          dragStartX.current = e.clientX
                        }}
                        onPointerUp={(e) => {
                          const delta = dragStartX.current - e.clientX
                          if (Math.abs(delta) > SWIPE_THRESHOLD) {
                            goToScreen(activeScreen + (delta > 0 ? 1 : -1))
                          }
                        }}
                      >
                        <AnimatePresence mode="wait">
                          <motion.div
                            key={`${activeProduct.id}-screen-${activeScreen}`}
                            className="screen-slide"
                            initial={isReduced ? { opacity: 1 } : { opacity: 0, x: 48 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={isReduced ? { opacity: 1 } : { opacity: 0, x: -48 }}
                            transition={{ duration: 0.22, ease: [0.33, 1, 0.68, 1] }}
                          >
                            {renderScreen(activeScreen)}
                          </motion.div>
                        </AnimatePresence>
                      </div>

                      {/* Screen navigation: arrows + dot stepper */}
                      <div className="screen-nav" aria-label="Screen navigation">
                        {/* Prev arrow */}
                        <button
                          className="screen-arrow screen-arrow--prev"
                          onClick={() => goToScreen(activeScreen - 1)}
                          disabled={activeScreen === 0}
                          aria-label="Previous screen"
                        >
                          <svg
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                          >
                            <path d="M15 18l-6-6 6-6" />
                          </svg>
                        </button>

                        {/* Dot stepper */}
                        <div className="screen-dots" role="tablist" aria-label="Screen indicators">
                          {SCREEN_LABELS.map((label, idx) => (
                            <button
                              key={label}
                              role="tab"
                              aria-selected={idx === activeScreen}
                              aria-label={`Go to ${label} screen`}
                              className={`screen-dot ${idx === activeScreen ? 'screen-dot--active' : ''}`}
                              style={idx === activeScreen ? { background: activeProduct.accent } : undefined}
                              onClick={() => goToScreen(idx)}
                            />
                          ))}
                        </div>

                        {/* Next arrow */}
                        <button
                          className="screen-arrow screen-arrow--next"
                          onClick={() => goToScreen(activeScreen + 1)}
                          disabled={activeScreen === TOTAL_SCREENS - 1}
                          aria-label="Next screen"
                        >
                          <svg
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                          >
                            <path d="M9 18l6-6-6-6" />
                          </svg>
                        </button>
                      </div>

                      {/* Screen label badge */}
                      <div className="screen-label-badge" aria-live="polite" aria-atomic="true">
                        <span className="screen-label-text">{SCREEN_LABELS[activeScreen]}</span>
                        <span className="screen-label-count">
                          {activeScreen + 1} / {TOTAL_SCREENS}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </LayoutGroup>
    </section>
  )
}
