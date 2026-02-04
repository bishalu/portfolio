import React, { useRef, useEffect, useState } from 'react'
import { motion, useScroll, useSpring, useTransform } from 'framer-motion'
import { NeuralDust, ActiveGrid, SynapticLightning, RisingEmbers } from './WorldPatterns'

/**
 * WorldManager
 * Controls the atmospheric background ("The World") as the user scrolls.
 * Creates a fluid journey from Ethereal -> Midnight -> Clean -> Warm.
 */
const WorldManager = () => {
  const [activeWorld, setActiveWorld] = useState(0)

  // We'll use a simple intersection observer for lightweight active section detection
  // This is often more reliable for "snap" states than raw scroll mapping for this specific use case
  useEffect(() => {
    const handleScroll = () => {
      // Get all section elements (Removed 'plans' since it was deleted)
      const sections = ['hero', 'vibeset', 'datalens', 'lets-build']
      const sectionElements = sections.map(id => document.getElementById(id))

      // Find the one most visible in viewport
      let maxVisibility = 0
      let currentBestIndex = 0

      sectionElements.forEach((el, index) => {
        if (!el) return

        const rect = el.getBoundingClientRect()
        const viewHeight = window.innerHeight

        // Calculate how much of the element is visible
        const visibleHeight = Math.min(rect.bottom, viewHeight) - Math.max(rect.top, 0)
        const percentVisible = Math.max(0, visibleHeight / viewHeight)

        if (percentVisible > maxVisibility) {
          maxVisibility = percentVisible
          currentBestIndex = index
        }
      })

      // Map section index to World ID
      // 0 (Hero) -> World 0 (Gateway)
      // 1 (Vibeset) -> World 1 (Midnight Console)
      // 2 (DataLens) -> World 2 (Optical Lab)
      // 3 (CTA) -> World 3 (Invitation - kept same for continuity)
      // Note: Plans was removed, so we map the remaining indices.

      let newWorld = 0
      if (currentBestIndex === 0) newWorld = 0
      else if (currentBestIndex === 1) newWorld = 1
      else if (currentBestIndex === 2) newWorld = 2
      else newWorld = 3

      if (newWorld !== activeWorld) {
        setActiveWorld(newWorld)
      }
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    // Initial check
    handleScroll()

    return () => window.removeEventListener('scroll', handleScroll)
  }, [activeWorld])

  return (
    <div className="world-manager-fixed">
      {/* 
        WORLD 0: THE GATEWAY (Hero)
        Ethereal, breathable, light Aurora.
        Colors: Google Blue + Teal (Light)
        Pattern: Neural Dust
      */}
      <motion.div
        className="world-layer world-gateway"
        initial={{ opacity: 1 }}
        animate={{ opacity: activeWorld === 0 ? 1 : 0 }}
        transition={{ duration: 1, ease: "easeInOut" }}
      >
        <NeuralDust isActive={activeWorld === 0} />
      </motion.div>

      {/* 
        WORLD 1: THE CONSOLE (Vibeset)
        Deep, immersive, focus.
        Colors: Midnight Blue + Indigo + Subtle Grid
        Pattern: Active Grid pulses
      */}
      <motion.div
        className="world-layer world-console"
        initial={{ opacity: 0 }}
        animate={{ opacity: activeWorld === 1 ? 1 : 0 }}
        transition={{ duration: 1, ease: "easeInOut" }}
      >
        <div className="console-grid-overlay" />
        <ActiveGrid />
      </motion.div>

      {/* 
        WORLD 2: OPTICAL LAB (DataLens)
        Sharp, clinical, clean.
        Colors: Cool White + Sharp Blue Refractions
        Pattern: Synaptic Lightning (The Request)
      */}
      <motion.div
        className="world-layer world-lab"
        initial={{ opacity: 0 }}
        animate={{ opacity: activeWorld === 2 ? 1 : 0 }}
        transition={{ duration: 1, ease: "easeInOut" }}
      >
        <SynapticLightning isActive={activeWorld === 2} />
      </motion.div>

      {/* 
        WORLD 3: THE HEARTH (CTA)
        Warm, organic, human.
        Colors: Sunset Coral + Moss Green
        Pattern: Rising Embers
      */}
      <motion.div
        className="world-layer world-hearth"
        initial={{ opacity: 0 }}
        animate={{ opacity: activeWorld === 3 ? 1 : 0 }}
        transition={{ duration: 1, ease: "easeInOut" }}
      >
        <RisingEmbers />
      </motion.div>

      <style>{`
        .world-manager-fixed {
          position: fixed;
          top: 0;
          left: 0;
          width: 100vw;
          height: 100vh;
          z-index: -10;
          pointer-events: none;
          overflow: hidden;
          background: var(--background-color); /* Fallback */
        }
        
        .world-layer {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
        }
        
        /* 1. GATEWAY */
        .world-gateway {
          background: 
            radial-gradient(circle at 15% 15%, rgba(66, 133, 244, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 85% 85%, rgba(0, 212, 170, 0.12) 0%, transparent 45%);
        }
        
        /* 2. CONSOLE */
        .world-console {
          background: #0f1115; /* Deep dark base */
        }
        .world-console::before {
          content: '';
          position: absolute;
          inset: 0;
          /* Indigo spotlight */
          background: radial-gradient(circle at 50% 40%, rgba(88, 86, 214, 0.15), transparent 60%);
        }
        .console-grid-overlay {
          position: absolute;
          inset: 0;
          background-size: 50px 50px;
          background-image: linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                            linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
          mask-image: radial-gradient(circle at center, black 40%, transparent 80%);
        }
        
        /* 3. LAB */
        .world-lab {
          background: #fcfcfc; /* Clinical white base */
        }
        .world-lab::before {
          content: '';
          position: absolute;
          inset: 0;
          /* Sharp refraction beams */
          background: 
            linear-gradient(120deg, transparent 30%, rgba(66, 133, 244, 0.04) 45%, rgba(66, 133, 244, 0.08) 50%, rgba(66, 133, 244, 0.04) 55%, transparent 70%);
        }
        
        /* 4. HEARTH */
        .world-hearth {
          background: radial-gradient(ellipse at bottom, rgba(232, 168, 124, 0.15), transparent 70%);
        }
        .world-hearth::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 40%;
          background: linear-gradient(to top, rgba(52, 168, 83, 0.08), transparent);
        }
      `}</style>
    </div>
  )
}

export default WorldManager
