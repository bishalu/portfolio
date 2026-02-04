import React, { useEffect, useRef } from 'react'

/* ═══════════════════════════════════════════════════════════════════════════
   1. NEURAL DUST (HERO)
   Slow floating particles connecting to form ideas.
   ═══════════════════════════════════════════════════════════════════════════ */
export const NeuralDust = ({ isActive }: { isActive: boolean }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        let animationId: number
        let particles: { x: number; y: number; vx: number; vy: number; size: number }[] = []

        const resize = () => {
            canvas.width = window.innerWidth
            canvas.height = window.innerHeight
        }
        window.addEventListener('resize', resize)
        resize()

        // Init particles
        const particleCount = 40
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.2,
                vy: (Math.random() - 0.5) * 0.2,
                size: Math.random() * 2 + 0.5,
            })
        }

        const draw = () => {
            if (!isActive) {
                // Pause loop if not active to save battery
                animationId = requestAnimationFrame(draw)
                return
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height)
            ctx.fillStyle = 'rgba(66, 133, 244, 0.4)' // Google Blue-ish

            particles.forEach((p) => {
                p.x += p.vx
                p.y += p.vy

                // Wrap
                if (p.x < 0) p.x = canvas.width
                if (p.x > canvas.width) p.x = 0
                if (p.y < 0) p.y = canvas.height
                if (p.y > canvas.height) p.y = 0

                ctx.beginPath()
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
                ctx.fill()
            })

            animationId = requestAnimationFrame(draw)
        }

        draw()

        return () => {
            window.removeEventListener('resize', resize)
            cancelAnimationFrame(animationId)
        }
    }, [isActive])

    return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
}

/* ═══════════════════════════════════════════════════════════════════════════
   2. ACTIVE GRID (VIBESET)
   CSS Grid with random pulsing cells.
   ═══════════════════════════════════════════════════════════════════════════ */
export const ActiveGrid = () => {
    // Generate a fixed set of "active" cells to pulse
    // We use inline styles for random delays to avoid massive CSS
    const cells = Array.from({ length: 20 }).map((_, i) => ({
        id: i,
        top: `${Math.random() * 100}%`,
        left: `${Math.random() * 100}%`,
        delay: `${Math.random() * 5}s`,
        duration: `${3 + Math.random() * 4}s`,
    }))

    return (
        <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none">
            {/* Base Grid Line Pattern (CSS Radial/Linear) is in WorldManager styles. 
          This component adds the "Data Flow" pulses */}
            {cells.map((cell) => (
                <div
                    key={cell.id}
                    className="absolute w-1 h-1 bg-white rounded-full opacity-0 animate-pulse-data"
                    style={{
                        top: cell.top,
                        left: cell.left,
                        animationDelay: cell.delay,
                        animationDuration: cell.duration,
                        boxShadow: '0 0 8px 2px rgba(88, 86, 214, 0.6)',
                    }}
                />
            ))}
            <style>{`
        @keyframes pulse-data {
          0%, 100% { opacity: 0; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.5); }
        }
        .animate-pulse-data {
          animation-name: pulse-data;
          animation-iteration-count: infinite;
        }
      `}</style>
        </div>
    )
}

/* ═══════════════════════════════════════════════════════════════════════════
   3. SYNAPTIC LIGHTNING (RESEARCH)
   Ported High-Performance Lightning Canvas.
   ═══════════════════════════════════════════════════════════════════════════ */
export const SynapticLightning = ({ isActive }: { isActive: boolean }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        let animationId: number

        // Config
        const CONFIG = {
            colors: [
                'rgba(234, 67, 53, 0.7)',
                'rgba(251, 188, 4, 0.7)',
                'rgba(52, 168, 83, 0.7)',
                'rgba(66, 133, 244, 0.7)',
                'rgba(0, 212, 170, 0.7)',
            ],
            minInterval: 500,
            maxInterval: 2000,
            maxBolts: 10,
        }

        interface Bolt {
            segments: { x1: number; y1: number; x2: number; y2: number }[]
            color: string
            opacity: number
            fadeSpeed: number
        }

        let bolts: Bolt[] = []

        const resize = () => {
            canvas.width = window.innerWidth
            canvas.height = window.innerHeight
        }
        window.addEventListener('resize', resize)
        resize()

        // --- Lightning Math (Recursive) ---
        function generateBolt(x1: number, y1: number, x2: number, y2: number, depth = 0): any[] {
            const segments = []
            if (depth >= 4) {
                segments.push({ x1, y1, x2, y2 })
                return segments
            }
            const midX = (x1 + x2) / 2
            const midY = (y1 + y2) / 2
            const dx = x2 - x1
            const dy = y2 - y1
            const length = Math.sqrt(dx * dx + dy * dy)
            const perpX = -dy / length
            const perpY = dx / length
            const displacement = (Math.random() - 0.5) * length * 0.3
            const newMidX = midX + perpX * displacement
            const newMidY = midY + perpY * displacement

            segments.push(...generateBolt(x1, y1, newMidX, newMidY, depth + 1))
            segments.push(...generateBolt(newMidX, newMidY, x2, y2, depth + 1))

            if (depth < 3 && Math.random() < 0.5) {
                const branchAngle = (Math.random() > 0.5 ? 1 : -1) * (20 + Math.random() * 25) * (Math.PI / 180)
                const mainAngle = Math.atan2(dy, dx)
                const finalAngle = mainAngle + branchAngle
                const branchLength = length * 0.4
                segments.push(...generateBolt(newMidX, newMidY, newMidX + Math.cos(finalAngle) * branchLength, newMidY + Math.sin(finalAngle) * branchLength, depth + 2))
            }
            return segments
        }

        function createBolt() {
            if (bolts.length > CONFIG.maxBolts) return
            const x1 = Math.random() * canvasRef.current!.width
            const y1 = Math.random() * canvasRef.current!.height
            const angle = Math.random() * Math.PI * 2
            const len = 50 + Math.random() * 100
            const x2 = x1 + Math.cos(angle) * len
            const y2 = y1 + Math.sin(angle) * len

            bolts.push({
                segments: generateBolt(x1, y1, x2, y2),
                color: CONFIG.colors[Math.floor(Math.random() * CONFIG.colors.length)],
                opacity: 1,
                fadeSpeed: 0.02,
            })
        }

        // --- Loop ---
        let lastBoltTime = 0
        let nextBoltTime = 0

        const draw = (timestamp: number) => {
            if (!isActive) {
                // Just clear and wait
                ctx?.clearRect(0, 0, canvas.width, canvas.height)
                bolts = []
                animationId = requestAnimationFrame(draw)
                return
            }

            // Create new bolt?
            if (timestamp - lastBoltTime > nextBoltTime) {
                createBolt()
                lastBoltTime = timestamp
                nextBoltTime = CONFIG.minInterval + Math.random() * CONFIG.maxInterval
            }

            ctx?.clearRect(0, 0, canvas.width, canvas.height)

            // Update & Draw
            bolts = bolts.filter(b => b.opacity > 0)
            bolts.forEach(bolt => {
                ctx!.save()
                ctx!.globalAlpha = bolt.opacity
                ctx!.strokeStyle = bolt.color
                ctx!.lineWidth = 2
                ctx!.shadowColor = bolt.color
                ctx!.shadowBlur = 10
                ctx!.beginPath()
                bolt.segments.forEach(s => {
                    ctx!.moveTo(s.x1, s.y1)
                    ctx!.lineTo(s.x2, s.y2)
                })
                ctx!.stroke()
                ctx!.restore()

                bolt.opacity -= bolt.fadeSpeed
            })

            animationId = requestAnimationFrame(draw)
        }

        animationId = requestAnimationFrame(draw)

        return () => {
            window.removeEventListener('resize', resize)
            cancelAnimationFrame(animationId)
        }
    }, [isActive])

    return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
}

/* ═══════════════════════════════════════════════════════════════════════════
   4. RISING EMBERS (INVITATION)
   Warm organic bubbles rising.
   ═══════════════════════════════════════════════════════════════════════════ */
export const RisingEmbers = () => {
    const embers = Array.from({ length: 15 }).map((_, i) => ({
        id: i,
        left: `${Math.random() * 100}%`,
        delay: `${Math.random() * 10}s`,
        duration: `${10 + Math.random() * 10}s`,
        size: `${Math.random() * 100 + 50}px`
    }))

    return (
        <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none">
            {embers.map((ember) => (
                <div
                    key={ember.id}
                    className="absolute rounded-full bg-gradient-to-t from-[rgba(232,168,124,0.1)] to-transparent animate-rise"
                    style={{
                        left: ember.left,
                        bottom: '-20%',
                        width: ember.size,
                        height: ember.size,
                        animationDelay: ember.delay,
                        animationDuration: ember.duration,
                    }}
                />
            ))}
            <style>{`
        @keyframes rise {
          0% { transform: translateY(0) scale(0.9); opacity: 0; }
          20% { opacity: 0.5; }
          100% { transform: translateY(-120vh) scale(1.1); opacity: 0; }
        }
        .animate-rise {
          animation-name: rise;
          animation-iteration-count: infinite;
          animation-timing-function: linear;
        }
      `}</style>
        </div>
    )
}
