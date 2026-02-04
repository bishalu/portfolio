/**
 * Particle Burst Animation System
 * Creates magical particle explosion effects using Canvas 2D API
 */

import gsap from 'gsap';

interface ParticleConfig {
    x: number;
    y: number;
    vx: number;
    vy: number;
    life: number;
    maxLife: number;
    size: number;
    rotation: number;
    rotationSpeed: number;
    type: 'star' | 'note' | 'sparkle';
    color: string;
}

export class ParticleBurstSystem {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private particles: ParticleConfig[] = [];
    private animationId: number | null = null;
    private isActive: boolean = false;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas 2D context not available');
        this.ctx = context;
    }

    /**
     * Trigger particle burst from a specific point
     */
    public burst(centerX: number, centerY: number, particleCount: number = 100) {
        this.particles = [];
        this.isActive = true;

        // Generate particles with varied properties
        for (let i = 0; i < particleCount; i++) {
            const angle = (Math.PI * 2 * i) / particleCount + (Math.random() - 0.5) * 0.5;
            const speed = 200 + Math.random() * 400; // px/s

            // Convert to velocity components
            const vx = Math.cos(angle) * speed;
            const vy = Math.sin(angle) * speed;

            // Particle type distribution
            const rand = Math.random();
            let type: 'star' | 'note' | 'sparkle';
            let color: string;

            if (rand < 0.4) {
                // 40% stars (gold/white)
                type = 'star';
                color = Math.random() > 0.5 ? '#FBBC04' : '#FFFFFF';
            } else if (rand < 0.7) {
                // 30% musical notes (teal/purple)
                type = 'note';
                color = Math.random() > 0.5 ? '#00D4AA' : '#9334E9';
            } else {
                // 30% sparkles (various colors)
                type = 'sparkle';
                const colors = ['#4285F4', '#00D4AA', '#FBBC04', '#FFFFFF'];
                color = colors[Math.floor(Math.random() * colors.length)];
            }

            this.particles.push({
                x: centerX,
                y: centerY,
                vx,
                vy,
                life: 1.0,
                maxLife: 800 + Math.random() * 400, // ms
                size: 3 + Math.random() * 5,
                rotation: Math.random() * Math.PI * 2,
                rotationSpeed: (Math.random() - 0.5) * 10,
                type,
                color
            });
        }

        // Start animation loop
        this.animate();
    }

    /**
     * Animation loop
     */
    private animate() {
        if (!this.isActive || this.particles.length === 0) {
            this.isActive = false;
            return;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const now = performance.now();
        const deltaTime = 16 / 1000; // Approximate 60fps

        // Update and draw particles
        this.particles = this.particles.filter(p => {
            // Physics update
            p.x += p.vx * deltaTime;
            p.y += p.vy * deltaTime;

            // Apply gravity
            p.vy += 200 * deltaTime;

            // Rotation
            p.rotation += p.rotationSpeed * deltaTime;

            // Life decay
            p.life -= deltaTime / (p.maxLife / 1000);

            // Remove dead particles
            if (p.life <= 0) return false;

            // Draw particle
            this.drawParticle(p);

            return true;
        });

        if (this.particles.length > 0) {
            this.animationId = requestAnimationFrame(() => this.animate());
        } else {
            this.isActive = false;
        }
    }

    /**
     * Draw individual particle based on type
     */
    private drawParticle(p: ParticleConfig) {
        const alpha = Math.max(0, Math.min(1, p.life));
        const scale = 0.3 + p.life * 0.7; // Shrink as it fades

        this.ctx.save();
        this.ctx.translate(p.x, p.y);
        this.ctx.rotate(p.rotation);
        this.ctx.globalAlpha = alpha;

        // Motion blur trail effect
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = p.color;

        if (p.type === 'star') {
            this.drawStar(p.size * scale, p.color);
        } else if (p.type === 'note') {
            this.drawMusicNote(p.size * scale, p.color);
        } else {
            this.drawSparkle(p.size * scale, p.color);
        }

        this.ctx.restore();
    }

    /**
     * Draw a star shape
     */
    private drawStar(size: number, color: string) {
        this.ctx.fillStyle = color;
        this.ctx.beginPath();

        const spikes = 5;
        const outerRadius = size;
        const innerRadius = size * 0.5;

        for (let i = 0; i < spikes * 2; i++) {
            const radius = i % 2 === 0 ? outerRadius : innerRadius;
            const angle = (Math.PI * i) / spikes;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;

            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }

        this.ctx.closePath();
        this.ctx.fill();
    }

    /**
     * Draw a musical note symbol
     */
    private drawMusicNote(size: number, color: string) {
        this.ctx.fillStyle = color;
        this.ctx.font = `${size * 4}px Arial`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('♪', 0, 0);
    }

    /**
     * Draw a sparkle/diamond
     */
    private drawSparkle(size: number, color: string) {
        this.ctx.fillStyle = color;
        this.ctx.beginPath();

        // Diamond shape
        this.ctx.moveTo(0, -size);
        this.ctx.lineTo(size, 0);
        this.ctx.lineTo(0, size);
        this.ctx.lineTo(-size, 0);
        this.ctx.closePath();
        this.ctx.fill();

        // Add glow center
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.globalAlpha = 0.6;
        this.ctx.beginPath();
        this.ctx.arc(0, 0, size * 0.3, 0, Math.PI * 2);
        this.ctx.fill();
    }

    /**
     * Stop animation and clean up
     */
    public stop() {
        this.isActive = false;
        this.particles = [];
        if (this.animationId !== null) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}


/**
 * Fragment Dispersal System
 * Creates voronoi-style fragmentation effect for cover reveal
 */
export class FragmentDispersalSystem {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private fragments: any[] = [];

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas 2D context not available');
        this.ctx = context;
    }

    /**
     * Create and animate hexagonal fragments
     */
    public disperse(callback?: () => void) {
        const hexSize = 60;
        const rows = Math.ceil(this.canvas.height / (hexSize * 1.5)) + 1;
        const cols = Math.ceil(this.canvas.width / (hexSize * Math.sqrt(3))) + 1;

        this.fragments = [];

        // Generate hexagonal grid
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const x = col * hexSize * Math.sqrt(3) + (row % 2) * hexSize * Math.sqrt(3) / 2;
                const y = row * hexSize * 1.5;

                // Calculate distance from center for stagger timing
                const centerX = this.canvas.width / 2;
                const centerY = this.canvas.height / 2;
                const distFromCenter = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
                const delay = (distFromCenter / 500) * 0.3; // Max 300ms stagger

                this.fragments.push({
                    x,
                    y,
                    size: hexSize,
                    angle: Math.atan2(y - centerY, x - centerX),
                    rotation: 0,
                    scale: 1,
                    opacity: 1,
                    delay
                });
            }
        }

        // Animate fragments with GSAP
        const timeline = gsap.timeline({
            onComplete: () => {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                if (callback) callback();
            }
        });

        this.fragments.forEach(frag => {
            const distance = 300 + Math.random() * 200;
            const targetX = frag.x + Math.cos(frag.angle) * distance;
            const targetY = frag.y + Math.sin(frag.angle) * distance;

            timeline.to(frag, {
                x: targetX,
                y: targetY,
                rotation: (Math.random() - 0.5) * 720,
                scale: 0,
                opacity: 0,
                duration: 0.8,
                ease: 'expo.in',
                onUpdate: () => this.render()
            }, frag.delay);
        });
    }

    /**
     * Render all fragments
     */
    private render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.fragments.forEach(frag => {
            this.ctx.save();
            this.ctx.translate(frag.x, frag.y);
            this.ctx.rotate(frag.rotation * Math.PI / 180);
            this.ctx.globalAlpha = frag.opacity;
            this.ctx.scale(frag.scale, frag.scale);

            // Draw hexagon
            this.drawHexagon(frag.size);

            this.ctx.restore();
        });
    }

    /**
     * Draw hexagon shape
     */
    private drawHexagon(size: number) {
        this.ctx.fillStyle = 'rgba(20, 20, 25, 0.95)';
        this.ctx.strokeStyle = 'rgba(66, 133, 244, 0.3)';
        this.ctx.lineWidth = 1;

        this.ctx.beginPath();
        for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i;
            const x = Math.cos(angle) * size;
            const y = Math.sin(angle) * size;

            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();
    }
}
