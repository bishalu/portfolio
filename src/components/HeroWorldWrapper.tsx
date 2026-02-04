import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

interface Props {
    children: React.ReactNode;
}

export default function HeroWorldWrapper({ children }: Props) {
    const ref = useRef<HTMLDivElement>(null);

    // Track scroll progress relative to this container
    // "start start" = when top of container hits top of viewport
    // "end start" = when bottom of container hits top of viewport
    const { scrollYProgress } = useScroll({
        target: ref,
        offset: ["start start", "end start"]
    });

    // Unique World Effect: "The Ethereal Gateway"
    // As user scrolls down, the gateway fades, blurs, and scales down slightly,
    // creating a sense of leaving one dimension for another.

    const opacity = useTransform(scrollYProgress, [0, 0.5, 0.8], [1, 0.8, 0]);
    const scale = useTransform(scrollYProgress, [0, 0.8], [1, 0.9]);
    const blur = useTransform(scrollYProgress, [0, 0.6], ["0px", "10px"]);
    const y = useTransform(scrollYProgress, [0, 1], ["0%", "20%"]); // Parallax lag

    return (
        <motion.div
            ref={ref}
            style={{ opacity, scale, filter: `blur(${blur})`, y }}
            className="hero-world-wrapper"
        >
            {children}
        </motion.div>
    );
}
