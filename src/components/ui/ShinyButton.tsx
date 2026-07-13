import { cn } from '@/lib/utils'

interface ShinyButtonProps {
  children: React.ReactNode
  className?: string
  href?: string
}

/** Alpenglow pill button (see docs/design/DESIGN.md). Hover lift and press
 *  scale come from the global motion defaults in src/styles/motion.css. */
export const ShinyButton = ({ children, className, href, ...props }: ShinyButtonProps) => {
  const Container = (href ? 'a' : 'button') as 'a'

  return (
    <Container
      href={href}
      className={cn(
        'btn inline-block rounded-full border px-6 py-2 text-sm font-medium tracking-[0.08em] uppercase',
        'border-[var(--hairline-on-paper)] bg-[var(--crimson-deep)] font-[family-name:var(--font-mono)] text-[var(--paper)]',
        'hover:shadow-[var(--glow-crimson)]',
        className,
      )}
      {...props}
    >
      {children}
    </Container>
  )
}
