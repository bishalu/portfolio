/**
 * Signal reveal — scroll entrances for [data-reveal] elements.
 * Spec: docs/design/MOTION.md ("the pulse arrives").
 *
 * - Elements with [data-reveal] animate in once, at ~30% visibility.
 * - A parent with [data-reveal-stagger] delays its [data-reveal] children
 *   in reading order: 40ms per item, capped at 320ms.
 * - Under reduced motion, motion.css renders everything instantly; this
 *   script still adds the class so state stays consistent.
 * - Re-runs on Astro view transitions (astro:page-load).
 */
const STEP_MS = 40
const CAP_MS = 320

function init() {
  document.documentElement.classList.remove('no-js')

  const revealed = new WeakSet<Element>()

  // Assign stagger delays within each stagger group
  for (const group of document.querySelectorAll('[data-reveal-stagger]')) {
    const children = group.querySelectorAll<HTMLElement>('[data-reveal]')
    children.forEach((el, i) => {
      el.style.setProperty('--reveal-delay', `${Math.min(i * STEP_MS, CAP_MS)}ms`)
    })
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting && !revealed.has(entry.target)) {
          revealed.add(entry.target)
          entry.target.classList.add('is-revealed')
          observer.unobserve(entry.target)
        }
      }
    },
    { threshold: 0.3, rootMargin: '0px 0px -5% 0px' },
  )

  for (const el of document.querySelectorAll('[data-reveal]')) {
    // Already in view on load (e.g. hero) → reveal immediately, no observer flash
    const rect = el.getBoundingClientRect()
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      revealed.add(el)
      el.classList.add('is-revealed')
    } else {
      observer.observe(el)
    }
  }
}

document.addEventListener('astro:page-load', init)
// Fallback if view transitions are ever removed
if (document.readyState !== 'loading') init()
else document.addEventListener('DOMContentLoaded', init, { once: true })
