// 1. Import utilities from `astro:content`
import { defineCollection, z } from 'astro:content'

// 2. Import loader(s)
import { glob } from 'astro/loaders'

// 3. Define your collection(s)
//
// These collections are the SINGLE source of truth. Everything downstream is
// generated from them: the landing sections, the product pages, the
// schema.org JSON-LD, and Balgo's system prompt. Product facts used to live
// in four hand-maintained copies and had measurably drifted apart — if you
// find yourself retyping a number that already exists here, stop and generate
// it instead.

// Publications - Research papers and whitepapers
const publications = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    abstract: z.string(),
    year: z.number(),
    journal: z.string().optional(), // Journal or conference name
    doi: z.string().optional(), // Digital Object Identifier
    pdf: z.string().optional(), // Link to PDF
    coauthors: z.array(z.string()).default([]),
    tags: z.array(z.string()).default([]),
    /** Slug of the product this paper ships inside, if any. Drives the
     *  research → product spine (e.g. structured pruning → Choon's 27.7M model). */
    appliedIn: z.string().optional(),
    appliedNote: z.string().optional(),
  }),
})

/**
 * A claim the site makes, with a pointer to what backs it.
 *
 * `status` is the honest bit:
 *   shipped  — running in production today. MUST carry evidence; the build
 *              fails without it (scripts/check-claims.mjs).
 *   shipping — finished or all but, with a date. Needs evidence AND eta.
 *              Balgo may speak of it in the near future tense ("lands Friday"),
 *              which is the one honest way to answer a prospect asking about
 *              something that is real but not yet live.
 *   building — real work, in flight, no date. Renders normally and is listed
 *              in the claims report so it can't be forgotten.
 *
 * This is an internal engineering surface, not a public label. Nothing about
 * it changes what a visitor reads.
 */
const claim = z.object({
  claim: z.string(),
  evidence: z.string().optional(),
  evidenceUrl: z.string().optional(),
  status: z.enum(['shipped', 'shipping', 'building']).default('shipped'),
  /** Go-live date. Required when status is `shipping` — a date with no day on
      it is a hope, and Balgo would repeat it to a prospect as a commitment.
      Coerced because YAML parses a bare 2026-08-09 into a Date, and everything
      downstream wants the ISO string. */
  eta: z
    .union([z.string(), z.date()])
    .transform((v) => (v instanceof Date ? v.toISOString().slice(0, 10) : v))
    .optional(),
})

// Capabilities - the individual systems behind each product. Replaces the old
// `studios` collection. Deliberately NO closed id enum: that enum is why the
// count was hardcoded ("Seven") in two places and went stale at eight.
const capabilities = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/capabilities' }),
  schema: z.object({
    name: z.string(),
    tagline: z.string(),
    /** Which product this belongs to — must match a products slug. */
    product: z.enum(['curation', 'cue', 'choon']),
    features: z.array(z.string()).default([]),
    techStack: z.array(z.string()).default([]),
    order: z.number().default(0),
  }),
})

// Products - the three Vibeset product brands (source of truth for the
// landing triptych AND /vibeset/[product] pages)
const products = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/products' }),
  schema: z.object({
    name: z.string(),
    eyebrow: z.string(), // lifecycle stage: FIND / FIT / PROVE
    tagline: z.string(),
    description: z.string(),
    accent: z.string(), // Alpenglow family color — load-bearing, governs contrast
    /** The product's real Vibeset hue. Licensed for at most three non-text
     *  uses on its own page (docs/design/DESIGN.md §5). Never for text or
     *  controls — that would make the portfolio read as a Vibeset subsite. */
    quote: z.object({ hue: z.string(), name: z.string() }).optional(),
    liveUrl: z.string().optional(), // the real, verified product URL
    liveLabel: z.string().optional(), // CTA label for liveUrl
    blogUrl: z.string().optional(),
    stats: z.array(z.object({ label: z.string(), value: z.string() })).default([]),
    claims: z.array(claim).default([]),
    order: z.number().default(0),
  }),
})

// 4. Export collections
export const collections = { publications, capabilities, products }
