// 1. Import utilities from `astro:content`
import { defineCollection, z } from 'astro:content'

// 2. Import loader(s)
import { glob } from 'astro/loaders'

// 3. Define your collection(s)

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
  }),
})

// Products - the three Vibeset product brands (single source of truth for
// the landing triptych AND /vibeset/[product] pages)
const products = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/products' }),
  schema: z.object({
    name: z.string(),
    eyebrow: z.string(), // lifecycle stage: FIND / FIT / PROVE
    tagline: z.string(),
    description: z.string(),
    accent: z.string(), // Alpenglow family color for product pages
    liveUrl: z.string().optional(), // the real, verified product URL
    liveLabel: z.string().optional(), // CTA label for liveUrl
    blogUrl: z.string().optional(),
    stats: z.array(z.object({ label: z.string(), value: z.string() })).default([]),
    studios: z.array(z.string()).default([]),
    order: z.number().default(0),
  }),
})

// 4. Export collections
export const collections = { publications, products }
