// 1. Import utilities from `astro:content`
import { defineCollection, z } from 'astro:content'

// 2. Import loader(s)
import { glob } from 'astro/loaders'

// 3. Define your collection(s)

// Projects - Currently just Vibeset (the flagship)
const projects = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    category: z.enum(['perception', 'memory', 'logic', 'action']).default('logic'),
    featured: z.boolean().default(false),
    image: z.string().optional(),
    link: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
})

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

// Studios - VibeSet's arsenal of AI tools
const studios = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/studios' }),
  schema: z.object({
    id: z.enum([
      'live-search',
      'db-search',
      'optic-to-audio',
      'set-extender',
      'video-sync',
      'audio-fingerprint',
      'logic-flow',
    ]),
    name: z.string(),
    tagline: z.string(),
    description: z.string(),
    accent: z.string(), // CSS color value (hex, oklch, etc.)
    icon: z.string().optional(), // Path to SVG or loopable media
    features: z.array(z.string()).default([]),
    techStack: z.array(z.string()).default([]),
    status: z.enum(['live', 'beta', 'coming-soon']).default('live'),
    order: z.number().default(0),
  }),
})

// 4. Export collections
export const collections = { projects, publications, studios }

