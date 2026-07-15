import type { APIRoute } from 'astro'
import { BedrockRuntimeClient, ConverseCommand } from '@aws-sdk/client-bedrock-runtime'

export const prerender = false

// Balgo's context - inspiring, connecting ideas to Bishal's real work
const BISHAL_CONTEXT = `You are Balgo, Bishal's AI collaborator and the intelligent gateway to his technical portfolio.
Your role is to deeply analyze the user's input, answer their questions using the detailed context below, and ALWAYS provide an array of specific, highly relevant deep-links into the portfolio interface so the user can explore visually.

CRITICAL DIRECTIVE:
You are not Bishal. You are Balgo, his AI guide. Vibeset is Bishal's company —
always say "Bishal's" or "his", never "my" or "our".

BISHAL'S PROVEN WORK & LANDING PAGE MAP:

Bishal Upadhyaya: AI systems architect. Started in neuroscience (electrical signals in living neural circuits), now takes AI systems from peer-reviewed research to shipped products — agents, agentic RAG/retrieval, model distillation, guardrails for consequential AI, and audit-ready AWS/GCP infra, end to end. Domain-agnostic: he's shipped AI across biotech, medical imaging, finance-style risk, and media. Vibeset (his company) is where this ships in music today — proof of his range, not his only lane. Co-owner of Vibeset.

--- SECTION: Vibeset — his company (Anchor: #vibeset) ---
Vibeset is AI music tooling: one licensed, deeply-tagged catalog, three products spanning the music lifecycle — find it, fit it, prove it. Never mention the catalog's track count.
*   Curation (Anchor: #vibeset, live app: https://vibeset.ai/djapp): AI setlist generation. Describe a vibe — artists, genres, moods, BPM — get DJ-quality setlists: tempo-matched, harmonically compatible, energy-arc aware. Postgres + pgvector, three search modes (SQL / hybrid / semantic embeddings), LLM ensemble finisher. There is a LIVE demo of this right on the landing page (the "Find the vibe" widget in the Vibeset section).
*   Cue (Anchor: #vibeset, live app: https://cue.vibeset.ai): music perfectly synced to picture. Upload a cut; it reads pacing, mood, and moments, then matches licensed music with sync points. FastAPI + Lambda backend (OpenCV, librosa, multi-LLM), Next.js frontend. Live and free for creators.
*   Choon (Anchor: #vibeset, live: https://choon.vibeset.ai): audio fingerprinting + provenance. Hybrid matcher — Shazam-style spectral landmarks for clean audio, a 27.7M-parameter Conformer embedding model (FAISS + temporal alignment) for mangled audio. Benchmarked at 66,000-track scale for a major music label: 76.9% recall@1 under attack, 93.8% on core conditions, 285ms/query on a single CPU core. Full engineering story with charts: /notes/choon (link this when people ask about fingerprinting depth). Plus audio watermarking with C2PA signed manifests, in development.

--- SECTION: Research (Anchor: #research) ---
Four peer-reviewed papers. Deep-link with #paper-<slug>:
*   A Generalization of Continuous Relaxation in Structured Pruning (2023, Nvidia/Thermo Fisher, Anchor: #paper-structured-pruning): extracting smaller, efficient sub-networks from large models. Directly applied in Choon's 27.7M-parameter model.
*   Circumventing neural damage in a C. elegans chemosensory circuit (2021, Cell Systems, Anchor: #paper-neural-damage): genetically engineered synapses restoring circuit function.
*   FDG vs Amyloid PET for Deep Learning Prediction of Alzheimer's (2020, UCSF, Anchor: #paper-alzheimers-pet).
*   INX-18 and INX-19 electrical synapse roles (2019, PLoS Genetics, Anchor: #paper-inx-synapses): the biology of electrical signaling between neurons.

--- SECTION: Beyond music (on the landing page) ---
*   Golo: voice → structured identity. Listens to someone talk, returns a structured psychological profile (Big Five, public/private personas) with schema-enforced outputs across LLM providers.
*   KTM Capital: LLMs under risk discipline — news-sentiment paper-trading inside hard stop-loss and position caps. The point is the guardrails, not the alpha.
*   Production discipline: Terraform-managed AWS/GCP, least-privilege credentials, SOC2-minded logging, cost governance with receipts.
When asked about Bishal's skills, frame them as outcomes (what he can ship for you) backed by the live proof on this site — never as a list of tool names.

--- SECTION: Work With Me (Anchor: #contact) ---
*   Consultation (Free): a quick call to explore fit.
*   Retainer (Monthly): ongoing AI advisory — architecture reviews, prototype builds, an extension of your team.
*   Project (Custom): full-scope build, concept to deployment.
Direct: bishal@vibeset.ai · github.com/bishalu · linkedin.com/in/bishaluc

--- SECTION: About (Link: /about) ---
The longer story and his photo.

TONE: Plain-spoken, confident, first person about Balgo ("I"), third person about Bishal.
- Avoid sounding like a corporate assistant or documentation bot. No hype words.
- Use **bold** for product names and key tech; *italics* sparingly.
- Keep the reply to 2-4 punchy, simple sentences. Be specific: real numbers and real links beat adjectives.
- Include 1-2 relevant link buttons based on their query (use the anchors above).
- If they seem like a potential client, employer, or collaborator, ALWAYS surface the #contact link AND give Bishal's email (bishal@vibeset.ai), and warmly invite them to reach out — that is how they reach Bishal directly.`

export const POST: APIRoute = async ({ request }) => {
  try {
    const body = await request.json()
    const { message } = body

    if (!message || typeof message !== 'string') {
      return new Response(JSON.stringify({ error: 'Message is required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    // BALGO_-prefixed because Lambda strips custom AWS_* env vars entirely.
    // process.env is the runtime source on Netlify functions;
    // import.meta.env covers local `astro dev` (Vite loads .env)
    const accessKeyId = process.env.BALGO_AWS_KEY_ID || import.meta.env.BALGO_AWS_KEY_ID || import.meta.env.AWS_ID
    const secretAccessKey = process.env.BALGO_AWS_SECRET || import.meta.env.BALGO_AWS_SECRET || import.meta.env.AWS_SEC
    const region = process.env.BALGO_AWS_REGION || import.meta.env.BALGO_AWS_REGION || 'us-east-2'

    const client = new BedrockRuntimeClient({
      region,
      credentials: {
        accessKeyId: accessKeyId || '',
        secretAccessKey: secretAccessKey || '',
      },
    })

    const command = new ConverseCommand({
      // gpt-oss emits reasoning before its tool call — give it headroom
      modelId: process.env.BALGO_MODEL_ID || import.meta.env.BALGO_MODEL_ID || 'openai.gpt-oss-120b-1:0',
      messages: [{ role: 'user', content: [{ text: message }] }],
      system: [{ text: BISHAL_CONTEXT }],
      inferenceConfig: {
        maxTokens: 900,
        temperature: 0.7,
      },
      toolConfig: {
        tools: [
          {
            toolSpec: {
              name: 'generate_portfolio_response',
              description: 'Generates a structured conversational response with relevant links to Bishal\'s portfolio.',
              inputSchema: {
                json: {
                  type: 'object',
                  properties: {
                    reply: {
                      type: 'string',
                      description: 'The conversational response text.',
                    },
                    links: {
                      type: 'array',
                      description: 'A list of relevant portfolio links.',
                      items: {
                        type: 'object',
                        properties: {
                          title: { type: 'string', description: 'The text to show on the button.' },
                          href: { type: 'string', description: 'The anchor link or URL (e.g., #vibeset-cue).' },
                          emoji: { type: 'string', description: 'A relevant emoji for the button.' },
                        },
                        required: ['title', 'href', 'emoji'],
                      },
                    },
                  },
                  required: ['reply', 'links'],
                },
              },
            },
          },
        ],
        toolChoice: { any: {} },
      },
    })

    const response = await client.send(command)
    
    // Extract tool use content
    const content = response.output?.message?.content
    if (!content) {
      throw new Error('No content returned from Bedrock')
    }

    const toolUseBlock = content.find((block) => block.toolUse)

    // Prefer the structured tool call; fall back to plain text so a
    // reasoning-model quirk never turns into a user-facing error.
    const result = toolUseBlock?.toolUse
      ? toolUseBlock.toolUse.input
      : { reply: content.find((block) => block.text)?.text ?? "Let's connect directly — bishal@vibeset.ai.", links: [] }

    // Best-effort lead alert: if the visitor signals client/employer/collaborator
    // intent, forward the exchange to the same Netlify Forms inbox as the contact
    // form so it emails Bishal. Fire-and-forget — never blocks or breaks the chat.
    try {
      const intent =
        /\b(hir(e|ing)|work(ing)? with|collaborat|consult|project|build(ing)?|budget|pric(e|ing)|cost|quote|available|availabilit|retainer|contract|employ|reach out|contact|inquir)\b/i
      if (intent.test(message)) {
        const origin = new URL(request.url).origin
        const payload = new URLSearchParams({
          'form-name': 'balgo-lead',
          message: String(message).slice(0, 2000),
          reply: String((result as any)?.reply ?? '').slice(0, 2000),
        })
        void fetch(`${origin}/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: payload.toString(),
        }).catch(() => {})
      }
    } catch {}

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  } catch (error: any) {
    console.error('Chat routing error:', error)
    return new Response(
      JSON.stringify({
        error: error.message || 'Failed to process request through bedrock',
        reply: 'The neural network is initializing. Please try again or contact Bishal directly!',
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } },
    )
  }
}

