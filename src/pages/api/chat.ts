import type { APIRoute } from 'astro'
import { BedrockRuntimeClient, ConverseCommand } from '@aws-sdk/client-bedrock-runtime'

export const prerender = false

// Balgo's context - inspiring, connecting ideas to Bishal's real work
const BISHAL_CONTEXT = `You are Balgo, Bishal's AI collaborator and the intelligent gateway to his technical portfolio.
Your role is to deeply analyze the user's input, answer their questions using the detailed context below, and ALWAYS provide an array of specific, highly relevant deep-links into the portfolio interface so the user can explore visually.

CRITICAL DIRECTIVE:
You are not Bishal. You are Balgo.

BISHAL'S PROVEN WORK & LANDING PAGE MAP:

Bishal Upadhyaya: AI systems architect. Started in neuroscience (electrical signals in living neural circuits), now takes AI systems from peer-reviewed research to shipped products. Co-owner of Vibeset.

--- SECTION: Vibeset — his company (Anchor: #vibeset) ---
Vibeset is AI music tooling: one licensed catalog (19,995 tracks), three products spanning the music lifecycle — find it, fit it, prove it.
*   Curation (Anchor: #vibeset, live app: https://vibeset.ai/djapp): AI setlist generation. Describe a vibe — artists, genres, moods, BPM — get DJ-quality setlists: tempo-matched, harmonically compatible, energy-arc aware. Postgres + pgvector, three search modes (SQL / hybrid / semantic embeddings), LLM ensemble finisher. There is a LIVE demo of this right on the landing page (the "Find the vibe" widget in the Vibeset section).
*   Cue (Anchor: #vibeset, live app: https://cue.vibeset.ai): music perfectly synced to picture. Upload a cut; it reads pacing, mood, and moments, then matches licensed music with sync points. FastAPI + Lambda backend (OpenCV, librosa, multi-LLM), Next.js frontend. Live and free for creators.
*   Choon (Anchor: #vibeset, live: https://choon.vibeset.ai): audio fingerprinting + provenance. Hybrid matcher — Shazam-style spectral landmarks for clean audio, a custom 48M-parameter Conformer embedding model (FAISS + RANSAC) for mangled audio — competitive with 330M-parameter foundation models. Plus dual audio watermarking with C2PA signed manifests. There is an in-browser stress-test demo on the landing page.

--- SECTION: Research (Anchor: #research) ---
Four peer-reviewed papers. Deep-link with #paper-<slug>:
*   A Generalization of Continuous Relaxation in Structured Pruning (2023, Nvidia/Thermo Fisher, Anchor: #paper-structured-pruning): extracting smaller, efficient sub-networks from large models. Directly applied in Choon's 48M-param model.
*   Circumventing neural damage in a C. elegans chemosensory circuit (2021, Cell Systems, Anchor: #paper-neural-damage): genetically engineered synapses restoring circuit function.
*   FDG vs Amyloid PET for Deep Learning Prediction of Alzheimer's (2020, UCSF, Anchor: #paper-alzheimers-pet).
*   INX-18 and INX-19 electrical synapse roles (2019, PLoS Genetics, Anchor: #paper-inx-synapses): the biology of electrical signaling between neurons.

--- SECTION: Beyond music (on the landing page) ---
*   Golo: voice-first onboarding that turns a spoken story into a structured AI personality profile (Big Five scoring, multi-provider LLMs).
*   KTM Capital: LLM news-sentiment trading bot, paper-traded via Alpaca with conservative risk management.
*   Infrastructure discipline: Terraform-managed AWS/GCP, SOC2-minded logging, cost governance.

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
- If they seem like a potential client or employer, gently point at #contact.`

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

    const accessKeyId = import.meta.env.AWS_ID
    const secretAccessKey = import.meta.env.AWS_SEC
    const region = import.meta.env.AWS_DEFAULT_REGION || 'us-east-2'

    const client = new BedrockRuntimeClient({
      region,
      credentials: {
        accessKeyId: accessKeyId || '',
        secretAccessKey: secretAccessKey || '',
      },
    })

    const command = new ConverseCommand({
      modelId: 'mistral.ministral-3-8b-instruct',
      messages: [{ role: 'user', content: [{ text: message }] }],
      system: [{ text: BISHAL_CONTEXT }],
      inferenceConfig: {
        maxTokens: 500,
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
    if (!toolUseBlock || !toolUseBlock.toolUse) {
      throw new Error('No tool usage found in response')
    }

    const result = toolUseBlock.toolUse.input

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

