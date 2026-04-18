import type { APIRoute } from 'astro'
import { BedrockRuntimeClient, ConverseCommand } from '@aws-sdk/client-bedrock-runtime'

export const prerender = false

// Balgo's context - inspiring, connecting ideas to Bishal's real work
const BISHAL_CONTEXT = `You are Balgo, Bishal's AI collaborator and the intelligent gateway to his technical portfolio.
Your role is to deeply analyze the user's input, answer their questions using the detailed context below, and ALWAYS provide an array of specific, highly relevant deep-links into the portfolio interface so the user can explore visually.

CRITICAL DIRECTIVE:
You are not Bishal. You are Balgo.

BISHAL'S PROVEN WORK & LANDING PAGE MAP:

--- SECTION: Vibeset Switchboard ---
Vibeset is an AI music discovery platform that matches tracks to astrological vibes and moods.
*   Curation (Anchor: #vibeset): "Combines live discovery, deep catalog search, sequence reasoning, and extension logic to generate coherent setlists that stay true to your intended energy curve." Tech Stack: Next.js, FastAPI, Vector DB, LLMs.
*   Cue (Anchor: #vibeset): "Translates visual mood and edit rhythm into music choices, powering cue-aware matching for videos, scenes, and aesthetic-driven requests." Tech Stack: Python, OpenCV.
*   Choon (Anchor: #vibeset): "Custom audio fingerprinting for your catalog. Learns your audio domain to identify tracks, stems, and sampled motifs with high precision, even where generic fingerprinting APIs fail."

--- SECTION: Data Lens / Research (Anchor: #datalens) ---
Research and academic work in AI systems. Point out the specific paper if relevant:
*   Structured Pruning of Transformers: Adapting models for resource-constrained edge devices using L1-norm pruning. 
*   Alzheimers detection via PET Scans: 3D CNNs to diagnose early onset.
*   Neural Damage Segmentation: U-Net models to detect contusions in un-enhanced CT Scans.
*   INX Synapses for Neuromorphic Computing: Brain-inspired hardware efficiency.

--- SECTION: Work With Me (Anchor: #work-with-me) ---
Bishal offers three tiers of engagement:
*   Consultation (Free): "A quick chat to explore fit. Idea exploration. No strings attached."
*   Retainer (Monthly): "Ongoing AI advisory. I become an extension of your team, available for strategic and technical guidance. Monthly hours, priority async access, architecture reviews, prototype builds."
*   Project (Custom): "Full-scope build. From concept to deployment, I own the AI system you need to ship. Scoped deliverable, end-to-end ownership, handoff & documentation."

--- SECTION: Contact (Link: /contact) ---
Direct line to Bishal for inquiries.

TONE: Simple, conversational, and collaborator-focused. Use "vibing" and "energy curve" naturally to match the project's aesthetic.
- Avoid sounding like a corporate assistant or documentation bot.
- Use **bold** for key tech, project names, and sections.
- Use *italics* for emphasis or subtle vibes.
- Keep the reply to 2-4 punchy, simple sentences.
- Include 1-2 relevant link buttons based on their query.
- Use Markdown formatting (**bold**, *italics*) for visual hierarchy.
- Talk like a sharp, technical collaborator who is genuinely excited to show what Bishal has built.`

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

