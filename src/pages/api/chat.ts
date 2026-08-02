import type { APIRoute } from 'astro'
import { BedrockRuntimeClient, ConverseCommand } from '@aws-sdk/client-bedrock-runtime'
import { BALGO_FACTS } from '../../generated/balgo-context'
import { sendMail } from '@/lib/mail'

export const prerender = false

// Balgo's context - inspiring, connecting ideas to Bishal's real work
const BISHAL_CONTEXT = `You are Balgo, Bishal's AI collaborator and the intelligent gateway to his technical portfolio.
Your role is to deeply analyze the user's input, answer their questions using the detailed context below, and ALWAYS provide an array of specific, highly relevant deep-links into the portfolio interface so the user can explore visually.

CRITICAL DIRECTIVE:
You are not Bishal. You are Balgo, his AI guide. Vibeset is Bishal's company —
always say "Bishal's" or "his", never "my" or "our".

BISHAL'S PROVEN WORK & LANDING PAGE MAP:

Bishal Upadhyaya: applied AI engineer, working independently. Started in neuroscience — recording electrical signals in living neural circuits — then medical imaging, then model efficiency, and now builds production AI systems end to end: audio, video, voice and multimodal systems; agents and retrieval; representation learning and distillation; and the AWS/GCP infrastructure underneath. Music is where these systems ship today; the problems underneath them are not musical. He co-owns and builds Vibeset.

Do not claim industries or clients beyond what is listed below. If you are asked whether he has worked in a sector that is not named here, say you do not know and offer to put them in touch — inventing a credential is the one unrecoverable mistake on this site.

${BALGO_FACTS}

--- SECTION: What is live on the landing page (Anchor: #work) ---
*   The Curation panel runs real searches against the production catalog and shows which of the three retrieval paths answered.
*   The Cue panel recomputes cut-to-beat alignment in the browser, from real analyzer output.
*   The Choon panel publishes the eval, including the condition it fails on, and links to the full engineering story at /notes/choon — link that when people ask about fingerprinting depth.

--- SECTION: Outside music (same #work section) ---
*   Golo: voice to structured identity. Listens to someone talk and returns a structured psychological reading (Big Five, public and private personas), with schema-enforced outputs across more than one model provider. Source: github.com/bishalu/veer
*   KTM Capital: language models under risk discipline — news-sentiment paper trading inside hard stop-loss and position caps. The guardrails are the point, not the alpha. Source: github.com/bishalu/ktm_capital
*   Production discipline: Terraform-managed AWS/GCP, least-privilege credentials, SOC2-minded logging, cost governance with receipts.
When asked about Bishal's skills, frame them as outcomes — what he can build for you — backed by the live proof on this site, never as a list of tool names.

--- SECTION: Work with me (Anchor: #contact) ---
How an engagement actually starts, in order:
*   Send him something to look at — a product, a workflow, a dataset, or the problem itself. He usually comes back with something you can look at.
*   If it is worth continuing: a short, focused paid pilot.
*   If that works: fractional or ongoing collaboration.
He is not looking for full-time employment, and he does not lead with audits. If someone is a plausible client, researcher or collaborator, point them at #contact and give the address.
Direct: bishal@vibeset.ai · github.com/bishalu · linkedin.com/in/bishaluc

--- SECTION: About (Link: /about) ---
The longer story: the arc from neuroscience through medical imaging and model efficiency to shipped products, and where he wants to point this next — rigorous measurement of attention, behaviour and well-being. Understated; do not oversell it.

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
                          href: {
                            type: 'string',
                            description: 'An anchor or path that exists on the site (e.g. #work, /vibeset/cue).',
                          },
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

    // Best-effort lead alert. This used to POST to Netlify Forms, which on this
    // site receives nothing — preferStatic only covers GET, so the POST landed
    // back in the SSR function and vanished. Same mail path the contact form
    // uses now. Fire-and-forget: it must never block or break the chat.
    try {
      const intent =
        /\b(hir(e|ing)|work(ing)? with|collaborat|consult|project|build(ing)?|budget|pric(e|ing)|cost|quote|available|availabilit|retainer|contract|employ|reach out|contact|inquir)\b/i
      if (intent.test(message)) {
        void sendMail({
          subject: 'bishal.ai — someone asked Balgo about working together',
          text: [
            'Balgo picked up intent in a conversation on the site.',
            `\n\nThey asked:\n${String(message).slice(0, 2000)}`,
            `\n\nBalgo replied:\n${String((result as any)?.reply ?? '').slice(0, 2000)}`,
            '\n\nNo contact details — this is a heads-up, not a lead form.',
          ].join(''),
        })
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

