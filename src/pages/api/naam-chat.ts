/**
 * The free-form path's model call (docs/design/DESIGN.md §4, P9).
 *
 * WHY it is an Astro SSR route and not a netlify/function: the Netlify adapter
 * compiles these into the same Lambda, so they get the same ambient Blobs
 * context, and unlike netlify/functions they also see `import.meta.env` and
 * work under plain `astro dev`.
 *
 * THE GROUNDING GUARANTEE LIVES HERE, not in the prompt. The browser's matcher
 * chose a pool; this route resolves those ids against the document's own rows,
 * hands the model that pool and nothing else, and drops every id that comes
 * back from outside it. A name the model invents cannot reach the page,
 * because the page draws its cards from the local dataset.
 *
 * THREE THINGS THIS FIXES ABOUT src/pages/api/chat.ts, which is otherwise the
 * shape being copied:
 *   1. `client.send()` is wrapped in an AbortController at 8s. chat.ts has no
 *      timeout at all, so a hung Bedrock call blocks until the Lambda limit.
 *   2. Every failure returns HTTP 200 with `degraded: true`. chat.ts returns a
 *      500 with an apology in it. `reason` is a signal the page reads to pick
 *      which honest line to show — DESIGN.md §4 rule 4 — not an exception.
 *      Nothing has been rendered when this route is called: the agent leads
 *      now, so a failure here is a failure the visitor sees, and `reason` is
 *      the whole of what they are told.
 *   3. Nothing the model returns is trusted: ids outside the pool are dropped,
 *      picks are capped, the reply must be a string, and it is truncated.
 *      chat.ts pipes its reply into innerHTML on the other end (Hero.astro:491
 *      and :559). The /naam client renders this through JSX interpolation.
 *
 * Credentials are BALGO_-prefixed because Lambda strips inbound AWS_* vars.
 */
import type { APIRoute } from 'astro'
import {
  BedrockRuntimeClient,
  ConverseCommand,
  type Message,
  type ToolInputSchema,
} from '@aws-sdk/client-bedrock-runtime'
import {
  buildSearchResult,
  buildSystemPrompt,
  buildUserTurn,
  coerceModelReply,
  NAAM_MAX_PICKS,
  NAAM_SEARCH_SCHEMA,
  NAAM_SEARCH_TOOL,
  NAAM_TOOL_NAME,
  NAAM_TOOL_SCHEMA,
} from '@/lib/naam/prompt'
import { retrieve } from '@/lib/naam/match'
import { ceilinged, clientIp, isRowId, json, loadCoreRows, rateLimited, tidy } from '@/lib/naam/server'
import type { NaamRow } from '@/types/naam'

export const prerender = false

/** Well short of any Lambda limit: a slower answer is a worse answer anyway. */
const TIMEOUT_MS = 8000
/**
 * The whole request's budget, model call included. The dataset fetch has its
 * own 6s timeout in front of the model call, so on a cold instance 6 + 8 = 14s
 * of server work sat behind a client that gives up at 12s (CHAT_TIMEOUT_MS, in
 * src/lib/naam/ask.ts) and the numbers did not describe the system. One
 * deadline taken at entry fixes that: a warm instance still gets the full 8s, a
 * cold one gets what is left, and nothing outlives the client waiting for it.
 * That mattered less when the page had already answered LOCAL and this call was
 * an upgrade; the agent leads now, so an overrun here is a visitor watching a
 * lamp gutter, and the deadline is the only thing standing between them and it.
 */
const BUDGET_MS = 11_000
/** Per warm instance — best effort, not a wall (netlify/functions/vibeset-demo.ts). */
const RATE_LIMIT_PER_MIN = 12
/**
 * The process-wide wall. This route is an unauthenticated call to a paid model
 * and a per-ip limit is no defence against a distributed flood, which is the
 * shape a spend attack actually has.
 */
const CEILING_PER_MIN = 60
/**
 * Reasoning tokens and answer tokens come out of one budget on gpt-oss, so
 * this is not "how long may the reply be" — the reply is capped at 700 chars
 * by the prompt and at MAX_REPLY_CHARS here. It is headroom for the thinking
 * that precedes it. At 900 with `reasoning_effort: 'low'` the ceiling is never
 * approached; the number is what keeps a hard ask from dying mid-thought
 * rather than what it usually costs.
 */
const MAX_TOKENS = 2200
/** The prompt caps replies at 700; this is the outer wall in case that changes. */
const MAX_REPLY_CHARS = 1200
const POOL_MAX = 60
const ASK_MAX = 400

/**
 * The SDK types a tool schema as its own recursive DocumentType, which a
 * frozen `as const` object literal cannot structurally satisfy. The shape is
 * identical; the cast is the whole of the difference.
 */
const TOOL_SCHEMA = { json: NAAM_TOOL_SCHEMA } as unknown as ToolInputSchema
const SEARCH_SCHEMA = { json: NAAM_SEARCH_SCHEMA } as unknown as ToolInputSchema
/**
 * How many times the model may go back to the document before it must answer.
 * Two is the number: one round to translate a concept it could not find, one
 * more if that was also thin. A third round is a model going in circles, and
 * every round is a full Bedrock turn against an 11s budget.
 */
const MAX_SEARCH_ROUNDS = 2
/** Rows returned per search term. Enough to choose from, small enough to read. */
const SEARCH_LIMIT = 10

/** Never a 500. `reason` picks which honest line the page shows. */
function degraded(reason: string): Response {
  return json({ degraded: true, reason, reply: null, pickIds: [] })
}

export const POST: APIRoute = async (context) => {
  const deadline = Date.now() + BUDGET_MS
  const { request } = context

  let payload: unknown
  try {
    payload = await request.json()
  } catch {
    return degraded('bad-request')
  }

  const body = (payload ?? {}) as { ask?: unknown; poolIds?: unknown; absent?: unknown }
  const ask = tidy(body.ask, ASK_MAX)
  if (!ask) return degraded('bad-request')

  /**
   * Names the visitor typed that the document does not contain. Untrusted like
   * everything else on this request: tidied, length-capped, and capped in
   * count, because it is interpolated into the prompt. It carries no authority
   * — the pool still decides what may be named.
   */
  const absent = (Array.isArray(body.absent) ? body.absent : [])
    .map((name) => tidy(name, 40))
    .filter((name): name is string => Boolean(name))
    .slice(0, 3)

  const submitted = Array.isArray(body.poolIds) ? body.poolIds : []
  const poolIds = [...new Set(submitted.filter(isRowId))].slice(0, POOL_MAX)
  if (poolIds.length === 0) return degraded('empty-pool')

  // `clientAddress` is a getter that throws on adapters that cannot supply it,
  // so it is passed lazily rather than destructured.
  if (
    rateLimited(
      'chat',
      clientIp(request, () => context.clientAddress),
      RATE_LIMIT_PER_MIN,
    )
  )
    return degraded('rate-limited')
  if (ceilinged('chat', CEILING_PER_MIN)) return degraded('rate-limited')

  const accessKeyId = process.env.BALGO_AWS_KEY_ID || import.meta.env.BALGO_AWS_KEY_ID
  const secretAccessKey = process.env.BALGO_AWS_SECRET || import.meta.env.BALGO_AWS_SECRET
  const region = process.env.BALGO_AWS_REGION || import.meta.env.BALGO_AWS_REGION || 'us-east-2'
  const modelId = process.env.BALGO_MODEL_ID || import.meta.env.BALGO_MODEL_ID || 'openai.gpt-oss-120b-1:0'
  if (!accessKeyId || !secretAccessKey) return degraded('unset')

  // The pool is resolved server-side against the document's own rows, so the
  // model is never shown a row the client made up. The origin is a constant —
  // never `request.url` — because otherwise the Host header decides what "the
  // document" is. See src/lib/naam/server.ts.
  const rows = await loadCoreRows()
  const byId = new Map(rows.map((row) => [row.id, row]))
  const poolRows = poolIds.map((id) => byId.get(id)).filter((row) => row !== undefined)
  if (poolRows.length === 0) return degraded('empty-pool')

  const remaining = deadline - Date.now()
  if (remaining < 1000) return degraded('timeout')

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), Math.min(TIMEOUT_MS, remaining))
  try {
    const client = new BedrockRuntimeClient({ region, credentials: { accessKeyId, secretAccessKey } })

    /**
     * THE GROUNDING SET GROWS, AND THAT IS THE POINT — it never loosens.
     *
     * The model may only name ids it has actually been shown, and until now
     * that was exactly the pool the client sent. With a search tool it can be
     * shown more, so `allowed` accumulates every row the document hands back.
     * Every one of those came out of loadCoreRows() by id, so the guarantee is
     * unchanged in kind: a name that is not in the document cannot enter this
     * set, and coerceModelReply still drops anything outside it. What has
     * changed is that the model can go and find rows the first search missed
     * instead of being told the document is empty.
     */
    const allowed = new Set(poolRows.map((row) => row.id))
    const messages: Message[] = [{ role: 'user', content: [{ text: buildUserTurn(ask, poolRows, absent) }] }]

    const converse = (force: boolean) =>
      new ConverseCommand({
        modelId,
        system: [{ text: buildSystemPrompt() }],
        messages,
        inferenceConfig: { maxTokens: MAX_TOKENS, temperature: 0.6 },
        /**
         * gpt-oss is a reasoning model and its thinking is billed against the
         * SAME maxTokens as the answer. At `low` it spends a few dozen tokens
         * deciding; left at the default it enumerates the entire pool one row at
         * a time — measured, verbatim: "bhadra meaning deity? Not calm. bhaga
         * light... Not calm." — and on the asks that deserve the most care it ran
         * out of budget mid-thought and returned `stopReason: 'max_tokens'` with
         * a reasoningContent block and NO toolUse at all. That is the whole of
         * the intermittent `degraded: 'empty'`: not a parse failure, a model that
         * never got to the answer.
         *
         * Reading the pool row by row is also not the work. The pool is at most
         * 60 rows of one-line glosses and the job is to choose three and say why
         * — `low` is the honest setting for it, not a cost saving.
         */
        additionalModelRequestFields: { reasoning_effort: 'low' },
        toolConfig: {
          /**
           * On the last round the search tool is TAKEN AWAY rather than merely
           * discouraged. `toolChoice: any` forces a call to some tool, and a
           * model that still has a search available will keep reaching for it
           * — so the way to make it answer is to leave it nothing else it can
           * do. Removing the tool is also what keeps the 11s budget honest:
           * there is no round in which another search is possible.
           */
          tools: force
            ? [{ toolSpec: { name: NAAM_TOOL_NAME, inputSchema: TOOL_SCHEMA } }]
            : [
                { toolSpec: { name: NAAM_TOOL_NAME, inputSchema: TOOL_SCHEMA } },
                { toolSpec: { name: NAAM_SEARCH_TOOL, inputSchema: SEARCH_SCHEMA } },
              ],
          // `any` is a forced call to one of the declared tools, and it is the
          // choice mode every Bedrock model supports, including gpt-oss.
          toolChoice: { any: {} },
        },
      })

    /**
     * ASK, AND LET IT GO BACK TO THE DOCUMENT.
     *
     * Most turns end on the first pass: the pool was built by searching the
     * meanings for the visitor's own words, so "moon" arrives already holding
     * Sasi. The loop is for the turns where the visitor's word is not the
     * dictionary's word — "brave", which appears in no gloss here while
     * valiant, heroic and bold all do — and it costs a round trip only on the
     * asks that need one.
     */
    let raw: unknown
    for (let round = 0; ; round++) {
      const last = round >= MAX_SEARCH_ROUNDS || deadline - Date.now() < 3500
      const response = await client.send(converse(last), { abortSignal: controller.signal })
      const message = response.output?.message
      const content = message?.content ?? []
      const call = content.find((block) => block.toolUse?.name === NAAM_SEARCH_TOOL)?.toolUse

      if (!call || last) {
        const answer = content.find((block) => block.toolUse?.name === NAAM_TOOL_NAME)?.toolUse
        raw = answer?.input ?? content.find((block) => block.text)?.text
        break
      }

      // Run the model's own queries against the document's meanings.
      const input = (call.input ?? {}) as { queries?: unknown }
      const queries = (Array.isArray(input.queries) ? input.queries : [])
        .map((q) => tidy(q, 60))
        .filter((q): q is string => Boolean(q))
        .slice(0, 4)

      const found: NaamRow[] = []
      for (const query of queries) {
        for (const hit of retrieve(rows, query, SEARCH_LIMIT)) {
          if (allowed.has(hit.row.id) && found.some((r) => r.id === hit.row.id)) continue
          allowed.add(hit.row.id)
          if (!found.some((r) => r.id === hit.row.id)) found.push(hit.row)
        }
      }

      if (message) messages.push(message)
      messages.push({
        role: 'user',
        content: [
          {
            toolResult: {
              toolUseId: call.toolUseId,
              content: [{ text: buildSearchResult(queries, found) }],
            },
          },
        ],
      })
    }

    // Filtered against every row the model was actually SHOWN — the pool it
    // started with plus anything its own searches returned. All of it came out
    // of loadCoreRows() by id, so an id that names nothing still cannot
    // round-trip, which is the contract this route states.
    const { reply, pickIds } = coerceModelReply(
      raw,
      [...allowed].map((id) => byId.get(id)).filter((row) => row !== undefined),
    )
    if (!reply) return degraded('empty')

    return json({
      degraded: false,
      reply: reply.slice(0, MAX_REPLY_CHARS),
      pickIds: pickIds.slice(0, NAAM_MAX_PICKS),
    })
  } catch (error) {
    return degraded(controller.signal.aborted || (error as Error)?.name === 'AbortError' ? 'timeout' : 'error')
  } finally {
    clearTimeout(timer)
  }
}
