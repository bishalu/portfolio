import type { Handler, ScheduledEvent } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

type ModelSnapshot = {
  model: string;
  org?: string;
  license?: string;
  score: number;
  score_ci?: number;
  votes: number;
  rank: number;
  rank_spread?: string;
  preliminary?: boolean;
  url?: string;
};

type State = {
  current: ModelSnapshot;
  runner_up: ModelSnapshot;
  meta: {
    leaderboard_date?: string;
    total_votes?: number;
    model_count?: number;
    scraped_at: string;
  };
  stats: {
    lead_margin_score: number;
    lead_margin_votes: number;
    changes_count: number;
    last_change_at?: string;
  };
  history: Array<{
    changed_at: string;
    from: string;
    to: string;
    from_score: number;
    to_score: number;
    lead_margin_score: number;
    leaderboard_date?: string;
  }>;
};

const LEADERBOARD_URL =
  "https://arena.ai/leaderboard/text/overall-no-style-control";
const STORE_NAME = "arena-leaderboard";
const STATE_KEY = "state.json";
const MAX_HISTORY = 30;

const handler: Handler = async (_event, context) => {
  const debug =
    typeof _event === "object" &&
    _event !== null &&
    "queryStringParameters" in _event &&
    (_event as { queryStringParameters?: Record<string, string> })
      .queryStringParameters?.debug === "1";
  const testNotify =
    typeof _event === "object" &&
    _event !== null &&
    "queryStringParameters" in _event &&
    (_event as { queryStringParameters?: Record<string, string> })
      .queryStringParameters?.test === "1";
  const scheduledEvent = context as ScheduledEvent;
  const runId = scheduledEvent?.event?.id ?? "manual";
  const scrapedAt = new Date().toISOString();

  try {
    const html = await fetchLeaderboard();
    const parsed = parseLeaderboard(html, scrapedAt);

    if (!parsed) {
      await safeSlackFailure(
        "parse_failed",
        "Missing required fields while parsing leaderboard.",
        runId
      );
      return ok(debug ? "Parse failed (debug enabled)" : "Parse failed");
    }

    if (testNotify) {
      await sendSlackTest(parsed);
      return ok(debug ? "Test notification sent (debug enabled)" : "Test notification sent");
    }

    const store = getConfiguredStore();
    const previous = await store.get<State>(STATE_KEY, { type: "json" });

    if (!previous) {
      const initialState: State = {
        current: parsed.top1,
        runner_up: parsed.top2,
        meta: parsed.meta,
        stats: {
          lead_margin_score: parsed.leadMarginScore,
          lead_margin_votes: parsed.leadMarginVotes,
          changes_count: 0,
          last_change_at: undefined,
        },
        history: [],
      };
      await store.set(STATE_KEY, initialState);
      return ok(debug ? "Initialized state (debug enabled)" : "Initialized state");
    }

    const hasChange = previous.current.model !== parsed.top1.model;
    const nextState: State = {
      current: parsed.top1,
      runner_up: parsed.top2,
      meta: parsed.meta,
      stats: {
        lead_margin_score: parsed.leadMarginScore,
        lead_margin_votes: parsed.leadMarginVotes,
        changes_count: previous.stats.changes_count,
        last_change_at: previous.stats.last_change_at,
      },
      history: previous.history ?? [],
    };

    if (hasChange) {
      const changedAt = scrapedAt;
      nextState.stats.changes_count = previous.stats.changes_count + 1;
      nextState.stats.last_change_at = changedAt;

      nextState.history = [
        {
          changed_at: changedAt,
          from: previous.current.model,
          to: parsed.top1.model,
          from_score: previous.current.score,
          to_score: parsed.top1.score,
          lead_margin_score: parsed.leadMarginScore,
          leaderboard_date: parsed.meta.leaderboard_date,
        },
        ...(previous.history ?? []),
      ].slice(0, MAX_HISTORY);

      await store.set(STATE_KEY, nextState);
      await sendSlackChange(parsed, previous);
      return ok(debug ? "Change notified (debug enabled)" : "Change notified");
    }

    await store.set(STATE_KEY, nextState);
    return ok(debug ? "No change (debug enabled)" : "No change");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await safeSlackFailure("fetch_failed", message, runId);
    return ok(
      debug ? `Error handled (debug): ${message}` : "Error handled"
    );
  }
};

async function fetchLeaderboard(): Promise<string> {
  const rscUrl = `${LEADERBOARD_URL}?_rsc=1`;
  const response = await fetch(rscUrl, {
    headers: {
      "user-agent":
        "arena-leaderboard-watch/1.0 (+https://bishalup.netlify.app)",
      accept: "text/x-component,text/plain,*/*",
      RSC: "1",
    },
  });

  if (!response.ok) {
    throw new Error(`Leaderboard fetch failed: ${response.status}`);
  }

  return response.text();
}

function parseLeaderboard(rscPayload: string, scrapedAt: string) {
  const key = "\"leaderboardSlug\":\"overall-no-style-control\"";
  const idx = rscPayload.indexOf(key);
  if (idx === -1) return null;

  const start = rscPayload.lastIndexOf("{", idx);
  const end = rscPayload.indexOf("},\"plots\"", idx);
  if (start === -1 || end === -1) return null;

  const blob = rscPayload.slice(start, end + 1);
  let data: {
    entries: Array<{
      rank: number;
      rankUpper?: number;
      rankLower?: number;
      rankStyleControl?: number;
      modelDisplayName: string;
      rating: number;
      ratingUpper?: number;
      ratingLower?: number;
      votes: number;
      modelOrganization?: string;
      modelUrl?: string;
      license?: string;
    }>;
    voteCutoffISOString?: string;
    totalVotes?: number;
    totalModels?: number;
  };

  try {
    data = JSON.parse(blob);
  } catch {
    return null;
  }

  if (!data.entries || data.entries.length < 2) return null;

  const top1 = toSnapshot(data.entries[0]);
  const top2 = toSnapshot(data.entries[1]);

  if (!top1 || !top2) return null;

  const leaderboardDate = data.voteCutoffISOString
    ? formatDateUTC(data.voteCutoffISOString)
    : undefined;

  return {
    top1,
    top2,
    leadMarginScore: roundTo(top1.score - top2.score, 2),
    leadMarginVotes: top1.votes - top2.votes,
    meta: {
      leaderboard_date: leaderboardDate,
      total_votes: data.totalVotes,
      model_count: data.totalModels,
      scraped_at: scrapedAt,
    },
  };
}

function toSnapshot(entry: {
  rank: number;
  rankUpper?: number;
  rankLower?: number;
  rankStyleControl?: number;
  modelDisplayName: string;
  rating: number;
  ratingUpper?: number;
  ratingLower?: number;
  votes: number;
  modelOrganization?: string;
  modelUrl?: string;
  license?: string;
}): ModelSnapshot | null {
  if (!entry.modelDisplayName || typeof entry.rating !== "number") return null;
  if (typeof entry.votes !== "number") return null;

  const score = roundTo(entry.rating, 2);
  const scoreCi =
    typeof entry.ratingUpper === "number" && typeof entry.ratingLower === "number"
      ? roundTo((entry.ratingUpper - entry.ratingLower) / 2, 2)
      : undefined;

  const rankSpread =
    typeof entry.rankUpper === "number" && typeof entry.rankLower === "number"
      ? `${entry.rankUpper} ${entry.rankLower}`
      : undefined;

  return {
    model: entry.modelDisplayName,
    org: entry.modelOrganization,
    license: entry.license,
    score,
    score_ci: scoreCi,
    votes: entry.votes,
    rank: entry.rank,
    rank_spread: rankSpread,
    preliminary: false,
    url: entry.modelUrl,
  };
}

async function sendSlackChange(
  parsed: ReturnType<typeof parseLeaderboard>,
  previous: State
) {
  if (!parsed) return;
  const webhookUrl = process.env.SLACK_WEBHOOK_URL;
  if (!webhookUrl) return;

  const top1 = parsed.top1;
  const top2 = parsed.top2;
  const leadScore = parsed.leadMarginScore;
  const leadVotes = parsed.leadMarginVotes;
  const date = parsed.meta.leaderboard_date ?? "unknown";
  const totalVotes =
    parsed.meta.total_votes !== undefined
      ? parsed.meta.total_votes.toLocaleString()
      : "unknown";

  const message = [
    `*New #1:* ${top1.model}${top1.org ? ` (${top1.org})` : ""}`,
    `Score: ${formatScore(top1.score)}${
      top1.score_ci !== undefined ? `±${formatScore(top1.score_ci)}` : ""
    } | Votes: ${top1.votes.toLocaleString()}`,
    `#2: ${top2.model}${top2.org ? ` (${top2.org})` : ""} (${formatScore(
      top2.score
    )}${top2.score_ci !== undefined ? `±${formatScore(top2.score_ci)}` : ""} | Votes: ${top2.votes.toLocaleString()})`,
    `Prev #1: ${previous.current.model} (${formatScore(previous.current.score)}${
      previous.current.score_ci !== undefined
        ? `±${formatScore(previous.current.score_ci)}`
        : ""
    })`,
    `Lead over #2: +${formatScore(leadScore)} score | +${leadVotes.toLocaleString()} votes`,
    `Leaderboard date: ${date} | Total votes: ${totalVotes}`,
    `Link: ${LEADERBOARD_URL}`,
  ].join("\n");

  await postToSlack(webhookUrl, {
    text: `New #1 on Arena Text (no style control): ${top1.model}`,
    blocks: [
      {
        type: "section",
        text: { type: "mrkdwn", text: message },
      },
    ],
  });
}

async function sendSlackTest(parsed: ReturnType<typeof parseLeaderboard>) {
  if (!parsed) return;
  const webhookUrl = process.env.SLACK_WEBHOOK_URL;
  if (!webhookUrl) return;

  const top1 = parsed.top1;
  const top2 = parsed.top2;
  const leadScore = parsed.leadMarginScore;
  const leadVotes = parsed.leadMarginVotes;
  const date = parsed.meta.leaderboard_date ?? "unknown";
  const totalVotes =
    parsed.meta.total_votes !== undefined
      ? parsed.meta.total_votes.toLocaleString()
      : "unknown";

  const message = [
    "*TEST* Arena watcher is live.",
    `Current #1: ${top1.model}${top1.org ? ` (${top1.org})` : ""}`,
    `Score: ${formatScore(top1.score)}${
      top1.score_ci !== undefined ? `±${formatScore(top1.score_ci)}` : ""
    } | Votes: ${top1.votes.toLocaleString()}`,
    `#2: ${top2.model}${top2.org ? ` (${top2.org})` : ""} (${formatScore(
      top2.score
    )}${top2.score_ci !== undefined ? `±${formatScore(top2.score_ci)}` : ""} | Votes: ${top2.votes.toLocaleString()})`,
    `Lead: +${formatScore(leadScore)} score | +${leadVotes.toLocaleString()} votes`,
    `Leaderboard date: ${date} | Total votes: ${totalVotes}`,
    `Link: ${LEADERBOARD_URL}`,
  ].join("\n");

  await postToSlack(webhookUrl, {
    text: `TEST: Arena watcher live (${top1.model})`,
    blocks: [
      {
        type: "section",
        text: { type: "mrkdwn", text: message },
      },
    ],
  });
}

async function sendSlackFailure(
  kind: "fetch_failed" | "parse_failed",
  detail: string,
  runId: string
) {
  const webhookUrl = process.env.SLACK_WEBHOOK_URL;
  if (!webhookUrl) return;

  const text = [
    "*Arena Watcher Failure*",
    `Type: ${kind}`,
    `Run: ${runId}`,
    `Detail: ${detail}`,
    `URL: ${LEADERBOARD_URL}`,
  ].join("\n");

  await postToSlack(webhookUrl, {
    text: `Arena watcher failure: ${kind}`,
    blocks: [{ type: "section", text: { type: "mrkdwn", text } }],
  });
}

async function safeSlackFailure(
  kind: "fetch_failed" | "parse_failed",
  detail: string,
  runId: string
) {
  try {
    await sendSlackFailure(kind, detail, runId);
  } catch (error) {
    console.error("Slack failure alert failed", error);
  }
}

async function postToSlack(
  webhookUrl: string,
  payload: Record<string, unknown>
) {
  const response = await fetch(webhookUrl, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Slack webhook failed: ${response.status}`);
  }
}

function ok(message: string) {
  return {
    statusCode: 200,
    body: message,
  };
}

function getConfiguredStore() {
  const siteID =
    process.env.BLOBS_SITE_ID ||
    process.env.NETLIFY_SITE_ID ||
    process.env.NETLIFY_BLOBS_SITE_ID;
  const token =
    process.env.BLOBS_TOKEN ||
    process.env.NETLIFY_AUTH_TOKEN ||
    process.env.NETLIFY_ACCESS_TOKEN ||
    process.env.NETLIFY_BLOBS_TOKEN;
  if (siteID && token) {
    return getStore(STORE_NAME, { siteID, token });
  }
  return getStore(STORE_NAME);
}

function formatDateUTC(iso: string) {
  const date = new Date(iso);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

function roundTo(value: number, digits: number) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatScore(value: number) {
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}

export { handler, handler as default };
export const config = {
  schedule: "*/30 * * * *",
};
