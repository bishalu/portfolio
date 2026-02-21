import type { Handler, ScheduledEvent } from "@netlify/functions";
import { getStore } from "@netlify/blobs";
import * as cheerio from "cheerio";

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
      return ok("Parse failed");
    }

    const store = getStore(STORE_NAME);
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
      return ok("Initialized state");
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
      return ok("Change notified");
    }

    await store.set(STATE_KEY, nextState);
    return ok("No change");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await safeSlackFailure("fetch_failed", message, runId);
    return ok("Error handled");
  }
};

async function fetchLeaderboard(): Promise<string> {
  const response = await fetch(LEADERBOARD_URL, {
    headers: {
      "user-agent":
        "arena-leaderboard-watch/1.0 (+https://bishalup.netlify.app)",
      accept: "text/html,application/xhtml+xml",
    },
  });

  if (!response.ok) {
    throw new Error(`Leaderboard fetch failed: ${response.status}`);
  }

  return response.text();
}

function parseLeaderboard(html: string, scrapedAt: string) {
  const $ = cheerio.load(html);
  const text = $("body").text();
  const lines = text
    .split("\n")
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean);

  const headerIndex = lines.findIndex((line) =>
    /Model\s+Score\s+Votes/.test(line)
  );

  if (headerIndex === -1) {
    return null;
  }

  const anchorMap = buildAnchorMap($);
  const rowsLines = lines.slice(headerIndex + 1);
  const first = parseRow(rowsLines, 0, anchorMap);
  if (!first) {
    return null;
  }
  const second = parseRow(rowsLines, first.nextIndex, anchorMap);
  if (!second) {
    return null;
  }

  const leaderboardDate = findDate(text);
  const totalVotes = findCount(text, /([\d,]+)\s+votes/i);
  const modelCount = findCount(text, /([\d,]+)\s+models/i);

  if (
    !first.model ||
    !second.model ||
    Number.isNaN(first.score) ||
    Number.isNaN(first.votes) ||
    Number.isNaN(second.score) ||
    Number.isNaN(second.votes)
  ) {
    return null;
  }

  return {
    top1: first.model,
    top2: second.model,
    leadMarginScore: first.model.score - second.model.score,
    leadMarginVotes: first.model.votes - second.model.votes,
    meta: {
      leaderboard_date: leaderboardDate ?? undefined,
      total_votes: totalVotes ?? undefined,
      model_count: modelCount ?? undefined,
      scraped_at: scrapedAt,
    },
  };
}

function parseRow(
  lines: string[],
  startIndex: number,
  anchorMap: Map<string, string>
) {
  let i = startIndex;
  while (i < lines.length && !isRank(lines[i])) {
    i += 1;
  }
  if (i >= lines.length) {
    return null;
  }
  const rank = parseInt(lines[i], 10);
  i += 1;
  if (i >= lines.length) {
    return null;
  }
  const rankSpread = lines[i];
  if (!isRankSpread(rankSpread)) {
    return null;
  }
  i += 1;

  const metaLines: string[] = [];
  while (i < lines.length && !isScore(lines[i])) {
    metaLines.push(lines[i]);
    i += 1;
  }
  if (i >= lines.length) {
    return null;
  }
  const scoreLine = lines[i];
  i += 1;
  if (i >= lines.length) {
    return null;
  }
  const votesLine = lines[i];
  i += 1;

  const cleanedMeta = metaLines
    .filter((line) => line && !/^Image:/.test(line))
    .filter((line) => line !== "Preliminary");
  const preliminary = metaLines.includes("Preliminary");
  const licenseLine = cleanedMeta.find((line) => line.includes("·"));

  let org: string | undefined;
  let license: string | undefined;
  if (licenseLine) {
    const [orgPart, licensePart] = licenseLine.split("·").map((s) => s.trim());
    if (orgPart) org = orgPart;
    if (licensePart) license = licensePart;
  }

  let modelName: string | undefined;
  if (licenseLine) {
    const licenseIndex = cleanedMeta.indexOf(licenseLine);
    if (licenseIndex > 0) {
      modelName = cleanedMeta[licenseIndex - 1];
    }
    if (!org && licenseIndex > 1) {
      org = cleanedMeta[licenseIndex - 2];
    }
  }

  if (!modelName && cleanedMeta.length > 0) {
    modelName = cleanedMeta[cleanedMeta.length - 1];
  }

  if (!org && cleanedMeta.length > 1) {
    org = cleanedMeta[0];
  }

  if (!modelName) {
    return null;
  }

  const { score, scoreCi } = parseScore(scoreLine);
  const votes = parseVotes(votesLine);
  const url = anchorMap.get(modelName);

  const model: ModelSnapshot = {
    model: modelName,
    org,
    license,
    score,
    score_ci: scoreCi,
    votes,
    rank,
    rank_spread: rankSpread,
    preliminary,
    url,
  };

  return { model, nextIndex: i };
}

function isRank(value: string) {
  return /^\d+$/.test(value);
}

function isRankSpread(value: string) {
  return /^\d+\s+\d+$/.test(value);
}

function isScore(value: string) {
  return /^\d{3,4}±\d+$/.test(value);
}

function parseScore(value: string) {
  const match = value.match(/^(\d{3,4})±(\d+)$/);
  if (!match) {
    return { score: Number.NaN, scoreCi: undefined };
  }
  return { score: Number(match[1]), scoreCi: Number(match[2]) };
}

function parseVotes(value: string) {
  const match = value.match(/^[\d,]+$/);
  if (!match) return Number.NaN;
  return Number(value.replace(/,/g, ""));
}

function findDate(text: string) {
  const match = text.match(
    /\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b/
  );
  return match ? match[0] : null;
}

function findCount(text: string, regex: RegExp) {
  const match = text.match(regex);
  if (!match) return null;
  return Number(match[1].replace(/,/g, ""));
}

function buildAnchorMap($: cheerio.CheerioAPI) {
  const map = new Map<string, string>();
  $("a[href]").each((_i, el) => {
    const text = $(el).text().trim();
    const href = $(el).attr("href");
    if (!text || !href) return;
    if (!map.has(text)) {
      map.set(text, href);
    }
  });
  return map;
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
    `Score: ${top1.score}${top1.score_ci ? `±${top1.score_ci}` : ""} | Votes: ${top1.votes.toLocaleString()}`,
    `Prev #1: ${previous.current.model} (${previous.current.score}${
      previous.current.score_ci ? `±${previous.current.score_ci}` : ""
    })`,
    `Lead over #2: +${leadScore} score | +${leadVotes.toLocaleString()} votes`,
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

export { handler, handler as default };
export const config = {
  schedule: "*/30 * * * *",
};
