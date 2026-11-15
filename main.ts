// free-model-aggregator-01: one server, two client APIs, ten free-tier providers.
//
// The server reads the trial brief (provider directory, quality table, workload manifest), probes
// every provider's catalog in its own wire, and registers one catalog document with the env. For
// each request the client sends it then:
//
//   1. picks the strongest free model that meets the request's requirements (tools, vision, JSON
//      mode, minimum context) — quality is 60% of the score, so the pick comes from the quality
//      table's ratings, not from listing order;
//   2. plans the whole workload against the providers' free-tier budgets before the first request
//      arrives: requests are assigned greedily, highest weight and highest regret (the drop to the
//      next provider) first, under a per-provider call cap set at the floor of each provider's
//      budget range, so no provider is planned past what it can certainly pay for — a planned
//      call that refuses anyway is a wasted call, and calls are the tie-break;
//   3. speaks the serving provider's wire — OpenAI, Anthropic or Gemini — translating the request
//      through one intermediate representation that preserves exactly what a provider can attest
//      (system text, last user text, tool names, max_tokens, stops, image, JSON mode);
//   4. fails over on any refusal (402, 429, 5xx, transport error) to the next-best candidate,
//      tracking each provider's refusals: two refusals demote a provider, three retire it for the
//      rest of the trial, which is how a budget spent mid-trial or an outage is ridden out. The
//      first candidate is retried once when the refusal came deep inside its provider's
//      budget (the refused call number below the budget floor, and at most one refusal
//      before): such a refusal is usually the provider's scheduled failure for that one call
//      number, and the next call steps past it, keeping the request on its best model;
//   5. renders the answer in the client's wire, JSON or SSE, from the one parsed provider answer,
//      so streamed and plain answers carry identical text, usage and stop reason.
//
// Providers are called one at a time and never twice for one request once one has answered: the
// accounting that matters counts exactly one successful call per request. Everything is
// deterministic — no clock, no randomness; a response's ids and timestamps are fixed strings and
// counters — so a replayed trial reproduces the live pass byte for byte.
//
// `node main.ts --list` prints the catalog document the server registers, byte for byte, and exits.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  buildAnthropicBody,
  buildGeminiBody,
  buildOpenAIBody,
  parseAnthropicAnswer,
  parseAnthropicRequest,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  parseOpenAIRequest,
  renderAnthropicError,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIError,
  renderOpenAIJson,
  renderOpenAIStream,
  type Answer,
  type IrRequest,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Refusals after which a provider is demoted behind every healthier candidate. */
const DEMOTE_AFTER_REFUSALS = 2;
/** Refusals after which a provider is treated as out for the rest of the trial. */
const RETIRE_AFTER_REFUSALS = 3;

/**
 * The floor of each provider's per-trial call budget, from the competition's published provider
 * table. A trial draws each budget inside a range and never publishes the draw, so the plan is
 * built against the floor: a plan that fits the floor fits every draw. A provider missing from
 * the table gets the conservative default.
 */
const BUDGET_FLOOR: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_BUDGET_FLOOR = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The plan: workload request id -> the candidate chosen for it before the trial starts. */
const plan = new Map<string, Candidate>();
/** How many chat requests have arrived: the client sends the manifest in order, one at a time. */
let manifestIndex = 0;
/** Response counter: response ids are built from it, so a replay mints the same ids. */
let responseSeq = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------- candidates, plan, provider health ----------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}

interface ProviderHealth {
  calls: number;
  refusals: number;
}

const health = new Map<string, ProviderHealth>();

function healthOf(providerId: string): ProviderHealth {
  let h = health.get(providerId);
  if (h === undefined) {
    h = { calls: 0, refusals: 0 };
    health.set(providerId, h);
  }
  return h;
}

const candidateOrder = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating || a.providerIndex - b.providerIndex || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);

/**
 * The best free model per provider for one request: every provider that lists a free model
 * meeting the requirements contributes its highest-rated one, best first. When `jsonWireBlock`
 * holds (an OpenAI client asking for JSON mode), anthropic-wire providers are excluded: their
 * wire cannot carry the JSON-mode flag, so their answers could not attest to it.
 */
function candidatesFor(requires: Requirement, jsonWireBlock: boolean): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    if (jsonWireBlock && probe.provider.wire === 'anthropic') return;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const candidate: Candidate = { probe, model, rating: row?.rating_milli ?? 0, providerIndex };
      if (best === null || candidateOrder(candidate, best) < 0) best = candidate;
    }
    if (best !== null) out.push(best);
  });
  return out.sort(candidateOrder);
}

/**
 * Assign every manifest request a candidate before the trial starts. Requests are placed
 * highest-weight first, ties broken by regret — the rating lost if the best provider is full —
 * so scarce budget goes where losing it costs most. A provider is planned at most its budget
 * floor of calls; when every candidate is at its cap the best is still assigned (the runtime
 * fails over when a drawn budget turns out smaller than the plan).
 */
function buildPlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  const used = new Map<string, number>();
  const rows = manifest.map((entry, index) => {
    const cands = candidatesFor(entry.requires, entry.api === 'openai' && entry.requires.json === true);
    const best = cands[0];
    const alternative = best === undefined ? undefined : cands.find((c) => c.probe.provider.id !== best.probe.provider.id);
    const regret = best === undefined ? 0 : best.rating - (alternative?.rating ?? 0);
    return { entry, index, cands, regret };
  });
  rows.sort((a, b) => b.entry.weight - a.entry.weight || b.regret - a.regret || a.index - b.index);
  for (const { entry, cands } of rows) {
    if (cands.length === 0) continue;
    const pick =
      cands.find((c) => (used.get(c.probe.provider.id) ?? 0) < (BUDGET_FLOOR[c.probe.provider.id] ?? DEFAULT_BUDGET_FLOOR)) ??
      cands[0];
    used.set(pick.probe.provider.id, (used.get(pick.probe.provider.id) ?? 0) + 1);
    plan.set(entry.id, pick);
  }
}

/**
 * The order candidates are tried for one request: the planned candidate first (unless its
 * provider is already demoted, where leading with it would likely waste a call), the rest by
 * rating, with providers that have refused twice demoted behind healthier ones and providers
 * that have refused three times left out while anything else remains.
 */
function orderedCandidates(cands: Candidate[], planned: Candidate | null): Candidate[] {
  const alive = cands.filter((c) => healthOf(c.probe.provider.id).refusals < RETIRE_AFTER_REFUSALS);
  const pool = alive.length > 0 ? alive : cands;
  const sorted = [...pool].sort((a, b) => {
    const demoteA = healthOf(a.probe.provider.id).refusals >= DEMOTE_AFTER_REFUSALS ? 1 : 0;
    const demoteB = healthOf(b.probe.provider.id).refusals >= DEMOTE_AFTER_REFUSALS ? 1 : 0;
    return demoteA - demoteB || candidateOrder(a, b);
  });
  if (planned !== null && healthOf(planned.probe.provider.id).refusals < DEMOTE_AFTER_REFUSALS) {
    const at = sorted.findIndex((c) => c.probe.provider.id === planned.probe.provider.id && c.model.id === planned.model.id);
    if (at > 0) {
      const [first] = sorted.splice(at, 1);
      sorted.unshift(first);
    }
  }
  return sorted;
}

// ---------- upstream calls ----------

/** One inference call to one provider, in its wire. Null when the provider refused or broke. */
async function callProvider(cand: Candidate, ir: IrRequest): Promise<Answer | null> {
  const provider: ProviderDirectoryEntry = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = buildOpenAIBody(ir, cand.model.id);
  } else if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    url = `${base}/v1/messages`;
    body = buildAnthropicBody(ir, cand.model.id);
  } else {
    const built = buildGeminiBody(ir, cand.model.id);
    url = `${base}${built.path}`;
    body = built.body;
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return null; // transport failure: the provider never saw this call, so it spends nothing
  }
  const h = healthOf(provider.id);
  h.calls += 1;
  if (!res.ok) {
    h.refusals += 1;
    try {
      await res.text();
    } catch {
      // The refusal stands whether or not its body can be read.
    }
    return null;
  }
  let json: unknown = null;
  try {
    json = await res.json();
  } catch {
    json = null;
  }
  if (provider.wire === 'openai') return parseOpenAIAnswer(json);
  if (provider.wire === 'anthropic') return parseAnthropicAnswer(json);
  return parseGeminiAnswer(json);
}

// ---------- HTTP plumbing ----------

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  buildPlan();
}

/** The free model ids across the providers that answered, deduplicated: what GET /v1/models lists. */
function servedModelIds(): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push(model.id);
    }
  }
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendText(res, status, 'application/json', JSON.stringify(body));
}

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, message: string): void {
  if (api === 'anthropic') sendText(res, status, 'application/json', renderAnthropicError(message));
  else sendText(res, status, 'application/json', renderOpenAIError(message));
}

/** Requirements read off a request body, for a brief whose manifest has no row for it. */
function requirementsFromIr(ir: IrRequest): Requirement {
  return {
    tools: ir.tools.length > 0,
    vision: ir.turns.some((t) => t.role === 'user' && t.images.length > 0),
    json: ir.jsonMode,
  };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const ids = servedModelIds();
  if (typeof req.headers['x-api-key'] === 'string' || typeof req.headers['anthropic-version'] === 'string') {
    sendJson(res, 200, {
      data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: '2024-01-01T00:00:00Z' })),
      has_more: false,
      first_id: ids.length > 0 ? ids[0] : null,
      last_id: ids.length > 0 ? ids[ids.length - 1] : null,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: ids.map((id) => ({ id, object: 'model', created: 0, owned_by: 'free-model-aggregator' })),
    });
  }
}

/** Serve one chat request on either client wire: plan, failover, render. */
async function chat(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, 'the request body is not one JSON object');
  const acceptStream = String(req.headers.accept ?? '').includes('text/event-stream');
  const ir = api === 'openai' ? parseOpenAIRequest(body, acceptStream) : parseAnthropicRequest(body, acceptStream);

  // The client sends the manifest's requests in order, one at a time, so the arriving request is
  // the manifest row at the arrival counter; its declared requirements and the plan govern. A row
  // that does not match the arriving wire falls back to requirements read off the body itself.
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  let requires: Requirement;
  let planned: Candidate | null = null;
  if (entry !== undefined && entry.api === api && entry.stream === ir.stream) {
    requires = entry.requires;
    planned = plan.get(entry.id) ?? null;
  } else {
    requires = requirementsFromIr(ir);
  }

  const cands = candidatesFor(requires, api === 'openai' && ir.jsonMode);
  const ordered = orderedCandidates(cands, planned);
  responseSeq += 1;
  const seq = responseSeq;
  const firstRefusals = ordered.length > 0 ? healthOf(ordered[0].probe.provider.id).refusals : 0;
  for (let i = 0; i < ordered.length; i++) {
    const cand = ordered[i];
    let answer = await callProvider(cand, ir);
    if (answer === null && i === 0 && firstRefusals <= 1) {
      // The first candidate refused. Retry it once only when the refusal sits deep inside the
      // provider's budget — the refused call number below its budget floor — where a refusal
      // is the provider's scheduled failure for that call number, not an empty budget or an
      // outage; the next call steps past it. Near the floor a refusal is likelier permanent,
      // and a wasted retry is a wasted call.
      const h = healthOf(cand.probe.provider.id);
      if (h.calls < (BUDGET_FLOOR[cand.probe.provider.id] ?? DEFAULT_BUDGET_FLOOR)) {
        answer = await callProvider(cand, ir);
      }
    }
    if (answer === null) continue; // refused: count it (done in callProvider) and try the next model
    if (ir.stream) {
      sendText(res, 200, 'text/event-stream', api === 'openai' ? renderOpenAIStream(answer, cand.model.id, seq) : renderAnthropicStream(answer, cand.model.id, seq));
    } else {
      sendText(res, 200, 'application/json', api === 'openai' ? renderOpenAIJson(answer, cand.model.id, seq) : renderAnthropicJson(answer, cand.model.id, seq));
    }
    return;
  }
  sendError(res, api, 502, 'every free provider refused this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await chat('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. Wait for stdout to flush so the tail is not lost.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModelIds().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
