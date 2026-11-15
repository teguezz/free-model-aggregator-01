// Wire translations for the free-model aggregator.
//
// Every request the client sends is parsed once into an intermediate
// representation (IR) that keeps exactly the fields a provider can observe:
// the system text, the conversation turns (text plus images), the tool
// definitions, max_tokens, the stop sequences and the JSON-mode flag. The
// upstream body for a provider is then rendered from the IR in that
// provider's wire, so the answer a provider produces attests to the same
// request no matter which wire the client used or which wire served it.
//
// Providers are always called non-streaming; the answer is rendered back in
// the client's wire, as one JSON document or as a synthesized SSE stream.
// That keeps one code path per direction and makes the streamed and plain
// answers identical in text, usage and stop reason by construction.

export interface IrImage {
  kind: "data" | "url";
  mediaType: string; // for kind "data": the image media type
  value: string; // for kind "data": base64 payload; for kind "url": the URL
}

export interface IrTurn {
  role: "user" | "assistant";
  text: string;
  images: IrImage[];
}

export interface IrTool {
  name: string;
  description: string | null;
  schema: unknown;
}

export interface IrRequest {
  system: string; // "" when the client sent no system text
  turns: IrTurn[];
  tools: IrTool[];
  toolChoice: unknown; // the client's raw tool_choice, null when absent
  toolChoiceFrom: "openai" | "anthropic";
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
}

export interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string; // the answering provider's raw stop token, in its own wire
}

// ---------- small helpers ----------

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringOrNull(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

function dataUrlImage(url: string): IrImage | null {
  if (!url.startsWith("data:")) return null;
  const comma = url.indexOf(",");
  if (comma < 0) return null;
  const meta = url.slice(5, comma);
  const mediaType = meta.split(";")[0] || "application/octet-stream";
  return { kind: "data", mediaType, value: url.slice(comma + 1) };
}

function urlImage(url: string): IrImage {
  const data = dataUrlImage(url);
  if (data !== null) return data;
  return { kind: "url", mediaType: "", value: url };
}

// ---------- client request parsing: OpenAI wire ----------

// The text of an OpenAI content field: a plain string, or the text parts of
// a content array joined with newlines (the reading every consumer uses).
function openAIContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const part of content) {
    if (isRecord(part) && (part.type === "text" || part.type === "input_text") && typeof part.text === "string") {
      parts.push(part.text);
    }
  }
  return parts.join("\n");
}

function openAIContentImages(content: unknown): IrImage[] {
  if (!Array.isArray(content)) return [];
  const images: IrImage[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === "image_url" && isRecord(part.image_url) && typeof part.image_url.url === "string") {
      images.push(urlImage(part.image_url.url));
    } else if (part.type === "input_image" && typeof part.image_url === "string") {
      images.push(urlImage(part.image_url));
    }
  }
  return images;
}

export function parseOpenAIRequest(body: Record<string, unknown>, acceptStream: boolean): IrRequest {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemTexts: string[] = [];
  const turns: IrTurn[] = [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (message.role === "system") {
      systemTexts.push(openAIContentText(message.content));
    } else if (message.role === "user") {
      turns.push({ role: "user", text: openAIContentText(message.content), images: openAIContentImages(message.content) });
    } else if (message.role === "assistant") {
      turns.push({ role: "assistant", text: openAIContentText(message.content), images: [] });
    }
  }
  const tools: IrTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === "string") {
        tools.push({
          name: tool.function.name,
          description: stringOrNull(tool.function.description),
          schema: tool.function.parameters ?? { type: "object" },
        });
      }
    }
  }
  const stops: string[] = [];
  if (typeof body.stop === "string") stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const s of body.stop) if (typeof s === "string") stops.push(s);
  const responseFormat = isRecord(body.response_format) ? body.response_format : null;
  return {
    system: systemTexts.join("\n"),
    turns,
    tools,
    toolChoice: body.tool_choice ?? null,
    toolChoiceFrom: "openai",
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops,
    jsonMode: responseFormat !== null && (responseFormat.type === "json_object" || responseFormat.type === "json_schema"),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stream: body.stream === true || acceptStream,
  };
}

// ---------- client request parsing: Anthropic wire ----------

function anthropicBlocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

function anthropicBlockImages(content: unknown): IrImage[] {
  if (!Array.isArray(content)) return [];
  const images: IrImage[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== "image" || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === "base64" && typeof source.data === "string") {
      images.push({ kind: "data", mediaType: stringOrNull(source.media_type) ?? "application/octet-stream", value: source.data });
    } else if (source.type === "url" && typeof source.url === "string") {
      images.push({ kind: "url", mediaType: "", value: source.url });
    }
  }
  return images;
}

export function parseAnthropicRequest(body: Record<string, unknown>, acceptStream: boolean): IrRequest {
  let system = "";
  if (typeof body.system === "string") system = body.system;
  else if (Array.isArray(body.system)) system = anthropicBlocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const turns: IrTurn[] = [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (message.role === "user") {
      turns.push({ role: "user", text: anthropicBlocksText(message.content), images: anthropicBlockImages(message.content) });
    } else if (message.role === "assistant") {
      turns.push({ role: "assistant", text: anthropicBlocksText(message.content), images: [] });
    }
  }
  const tools: IrTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === "string") {
        tools.push({ name: tool.name, description: stringOrNull(tool.description), schema: tool.input_schema ?? { type: "object" } });
      }
    }
  }
  const stops: string[] = [];
  if (Array.isArray(body.stop_sequences)) for (const s of body.stop_sequences) if (typeof s === "string") stops.push(s);
  return {
    system,
    turns,
    tools,
    toolChoice: body.tool_choice ?? null,
    toolChoiceFrom: "anthropic",
    maxTokens: numOrNull(body.max_tokens),
    stops,
    jsonMode: false, // the Anthropic wire carries no JSON-mode flag
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stream: body.stream === true || acceptStream,
  };
}

// ---------- tool_choice translation ----------

function toolChoiceForOpenAI(ir: IrRequest): unknown {
  if (ir.toolChoice === null) return undefined;
  if (ir.toolChoiceFrom === "openai") return ir.toolChoice;
  if (isRecord(ir.toolChoice)) {
    if (ir.toolChoice.type === "auto") return "auto";
    if (ir.toolChoice.type === "any") return "required";
    if (ir.toolChoice.type === "none") return "none";
    if (ir.toolChoice.type === "tool" && typeof ir.toolChoice.name === "string") {
      return { type: "function", function: { name: ir.toolChoice.name } };
    }
  }
  return undefined;
}

function toolChoiceForAnthropic(ir: IrRequest): unknown {
  if (ir.toolChoice === null) return undefined;
  if (ir.toolChoiceFrom === "anthropic") return ir.toolChoice;
  if (typeof ir.toolChoice === "string") {
    if (ir.toolChoice === "auto") return { type: "auto" };
    if (ir.toolChoice === "required") return { type: "any" };
    if (ir.toolChoice === "none") return { type: "none" };
  }
  if (isRecord(ir.toolChoice) && isRecord(ir.toolChoice.function) && typeof ir.toolChoice.function.name === "string") {
    return { type: "tool", name: ir.toolChoice.function.name };
  }
  return undefined;
}

// ---------- upstream body builders ----------

function openAIImagePart(image: IrImage): Record<string, unknown> {
  const url = image.kind === "data" ? `data:${image.mediaType};base64,${image.value}` : image.value;
  return { type: "image_url", image_url: { url } };
}

export function buildOpenAIBody(ir: IrRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (ir.system !== "") messages.push({ role: "system", content: ir.system });
  for (const turn of ir.turns) {
    if (turn.role === "assistant") {
      messages.push({ role: "assistant", content: turn.text });
    } else if (turn.images.length > 0) {
      const parts: unknown[] = [];
      if (turn.text !== "") parts.push({ type: "text", text: turn.text });
      for (const image of turn.images) parts.push(openAIImagePart(image));
      messages.push({ role: "user", content: parts });
    } else {
      messages.push({ role: "user", content: turn.text });
    }
  }
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (ir.tools.length > 0) {
    body.tools = ir.tools.map((t) => ({
      type: "function",
      function: {
        name: t.name,
        ...(t.description !== null ? { description: t.description } : {}),
        parameters: t.schema,
      },
    }));
    const choice = toolChoiceForOpenAI(ir);
    if (choice !== undefined) body.tool_choice = choice;
  }
  if (ir.maxTokens !== null) body.max_tokens = ir.maxTokens;
  if (ir.stops.length > 0) body.stop = ir.stops;
  if (ir.jsonMode) body.response_format = { type: "json_object" };
  if (ir.temperature !== null) body.temperature = ir.temperature;
  if (ir.topP !== null) body.top_p = ir.topP;
  return body;
}

function anthropicImageBlock(image: IrImage): Record<string, unknown> {
  if (image.kind === "data") {
    return { type: "image", source: { type: "base64", media_type: image.mediaType, data: image.value } };
  }
  return { type: "image", source: { type: "url", url: image.value } };
}

export function buildAnthropicBody(ir: IrRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  for (const turn of ir.turns) {
    if (turn.role === "assistant") {
      messages.push({ role: "assistant", content: turn.text });
    } else if (turn.images.length > 0) {
      const blocks: unknown[] = [];
      if (turn.text !== "") blocks.push({ type: "text", text: turn.text });
      for (const image of turn.images) blocks.push(anthropicImageBlock(image));
      messages.push({ role: "user", content: blocks });
    } else {
      messages.push({ role: "user", content: turn.text });
    }
  }
  const body: Record<string, unknown> = { model, messages };
  if (ir.system !== "") body.system = ir.system;
  if (ir.tools.length > 0) {
    body.tools = ir.tools.map((t) => ({
      name: t.name,
      ...(t.description !== null ? { description: t.description } : {}),
      input_schema: t.schema,
    }));
    const choice = toolChoiceForAnthropic(ir);
    if (choice !== undefined) body.tool_choice = choice;
  }
  // max_tokens is only sent when the client sent one: an invented value
  // would change the request the provider attests to.
  if (ir.maxTokens !== null) body.max_tokens = ir.maxTokens;
  if (ir.stops.length > 0) body.stop_sequences = ir.stops;
  if (ir.temperature !== null) body.temperature = ir.temperature;
  if (ir.topP !== null) body.top_p = ir.topP;
  return body;
}

function geminiImagePart(image: IrImage): Record<string, unknown> {
  if (image.kind === "data") return { inlineData: { mimeType: image.mediaType, data: image.value } };
  return { fileData: { fileUri: image.value } };
}

export function buildGeminiBody(ir: IrRequest, model: string): { path: string; body: Record<string, unknown> } {
  const contents: Record<string, unknown>[] = [];
  for (const turn of ir.turns) {
    const parts: unknown[] = [];
    if (turn.text !== "") parts.push({ text: turn.text });
    if (turn.role === "user") for (const image of turn.images) parts.push(geminiImagePart(image));
    if (parts.length === 0) parts.push({ text: "" });
    contents.push({ role: turn.role === "assistant" ? "model" : "user", parts });
  }
  const body: Record<string, unknown> = { contents };
  if (ir.system !== "") body.systemInstruction = { parts: [{ text: ir.system }] };
  if (ir.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: ir.tools.map((t) => ({
          name: t.name,
          ...(t.description !== null ? { description: t.description } : {}),
          parameters: t.schema,
        })),
      },
    ];
  }
  const config: Record<string, unknown> = {};
  if (ir.maxTokens !== null) config.maxOutputTokens = ir.maxTokens;
  if (ir.stops.length > 0) config.stopSequences = ir.stops;
  if (ir.jsonMode) config.responseMimeType = "application/json";
  if (ir.temperature !== null) config.temperature = ir.temperature;
  if (ir.topP !== null) config.topP = ir.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return { path: `/v1beta/models/${model}:generateContent`, body };
}

// ---------- upstream answer parsing ----------

export function parseOpenAIAnswer(json: unknown): Answer {
  const root = isRecord(json) ? json : {};
  const choices = Array.isArray(root.choices) ? root.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(first.message) ? first.message : {};
  const usage = isRecord(root.usage) ? root.usage : {};
  return {
    text: typeof message.content === "string" ? message.content : openAIContentText(message.content),
    inputTokens: numOrNull(usage.prompt_tokens) ?? 0,
    outputTokens: numOrNull(usage.completion_tokens) ?? 0,
    stop: stringOrNull(first.finish_reason) ?? "stop",
  };
}

export function parseAnthropicAnswer(json: unknown): Answer {
  const root = isRecord(json) ? json : {};
  const usage = isRecord(root.usage) ? root.usage : {};
  return {
    text: anthropicBlocksText(root.content),
    inputTokens: numOrNull(usage.input_tokens) ?? 0,
    outputTokens: numOrNull(usage.output_tokens) ?? 0,
    stop: stringOrNull(root.stop_reason) ?? "end_turn",
  };
}

export function parseGeminiAnswer(json: unknown): Answer {
  const root = isRecord(json) ? json : {};
  const candidates = Array.isArray(root.candidates) ? root.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = "";
  for (const part of parts) if (isRecord(part) && typeof part.text === "string") text += part.text;
  const usage = isRecord(root.usageMetadata) ? root.usageMetadata : {};
  return {
    text,
    inputTokens: numOrNull(usage.promptTokenCount) ?? 0,
    outputTokens: numOrNull(usage.candidatesTokenCount) ?? 0,
    stop: stringOrNull(first.finishReason) ?? "STOP",
  };
}

// ---------- stop-reason mapping ----------

// Families, mirroring the reading consumers use: 0 ended, 1 length,
// 2 tool call, 3 refusal/filter. Unknown tokens fall back to 0 (ended),
// which is what every provider here produces in practice.
const STOP_FAMILY: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

export function stopForOpenAI(stop: string): string {
  return ["stop", "length", "tool_calls", "content_filter"][STOP_FAMILY[stop] ?? 0];
}

export function stopForAnthropic(stop: string): string {
  return ["end_turn", "max_tokens", "tool_use", "refusal"][STOP_FAMILY[stop] ?? 0];
}

// ---------- client response rendering ----------

const CREATED = 1700000000; // a fixed timestamp: output must be deterministic

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += 13) pieces.push(text.slice(i, i + 13));
  return pieces;
}

export function renderOpenAIJson(answer: Answer, model: string, seq: number): string {
  return JSON.stringify({
    id: `chatcmpl-agg-${seq}`,
    object: "chat.completion",
    created: CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: stopForOpenAI(answer.stop),
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
}

export function renderAnthropicJson(answer: Answer, model: string, seq: number): string {
  return JSON.stringify({
    id: `msg_agg_${seq}`,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: stopForAnthropic(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  });
}

export function renderOpenAIStream(answer: Answer, model: string, seq: number): string {
  const id = `chatcmpl-agg-${seq}`;
  const frame = (value: unknown): string => `data: ${JSON.stringify(value)}\n\n`;
  const chunk = (delta: Record<string, unknown>, finish: string | null): string =>
    frame({ id, object: "chat.completion.chunk", created: CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  let out = chunk({ role: "assistant", content: "" }, null);
  for (const piece of splitText(answer.text)) out += chunk({ content: piece }, null);
  out += chunk({}, stopForOpenAI(answer.stop));
  out += frame({
    id,
    object: "chat.completion.chunk",
    created: CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
  out += "data: [DONE]\n\n";
  return out;
}

export function renderAnthropicStream(answer: Answer, model: string, seq: number): string {
  const frame = (event: string, value: unknown): string => `event: ${event}\ndata: ${JSON.stringify(value)}\n\n`;
  let out = frame("message_start", {
    type: "message_start",
    message: {
      id: `msg_agg_${seq}`,
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
    },
  });
  out += frame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  for (const piece of splitText(answer.text)) {
    out += frame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } });
  }
  out += frame("content_block_stop", { type: "content_block_stop", index: 0 });
  out += frame("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stopForAnthropic(answer.stop), stop_sequence: null },
    usage: { output_tokens: answer.outputTokens },
  });
  out += frame("message_stop", { type: "message_stop" });
  return out;
}

// ---------- client error rendering (used only when every provider refused) ----------

export function renderOpenAIError(message: string): string {
  return JSON.stringify({ error: { message, type: "api_error", code: "no_provider_available" } });
}

export function renderAnthropicError(message: string): string {
  return JSON.stringify({ type: "error", error: { type: "api_error", message } });
}
