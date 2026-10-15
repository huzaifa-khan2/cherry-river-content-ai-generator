// mcp.js — MCP wrapper for the Cherry River AI Engine (Option A)
// Exposes the existing REST API as MCP tools over Streamable HTTP (stateless).
// Auth: signs in as a designated Supabase user (env creds) to obtain a real JWT
// that the existing /api endpoints accept via supabase.auth.getUser(token).
// No secrets in repo — everything comes from Vercel env vars.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { createClient } from "@supabase/supabase-js";
import { timingSafeEqual } from "crypto";

const MCP_SECRET = process.env.MCP_SECRET;
const MCP_USER_EMAIL = process.env.MCP_USER_EMAIL;
const MCP_USER_PASSWORD = process.env.MCP_USER_PASSWORD;

// Auth-only Supabase client. Anon key preferred; falls back to the service role
// key the app already has in Vercel so no new secret is required.
const authClient = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY,
  { auth: { persistSession: false, autoRefreshToken: false } }
);

let cachedToken = null;
let cachedExpMs = 0;
async function getAccessToken() {
  const now = Date.now();
  if (cachedToken && now < cachedExpMs - 60000) return cachedToken;
  const { data, error } = await authClient.auth.signInWithPassword({
    email: MCP_USER_EMAIL,
    password: MCP_USER_PASSWORD,
  });
  if (error || !data?.session) throw new Error("MCP auth failed: " + (error?.message || "no session"));
  cachedToken = data.session.access_token;
  cachedExpMs = data.session.expires_at ? data.session.expires_at * 1000 : now + 3600000;
  return cachedToken;
}

// Deliberately does NOT fall back to x-forwarded-host/x-forwarded-proto: those are
// client-supplied headers, and this function's result is used as the target of an
// outbound fetch carrying a real Bearer token — trusting them would let a caller
// redirect that authenticated request (and the token) to an attacker-controlled host.
function baseUrl(_req) {
  const base = process.env.PUBLIC_BASE_URL;
  if (!base) throw new Error("PUBLIC_BASE_URL must be set to use MCP tools");
  return base.replace(/\/$/, "");
}

async function apiCall(req, method, path, body) {
  const token = await getAccessToken();
  const res = await fetch(baseUrl(req) + path, {
    method,
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = text; }
  if (res.status === 401) { cachedToken = null; } // force re-auth next call
  if (!res.ok) throw new Error(`API ${method} ${path} -> ${res.status}: ${typeof json === "string" ? json : JSON.stringify(json)}`);
  return json;
}

const ok = (d) => ({ content: [{ type: "text", text: typeof d === "string" ? d : JSON.stringify(d, null, 2) }] });
const fail = (e) => ({ content: [{ type: "text", text: "ERROR: " + (e?.message || String(e)) }], isError: true });

function buildServer(req) {
  const server = new McpServer({ name: "cherry-river-ai", version: "1.0.0" });

  server.tool("get_health", "Ping the Cherry River AI Engine API. Returns {status:'ok'} when live.", {},
    async () => { try { return ok(await apiCall(req, "GET", "/api/health")); } catch (e) { return fail(e); } });

  server.tool("get_stats", "Usage stats for the authenticated Cherry River account: images, videos, products, spend this month and total.", {},
    async () => { try { return ok(await apiCall(req, "GET", "/api/stats")); } catch (e) { return fail(e); } });

  server.tool("list_products", "List the product catalogue (id, name, category, color, product_type, image, status).", {},
    async () => { try { return ok(await apiCall(req, "GET", "/api/products")); } catch (e) { return fail(e); } });

  server.tool("list_generations", "Recent generated assets (images/videos) for this account.",
    { limit: z.number().int().positive().max(200).optional().describe("Max rows to return") },
    async ({ limit }) => { try { return ok(await apiCall(req, "GET", `/api/generations${limit ? `?limit=${limit}` : ""}`)); } catch (e) { return fail(e); } });

  server.tool("list_team_generations", "Recent generated assets across the whole team.",
    { limit: z.number().int().positive().max(200).optional() },
    async ({ limit }) => { try { return ok(await apiCall(req, "GET", `/api/generations/team${limit ? `?limit=${limit}` : ""}`)); } catch (e) { return fail(e); } });

  server.tool("list_scenes", "Photography scene presets for image generation (id, name, category).", {},
    async () => { try { return ok(await apiCall(req, "GET", "/api/scenes")); } catch (e) { return fail(e); } });

  server.tool("list_concept_scenes", "Packaging-concept scene templates for a given package format.",
    { format: z.enum(["can", "bottle750", "bottle114"]).describe("Package format") },
    async ({ format }) => { try { return ok(await apiCall(req, "GET", `/api/concept-scenes?format=${format}`)); } catch (e) { return fail(e); } });

  server.tool("list_concept_options", "Available finishes, moods, cap colors and liquid-visibility options for Concept Studio.", {},
    async () => { try { return ok(await apiCall(req, "GET", "/api/concept-options")); } catch (e) { return fail(e); } });

  server.tool("list_concept_examples", "Preset style combinations (id, name) for Concept Studio.", {},
    async () => { try { return ok(await apiCall(req, "GET", "/api/concept-examples")); } catch (e) { return fail(e); } });

  // ---- Generation tools: create DRAFT assets in the account. Not public publishing. ----
  server.tool("generate_image",
    "Generate marketing product photography (Flux). Creates a DRAFT asset (~$0.055/image). Human validation required before any public use.",
    {
      productName: z.string().describe("Product name to feature"),
      productColor: z.string().optional().describe("Brand/liquid color (hex or name)"),
      productId: z.string().optional().describe("Existing product id, to use its reference photo"),
      prompt: z.string().optional().describe("Extra creative direction"),
      format: z.string().optional().describe("Aspect/format, e.g. '1:1','4:5','16:9'"),
      quality: z.string().optional(),
      quantity: z.number().int().positive().max(4).optional().describe("Number of images (default 1)"),
    },
    async (a) => { try { return ok(await apiCall(req, "POST", "/api/generate-image", a)); } catch (e) { return fail(e); } });

  server.tool("generate_collection",
    "Generate a styled collection shot featuring multiple products together (~$0.055/image). DRAFT asset.",
    { productIds: z.array(z.string()).min(1), format: z.string().optional(), prompt: z.string().optional() },
    async (a) => { try { return ok(await apiCall(req, "POST", "/api/generate-collection", a)); } catch (e) { return fail(e); } });

  server.tool("generate_concept_scene",
    "Generate ONE packaging-concept mockup (blank/unlabeled surface) for brainstorming. DRAFT asset.",
    {
      flavorName: z.string(), format: z.enum(["can", "bottle750", "bottle114"]),
      colorHex: z.string().optional(), sceneId: z.string().optional(), customScene: z.string().optional(),
      finish: z.string().optional(), mood: z.string().optional(), exampleId: z.string().optional(),
      capColor: z.string().optional(), liquidVisibility: z.string().optional(), customPrompt: z.string().optional(),
    },
    async (a) => { try { return ok(await apiCall(req, "POST", "/api/generate-concept-scene", a)); } catch (e) { return fail(e); } });

  server.tool("generate_concept_board",
    "Generate a concept board (multiple packaging-concept scenes) for one flavor. DRAFT assets.",
    {
      flavorName: z.string(), format: z.enum(["can", "bottle750", "bottle114"]),
      colorHex: z.string().optional(), finish: z.string().optional(), mood: z.string().optional(),
      exampleId: z.string().optional(), capColor: z.string().optional(),
      liquidVisibility: z.string().optional(), customPrompt: z.string().optional(),
    },
    async (a) => { try { return ok(await apiCall(req, "POST", "/api/generate-concept-board", a)); } catch (e) { return fail(e); } });

  server.tool("generate_video",
    "Generate a short cinematic product video (Seedance, ~$0.008/sec). Returns a task id; poll with get_video_status. DRAFT asset.",
    { productId: z.string(), customPrompt: z.string().optional(), duration: z.number().int().positive().max(10).optional(), format: z.string().optional() },
    async (a) => { try { return ok(await apiCall(req, "POST", "/api/generate-video", a)); } catch (e) { return fail(e); } });

  server.tool("get_video_status", "Poll the status/progress of a video generation task by id.",
    { id: z.string() },
    async ({ id }) => { try { return ok(await apiCall(req, "GET", `/api/video-status/${id}`)); } catch (e) { return fail(e); } });

  return server;
}

// Constant-time compare, length-checked first (timingSafeEqual throws on mismatched
// lengths rather than returning false).
function secretMatches(candidate) {
  if (typeof candidate !== "string" || candidate.length !== MCP_SECRET.length) return false;
  return timingSafeEqual(Buffer.from(candidate), Buffer.from(MCP_SECRET));
}

export function mountMcp(app) {
  if (!MCP_SECRET) { console.warn("[mcp] MCP_SECRET not set - /api/mcp disabled"); return; }
  app.post("/api/mcp", async (req, res) => {
    // Header only — never the query string, which ends up in server access logs,
    // proxy logs, and the Referer header of any outbound request from this page.
    const key = req.headers["x-mcp-secret"];
    if (!secretMatches(key)) {
      return res.status(401).json({ jsonrpc: "2.0", error: { code: -32001, message: "Unauthorized" }, id: null });
    }
    try {
      const server = buildServer(req);
      const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
      res.on("close", () => { transport.close(); server.close(); });
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (e) {
      console.error("[mcp] error", e);
      if (!res.headersSent) res.status(500).json({ jsonrpc: "2.0", error: { code: -32603, message: String(e?.message || e) }, id: null });
    }
  });
  const reject = (_req, res) => res.status(405).json({ jsonrpc: "2.0", error: { code: -32000, message: "Method not allowed (stateless MCP)" }, id: null });
  app.get("/api/mcp", reject);
  app.delete("/api/mcp", reject);
  console.log("[mcp] Cherry River AI MCP mounted at POST /api/mcp");
}
