import "dotenv/config";
import express from "express";
import { mountMcp } from "./mcp.js";
import cors from "cors";
import Replicate from "replicate";
import Anthropic from "@anthropic-ai/sdk";
import multer from "multer";
import { createClient } from "@supabase/supabase-js";
import sharp from "sharp";
import { execFile } from "child_process";
import { promisify } from "util";
import { tmpdir } from "os";
import { join } from "path";
import { writeFile, readFile, unlink } from "fs/promises";
import ffmpegInstaller from "@ffmpeg-installer/ffmpeg";
import { createHash } from "crypto";
import { extractAccessToken } from "./auth-token.js";

const execFileAsync = promisify(execFile);
// Use bundled ffmpeg (works on Vercel serverless) with local system ffmpeg as fallback
const FFMPEG_PATH = ffmpegInstaller.path;
const FFPROBE_PATH = FFMPEG_PATH.replace("ffmpeg", "ffprobe");

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));

// Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const BUCKET = "product-images";
const VIDEO_BUCKET = "videos";

const FLUX_COST_PER_IMAGE = 0.055;       // Replicate Flux 2 Pro, per image
const SEEDANCE_COST_PER_SECOND = 0.008;  // PiAPI Seedance 2 Max, per second

// Multer — memory storage (files go to Supabase, not local disk)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// Image generation engine — Sep 2026 upgrade from Flux 2 Pro to OpenAI's GPT-Image-2.5
// (via Replicate, not OpenAI's own API — see reference_gpt_image_2_5_replicate memory for why).
// "flux" kept reachable (not deleted) as a fallback, not currently used by any call site.
const IMAGE_MODEL_SLUGS = {
  flare: "openai/gpt-image-2.5-flare",       // fast/everyday — default for single-reference, simple scenes
  sunburst: "openai/gpt-image-2.5-sunburst", // precise/high-control — harder composited or multi-reference scenes
  flux: "black-forest-labs/flux-2-pro",
};

// GPT-Image-2.5 only accepts a fixed aspect_ratio enum (confirmed via a live 422) — unlike Flux,
// which took any "W:H" string. "4:5" (this app's portrait can/bottle format) isn't in that enum;
// "3:4" is the closest supported ratio (0.75 vs 4:5's 0.8).
const OPENAI_ASPECT_RATIOS = new Set(["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16"]);
function normalizeAspectRatioForOpenAI(aspectRatio) {
  if (OPENAI_ASPECT_RATIOS.has(aspectRatio)) return aspectRatio;
  if (aspectRatio === "4:5") return "3:4";
  if (aspectRatio === "5:4") return "4:3";
  return "1:1";
}

// Normalizes the different input schemas between GPT-Image-2.5 (openai/*) and Flux 2 Pro,
// and always returns an array of delivery URLs (length 1 unless `n` > 1).
async function runImageModel({ prompt, inputImages = [], aspectRatio = "1:1", n = 1, model = "flare" }) {
  const slug = IMAGE_MODEL_SLUGS[model] || IMAGE_MODEL_SLUGS.flare;
  const input = slug.startsWith("openai/")
    ? { prompt, input_images: inputImages, aspect_ratio: normalizeAspectRatioForOpenAI(aspectRatio), number_of_images: n, quality: "high", output_format: "png" }
    : {
        prompt,
        resolution: "4 MP",
        aspect_ratio: aspectRatio,
        output_format: "png",
        output_quality: 100,
        safety_tolerance: 5,
        guidance: 3,
        prompt_upsampling: true,
        ...(inputImages.length ? { input_images: inputImages } : {}),
      };
  const output = await replicate.run(slug, { input });
  const outputs = Array.isArray(output) ? output : [output];
  return outputs.map((o) => (o && typeof o.url === "function" ? o.url() : String(o)));
}

// Upload image buffer to Supabase Storage, return public URL
async function uploadToStorage(buffer, mimetype, filename) {
  const { data, error } = await supabase.storage
    .from(BUCKET)
    .upload(filename, buffer, { contentType: mimetype, upsert: true });
  if (error) throw new Error(`Storage upload failed: ${error.message}`);
  const { data: { publicUrl } } = supabase.storage.from(BUCKET).getPublicUrl(data.path);
  return publicUrl;
}

// Download image from URL and upload to Supabase Storage, return permanent public URL
async function saveGeneratedImage(url, filename) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch image: ${response.status}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  const mimetype = response.headers.get("content-type") || "image/png";
  return uploadToStorage(buffer, mimetype, `generated/${filename}`);
}

// Averse's plain Vodka Premium is genuinely clear/colorless — only the flavored line (Framboise,
// Mangue, Pomme) actually has a tinted liquid. Nothing in the prompt constrained liquid color before,
// so the model was free to invent one (e.g. a red tint near the base, likely pulled from the "CHERRY"
// wordmark) — flagged as a real bug found in a fresh generation. This pins the correct color per flavor.
function getAverseLiquidDescriptor(productName) {
  const n = (productName || "").toLowerCase();
  if (n.includes("framboise")) return "a deep red/pink raspberry-tinted vodka";
  if (n.includes("mangue")) return "a warm orange/amber mango-tinted vodka";
  if (n.includes("pomme")) return "a pale green apple-tinted vodka";
  return "genuinely clear, colorless vodka — never tinted red, amber, or any other color, including near the base";
}

// Averse's dome-shaped base/foot is its OWN opaque, solid-colored section (confirmed against real
// product photos, Sep 25 2026 — a lineup shot shows the base as a solid gray/white, orange, magenta,
// or green dome depending on flavor, never clear glass). Flagged as a real bug: the mid-body clear-glass
// fix (see hasTransparentGlass) had been over-applied all the way to the base, making the whole lower
// bottle read as see-through when only the mid-body should be — the base itself must stay opaque.
function getAverseDomeColor(productName) {
  const n = (productName || "").toLowerCase();
  if (n.includes("pomme") || n.includes("apple")) return "apple green";
  if (n.includes("framboise") || n.includes("raspberry")) return "deep crimson red";
  if (n.includes("mangue") || n.includes("mango")) return "warm amber orange";
  if (n.includes("limonade")) return "lemon yellow";
  if (n.includes("basilic") || n.includes("petits fruits")) return "mauve purple";
  if (n.includes("gin") || n.includes("london")) return "midnight navy";
  return "white";
}

// Enhance prompt using Claude — with vision if product image available
async function enhancePrompt(userPrompt, product) {
  const isCan = product?.product_type === "can";
  const isAverse = product?.name?.toLowerCase().includes("averse");
  const averseLiquid = isAverse ? getAverseLiquidDescriptor(product?.name) : "";
  const averseDome = isAverse ? getAverseDomeColor(product?.name) : "";
  const userContent = [];

  // Attach product image so Claude sees exact label, colors, shape
  if (product?.image) {
    try {
      const imgResp = await fetch(product.image);
      if (imgResp.ok) {
        const imgBuffer = Buffer.from(await imgResp.arrayBuffer());
        const base64 = imgBuffer.toString("base64");
        const contentType = imgResp.headers.get("content-type") || "image/jpeg";
        userContent.push({ type: "image", source: { type: "base64", media_type: contentType, data: base64 } });
      }
    } catch (e) {
      console.error("Failed to fetch product image for image prompt:", e.message);
    }
  }

  userContent.push({
    type: "text",
    text: `You are a world-class product photography director for premium beverage brands.
Study the product image carefully — memorize the exact ${isCan ? "can shape, label colors, design, pull-tab" : isAverse ? `frosted matte bottle shape, colored dome base cap, label design — this bottle has NO cork. CRITICAL, this bottle has exactly THREE zones, never confuse them: (1) the very top neck/shoulder section carrying the 'a|v|e|r|s|e' logo is frosted matte opaque white; (2) the mid-body — starting right at the 'VODKA SUPÉRIEURE PREMIUM' text and continuing down to where the bottle's rounded/flared foot begins — is genuinely CLEAR, TRANSPARENT glass: the 'VODKA SUPÉRIEURE PREMIUM' text sits directly on the clear glass (not on a white backing), so the bottle's vertical ribbed/faceted glass texture and the liquid inside are visible right through that text band too, not just further down. This mid-body zone must NEVER be described or rendered as frosted, opaque, or solid-colored — it is real see-through glass, always. IGNORE the flat white/bright studio-lighting cast visible on the reference photo's own lower glass section — that is a lighting artifact from how the catalog photo was shot, not a property of the glass itself. In the new scene, the clear mid-body glass must show genuine optical refraction of THIS scene's actual background/environment and lighting, never a plain white or milky haze copied from the reference photo. The liquid inside is ${averseLiquid} — do not invent any other liquid color. (3) the dome-shaped base/foot at the very bottom is its OWN separate, OPAQUE, solid ${averseDome}-colored section — NEVER clear or transparent glass, and it must NOT show the liquid or scene background through it, exactly like a real Averse bottle's base. CRITICAL, this dome is NARROWER than the bottle's own body — only about two-thirds of the bottle's width — sitting centered at the very bottom, with a visible margin of clear see-through glass wrapping around its left and right sides (the dome never touches or spans all the way to the bottle's outer glass walls on both sides at once; real reference photos confirm this narrower, inset dome shape, never a flat wide block filling the entire base width). The 'CHERRY RIVER' text printed on that solid ${averseDome} dome, AND the '40% ALC. / VOL. 750 ml' declaration line printed directly below it, are BOTH BLACK, not white or gray, sitting on the opaque ${averseDome} surface (not on glass)` : "bottle shape, glass color, label design, cork"}.
Enhance this prompt into a VIP professional photography brief. Keep it under 120 words.
Return ONLY the enhanced prompt, nothing else — no explanations, no labels.

Product: "${product?.name || "Premium beverage"}"
User prompt: ${userPrompt}

Rules:
- Describe THIS exact product from the image — specific colors, label details, bottle/can shape
- CRITICAL: any text or illustration actually printed on the product's real label must be reproduced verbatim from the reference photo — copy the exact wording and exact graphic/illustration style, never invent, reword, paraphrase, translate differently, or restyle any part of it (e.g. never turn a lace/vein-only illustration into a solid filled one, never make up new French/English copy that isn't on the real label)
- Keep the user's core scene/idea
- Add: exact product positioning, lighting details, camera settings (f-stop, depth of field), mood, photographic style
${isCan ? "- CRITICAL for cans: the can must be shown PERFECTLY UPRIGHT, standing vertically at 90 degrees, never tilted or angled. Label text must be fully readable and undistorted. Can is planted firmly on the surface, not floating." : ""}
${isAverse ? `- CRITICAL, always include this regardless of the user's own prompt wording: explicitly state that the top neck/shoulder 'a|v|e|r|s|e' logo band is opaque frosted white, that starting from the 'VODKA SUPÉRIEURE PREMIUM' text downward the MID-BODY only (down to where the bottle's foot/base begins) is clear, transparent, see-through glass — the 'VODKA SUPÉRIEURE PREMIUM' text sits directly on clear glass with the ribbed glass texture and liquid visible through it, not on a white background — this is a hard requirement, never omit it even if the user's prompt doesn't mention glass at all. Also explicitly state that this mid-body clear glass must show real refraction of THIS scene's own background and lighting (e.g. dark bar tones in a bar scene, warm daylight in an outdoor scene) — never a flat white, milky, or bright studio-highlight cast that ignores the scene around it. Explicitly state the liquid itself is ${averseLiquid} — never a different color. Then explicitly state that the dome-shaped base/foot at the very bottom is a SEPARATE, OPAQUE, solid ${averseDome}-colored section — never clear glass, never showing the liquid or scene through it — and that this dome is NARROWER than the bottle's body (about two-thirds of its width), centered at the very bottom with a visible margin of clear see-through glass wrapping around both its left and right sides, never spanning the bottle's full width or touching both outer glass walls at once. It carries the 'CHERRY RIVER' text and the '40% ALC. / VOL. 750 ml' line directly below it, both in solid black (not white or gray), sitting on that opaque ${averseDome} surface, unlike the white wordmark higher up on the bottle.` : ""}
- End with: "professional beverage advertising photography"`,
  });

  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 300,
    messages: [{ role: "user", content: userContent }],
  });

  return message.content[0].text.trim();
}

// Map format labels to aspect ratios
const aspectRatioMap = {
  "1:1": "1:1",
  "4:5": "4:5",
  "9:16": "9:16",
  "16:9": "16:9",
};

// Map quality labels to resolution
const qualityMap = {
  Draft: "1 MP",
  HD: "2 MP",
  "4K": "4 MP",
};

// Static scenes
const scenes = [
  { id: 1, name: "Winter Chalet", category: "Lifestyle", gradient: "from-blue-900 to-slate-700" },
  { id: 2, name: "Beach Sunset", category: "Lifestyle", gradient: "from-orange-500 to-pink-600" },
  { id: 3, name: "Mountain Lake", category: "Lifestyle", gradient: "from-emerald-800 to-cyan-700" },
  { id: 4, name: "Urban Night", category: "Lifestyle", gradient: "from-purple-900 to-gray-800" },
  { id: 5, name: "White Minimal", category: "Studio", gradient: "from-gray-100 to-white" },
  { id: 6, name: "Dark Moody", category: "Studio", gradient: "from-gray-900 to-black" },
  { id: 7, name: "Marble Surface", category: "Studio", gradient: "from-gray-300 to-gray-100" },
  { id: 8, name: "Neon Glow", category: "Studio", gradient: "from-pink-600 to-violet-800" },
  { id: 9, name: "Christmas", category: "Seasonal", gradient: "from-red-800 to-green-900" },
  { id: 10, name: "Summer Party", category: "Seasonal", gradient: "from-yellow-400 to-orange-500" },
  { id: 11, name: "Valentine's", category: "Seasonal", gradient: "from-rose-600 to-pink-400" },
  { id: 12, name: "Autumn Harvest", category: "Seasonal", gradient: "from-amber-700 to-orange-900" },
];

app.get("/api/scenes", (_req, res) => res.json(scenes));

// --- Concept Studio: packaging concept mockup generator ---
// Generates NEW packaging/label concepts for brainstorming with a designer.
// CRITICAL: never references a real product photo and never renders label text/logos —
// the generated mockup is shape + finish + color + scene only; real label artwork stays vector,
// added by the design team afterward. This also sidesteps Flux's unreliable text rendering.

const conceptShapeDescriptors = {
  can: "a tall slim 355ml cylindrical aluminum beverage can, standing perfectly upright",
  bottle750: "a 750ml glass spirit bottle with a defined neck and cap, standing perfectly upright",
  bottle114: "a 1.14L frosted matte glass bottle, TALL and SLENDER with an elongated body, straight vertical shoulders, and a narrow neck (NOT a short squat decanter or whiskey-bottle shape), with a distinctive dome-shaped pool of liquid visible near the base, standing perfectly upright",
};

// Real catalog photos used ONLY for exact shape/geometry conditioning on the master "studio" tile —
// enhanceConceptPrompt instructs Flux to copy proportions only and ignore the real label/color/text.
const conceptShapeReferenceImages = {
  can: "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/product-1776357928401.jpg", // CR Amaretto Sour 355mL
  bottle750: "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/product-1776357898479.jpg", // Cherry River Dry Gin 750mL
  bottle114: "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/product-1776357972390.jpg", // Averse Vodka Mangue 1.14L
};

// Catalog `color` values often capture a label ACCENT (e.g. mint-green text on a black can),
// not the can/bottle's actual dominant body color. Sample the real photo directly instead.
// Product shots are directionally lit (bright highlight sweeping across one side), so a plain
// median/mode over the visible surface skews toward that highlight, not the true base tone —
// e.g. a matte black can measured this way came back mid-gray. Instead, rank pixels by
// brightness and average the 10th-30th percentile band: dark enough to sit in shadow/base tone
// rather than the highlight, but not so dark it's just noise or the deepest shadow crease.
async function getDominantColor(imageUrl) {
  try {
    const res = await fetch(imageUrl);
    const imgBuf = Buffer.from(await res.arrayBuffer());
    const { data, info } = await sharp(imgBuf)
      .resize(150, 150, { fit: "cover" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const w = info.width, h = info.height, channels = info.channels;
    const x0 = Math.floor(w * 0.3), x1 = Math.floor(w * 0.7);
    const y0 = Math.floor(h * 0.2), y1 = Math.floor(h * 0.8);
    const pixels = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const idx = (y * w + x) * channels;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        pixels.push({ r, g, b, brightness: 0.299 * r + 0.587 * g + 0.114 * b });
      }
    }
    pixels.sort((a, b) => a.brightness - b.brightness);
    const lo = Math.floor(pixels.length * 0.1), hi = Math.floor(pixels.length * 0.3);
    const band = pixels.slice(lo, hi);
    const avg = (ch) => Math.round(band.reduce((s, p) => s + p[ch], 0) / band.length);
    const r = avg("r"), g = avg("g"), b = avg("b");
    return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
  } catch (e) {
    console.error("getDominantColor failed:", e.message);
    return null;
  }
}

// Prompt wording alone ("do not copy the label") isn't reliable protection against Flux
// replicating a reference image's real printed content — confirmed live: a can with a bold
// zigzag border + large "SANS ALCOOL" text leaked through in full even with an explicit
// per-word ban on "zigzag" in the prompt (Flux has no strength/weight dial on input_images,
// so wording is the only lever, and a high-contrast graphic label can just overpower it).
// A first attempt destroyed the printed content by downscaling + grayscaling + blurring the
// photo, but that still wasn't enough: a real label's color-block split (e.g. solid black body
// + a distinct accent-colored zone) is the single largest-area brightness contrast in the whole
// photo, so it survives at ANY blur radius (confirmed down to a 6px downscale + blur(40)) and
// Flux still reads the surviving brightness zone as "there's a distinct zone here that needs a
// color" — fighting the prompt's "single flat color" instruction and producing a mis-shapen
// patch instead of either a clean flat can or the real chevron.
// Real fix: don't derive the reference from the photo's actual tones at all. The catalog photos
// all sit on a pure black backdrop, so mask the object out by pixel brightness, then paint the
// interior with a synthetic per-row cylindrical highlight — same outer silhouette, proportions,
// and rim contour as the real product, but zero information about where its actual print's
// color zones sit.
const silhouetteReferenceCache = new Map();

async function getFlatSilhouette(imageUrl) {
  if (silhouetteReferenceCache.has(imageUrl)) return silhouetteReferenceCache.get(imageUrl);
  const res = await fetch(imageUrl);
  if (!res.ok) throw new Error(`Failed to fetch reference image: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const { data, info } = await sharp(buf).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height, channels } = info;
  const out = Buffer.alloc(width * height * channels);
  const isObject = (x, y) => {
    const idx = (y * width + x) * channels;
    return Math.max(data[idx], data[idx + 1], data[idx + 2]) > 8;
  };

  for (let y = 0; y < height; y++) {
    let left = -1, right = -1;
    for (let x = 0; x < width; x++) {
      if (isObject(x, y)) { if (left === -1) left = x; right = x; }
    }
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * channels;
      if (left === -1 || x < left || x > right) { out[idx] = out[idx + 1] = out[idx + 2] = 0; continue; }
      const t = (x - left) / Math.max(1, right - left);
      const highlight = Math.exp(-((t - 0.35) ** 2) / (2 * 0.22 * 0.22));
      let v = 65 + 140 * highlight;
      if (t < 0.04 || t > 0.96) v *= 0.5; // cylinder-wrap shadow at the extreme edges
      out[idx] = out[idx + 1] = out[idx + 2] = Math.round(Math.max(0, Math.min(255, v)));
    }
  }

  const processed = await sharp(out, { raw: { width, height, channels } }).blur(3).png().toBuffer();
  const filename = `concept-references/silhouette-${createHash("md5").update(imageUrl).digest("hex")}.png`;
  const url = await uploadToStorage(processed, "image/png", filename);
  silhouetteReferenceCache.set(imageUrl, url);
  return url;
}

const conceptSceneTemplates = {
  can: [
    { id: "studio", name: "Studio", description: "Clean white studio packaging reference shot", prompt: "Clean white studio background, soft even three-point lighting, the can centered, subtle shadow beneath, no props — a true packaging reference shot." },
    { id: "lifestyle", name: "Lifestyle Outdoor", description: "Natural daylight, softly blurred outdoor setting", prompt: "Outdoor lifestyle setting, natural daylight, the can held casually at chest height against a softly blurred park or patio background, candid editorial feel." },
    { id: "shelf", name: "Shelf Row", description: "Retail shelf row, repeated front-facing", prompt: "A retail cooler shelf row, the can repeated several times in a neat front-facing row alongside softly blurred neighboring cans, even shelf lighting, true-to-life retail context." },
    { id: "hero", name: "Hero Close-Up", description: "Dramatic close-up hero shot", prompt: "Dramatic hero close-up, the can large in frame with shallow depth of field, dynamic angled lighting sculpting the can's form, premium advertising composition." },
    { id: "hands", name: "Hands Holding", description: "Held naturally, label facing camera", prompt: "A hand holding the can naturally at a relaxed angle, soft daylight, the can's full label face turned toward camera, fingers not obscuring the label." },
    { id: "cooler", name: "Cooler", description: "Nestled in ice with condensation", prompt: "The can nestled in a cooler full of ice, condensation on the surface, cool blue-toned lighting, classic chilled-and-ready presentation." },
    { id: "bar", name: "Bar Counter", description: "Dark wood bar counter, warm pendant light", prompt: "Resting on a dark wood bar counter, a single warm pendant light overhead, soft bokeh bottles in the background, upscale bar presentation." },
    { id: "event", name: "Event Table", description: "Long event table, soft decor bokeh", prompt: "On a long event table alongside other drinkware, string lights or simple decor softly blurred in the background, social gathering presentation." },
  ],
  bottle750: [
    { id: "studio", name: "Studio", description: "Clean white studio packaging reference shot", prompt: "Clean white studio background, soft even three-point lighting, the bottle centered, glass refraction and the cap crisply rendered, no props." },
    { id: "lifestyle", name: "Lifestyle Outdoor", description: "Natural daylight through the glass", prompt: "Outdoor lifestyle setting, natural daylight passing through the glass and liquid, the bottle resting on a wooden table with a softly blurred garden or patio background." },
    { id: "shelf", name: "Shelf Row", description: "Retail shelf row, repeated front-facing", prompt: "A retail shelf row, the bottle repeated several times in a neat front-facing row, even shelf lighting, consistent glass reflections across the row." },
    { id: "hero", name: "Hero Close-Up", description: "Dramatic close-up hero shot", prompt: "Dramatic hero close-up, the bottle large in frame with shallow depth of field, dynamic angled rim lighting catching the glass edge and cap, premium advertising composition." },
    { id: "hands", name: "Hands Holding", description: "Held by the neck, label facing camera", prompt: "A hand holding the bottle by the neck at a relaxed angle, soft daylight passing through the liquid, the bottle's full face turned toward camera." },
    { id: "cooler", name: "Ice Bucket", description: "In a metal ice bucket with condensation", prompt: "The bottle resting in a metal ice bucket, ice cubes and condensation on the glass, cool tones, classic chilled presentation." },
    { id: "bar", name: "Bar Counter", description: "Dark wood bar, light refracting through glass", prompt: "Standing on a dark wood bar counter, a single warm pendant light overhead refracting through the glass and liquid, soft bokeh glassware in the background." },
    { id: "event", name: "Event Table", description: "Long event table, soft decor bokeh", prompt: "On a long event table alongside glassware, string lights or simple decor softly blurred in the background, social gathering presentation." },
  ],
  bottle114: [
    { id: "studio", name: "Studio", description: "Clean white studio packaging reference shot", prompt: "Clean white studio background, soft even three-point lighting, the bottle centered, frosted matte glass and dome cap crisply rendered, no props." },
    { id: "lifestyle", name: "Lifestyle Outdoor", description: "Natural daylight, softly blurred outdoor setting", prompt: "Outdoor lifestyle setting, natural daylight, the bottle resting on a wooden table with a softly blurred garden or patio background." },
    { id: "shelf", name: "Shelf Row", description: "Retail shelf row, repeated front-facing", prompt: "A retail shelf row, the bottle repeated several times in a neat front-facing row, even shelf lighting, consistent frosted-glass reflections across the row." },
    { id: "hero", name: "Hero Close-Up", description: "Dramatic close-up hero shot", prompt: "Dramatic hero close-up, the bottle large in frame with shallow depth of field, dynamic angled rim lighting sculpting the dome cap and frosted body, premium advertising composition." },
    { id: "hands", name: "Hands Holding", description: "Held naturally, label facing camera", prompt: "A hand holding the bottle naturally at a relaxed angle, soft daylight, the bottle's full face turned toward camera." },
    { id: "cooler", name: "Ice Bucket", description: "In a metal ice bucket with condensation", prompt: "The bottle resting in a metal ice bucket, ice cubes and condensation on the frosted glass, cool tones, classic chilled presentation." },
    { id: "bar", name: "Bar Counter", description: "Dark wood bar counter, warm pendant light", prompt: "Standing on a dark wood bar counter, a single warm pendant light overhead, soft bokeh glassware in the background, upscale bar presentation." },
    { id: "event", name: "Event Table", description: "Long event table, soft decor bokeh", prompt: "On a long event table alongside glassware, string lights or simple decor softly blurred in the background, social gathering presentation, generous shared-format energy." },
  ],
};

const conceptFinishes = {
  matte: "Matte finish — soft non-reflective surface, no glossy highlights, premium understated texture.",
  metallic: "Metallic finish — brushed metal sheen, cool reflective highlights, modern industrial texture.",
  glossy: "Glossy finish — high-shine reflective surface, crisp specular highlights, polished premium look.",
  frosted: "Frosted finish — translucent matte texture, soft diffused light passing through, soft premium look.",
  neon: "Bold neon finish — vivid saturated color block, high-contrast graphic look, eye-catching modern energy.",
  clean_minimal: "Clean minimal finish — flat solid pastel or neutral color, no texture distractions, editorial simplicity.",
};

const conceptMoods = {
  premium: "Premium, elevated mood — refined lighting, sophisticated color grading, upscale retail feel.",
  vibrant_summer: "Vibrant summer mood — bright saturated colors, energetic and refreshing feel.",
  natural_lifestyle: "Natural lifestyle mood — soft organic lighting, warm earthy tones, approachable everyday feel.",
  bold_festive: "Bold festive mood — high-energy colors, celebratory and eye-catching feel.",
  clean_editorial: "Clean editorial mood — neutral balanced lighting, minimal styling, gallery-quality presentation.",
};

// 15 reusable example prompts — one-click finish+mood presets, reusable across any flavor or format
const conceptExamplePrompts = [
  { id: "matte-black-premium", name: "Matte Black Premium", finish: "matte", mood: "premium", extra: "a deep black base tone" },
  { id: "metallic-summer-splash", name: "Metallic Summer Splash", finish: "metallic", mood: "vibrant_summer", extra: "bright citrus accent tones" },
  { id: "frosted-lifestyle-outdoor", name: "Frosted Lifestyle Outdoor", finish: "frosted", mood: "natural_lifestyle", extra: "soft sage and cream tones" },
  { id: "bold-neon-night", name: "Bold Neon Night", finish: "neon", mood: "bold_festive", extra: "hot pink and electric blue tones" },
  { id: "clean-minimal-studio", name: "Clean Minimal Studio", finish: "clean_minimal", mood: "clean_editorial", extra: "soft neutral beige tones" },
  { id: "glossy-premium-shine", name: "Glossy Premium Shine", finish: "glossy", mood: "premium", extra: "deep burgundy tones" },
  { id: "matte-pastel-spring", name: "Matte Pastel Spring", finish: "matte", mood: "natural_lifestyle", extra: "soft pastel pink and mint tones" },
  { id: "metallic-industrial-steel", name: "Metallic Industrial Steel", finish: "metallic", mood: "clean_editorial", extra: "cool gunmetal grey tones" },
  { id: "frosted-festive-sparkle", name: "Frosted Festive Sparkle", finish: "frosted", mood: "bold_festive", extra: "icy silver and gold accent tones" },
  { id: "neon-pop-art", name: "Neon Pop Art", finish: "neon", mood: "vibrant_summer", extra: "bold yellow and magenta tones" },
  { id: "glossy-tropical-citrus", name: "Glossy Tropical Citrus", finish: "glossy", mood: "vibrant_summer", extra: "vivid orange and lime tones" },
  { id: "matte-midnight-luxury", name: "Matte Midnight Luxury", finish: "matte", mood: "premium", extra: "deep midnight navy tones" },
  { id: "clean-botanical-fresh", name: "Clean Botanical Fresh", finish: "clean_minimal", mood: "natural_lifestyle", extra: "soft botanical green tones" },
  { id: "metallic-sunset-coastal", name: "Metallic Sunset Coastal", finish: "metallic", mood: "natural_lifestyle", extra: "warm coral and gold tones" },
  { id: "glossy-holiday-gold", name: "Glossy Holiday Gold", finish: "glossy", mood: "bold_festive", extra: "rich gold and deep red tones" },
];

app.get("/api/concept-scenes", (req, res) => {
  const format = req.query.format;
  const templates = conceptSceneTemplates[format];
  if (!templates) return res.status(400).json({ error: "Invalid format. Use can, bottle750, or bottle114." });
  res.json(templates.map(({ id, name, description }) => ({ id, name, description })));
});

app.get("/api/concept-options", (_req, res) => {
  res.json({
    finishes: Object.keys(conceptFinishes).map((id) => ({ id, label: id.replace(/_/g, " ") })),
    moods: Object.keys(conceptMoods).map((id) => ({ id, label: id.replace(/_/g, " ") })),
    capColors: Object.keys(conceptCapColors).map((id) => ({ id, label: id.replace(/_/g, " ") })),
    liquidVisibility: Object.keys(conceptLiquidVisibility).map((id) => ({ id, label: id.replace(/_/g, " ") })),
  });
});

app.get("/api/concept-examples", (_req, res) => {
  res.json(conceptExamplePrompts.map(({ id, name }) => ({ id, name })));
});

// Claude call for a packaging concept — no product photo reference, explicitly bans rendered text/logos
const conceptCapColors = {
  black: "a glossy black cap",
  gold: "a polished gold cap",
  silver: "a brushed silver cap",
  wood: "a natural wood-grain cap with warm brown tones and visible grain texture",
  matches_body: "a cap that matches the bottle's body color and finish",
};

const conceptLiquidVisibility = {
  visible: "the liquid inside should be visible through the glass, catching light naturally",
  opaque: "the bottle should read as opaque/frosted, the liquid inside not clearly visible",
};

async function enhanceConceptPrompt({ flavorName, colorHex, colorRole, shapeDescriptor, finishText, moodText, sceneText, extra, capText, liquidText, referenceMode }) {
  const colorInstruction = !colorHex
    ? ""
    : colorRole === "liquid"
    ? `The glass itself must render as fully clear, colorless, uncolored glass — no tint on the glass. The LIQUID visible inside should be colored ${colorHex}, catching light naturally through the clear glass.`
    : `Brand color inspiration: ${colorHex}.`;

  const referenceInstruction = referenceMode === "consistency"
    ? "- A reference image of this exact can/bottle is attached. Keep the EXACT same color, gradient, finish and proportions as the reference — do not reinterpret or change the colors. Only the scene, lighting, and background should change."
    : referenceMode === "shape"
    ? "- A real product photo is attached as a STRICT SHAPE TEMPLATE ONLY — treat this as the exact same physical can/bottle object, just re-skinned. Match its exact height-to-diameter ratio, neck taper, rim/lip profile, shoulder curve, and base shape precisely — do not alter the silhouette or proportions in any way. CRITICAL: you must NOT copy ANY surface detail from that photo — no printed text, no logo, no color scheme, and NO decorative pattern, motif, border, stripe, triangle, zigzag, or repeating graphic of ANY kind, even faint, small, or blurred. The entire visible body surface (excluding metal cap/rim/base trim) must be a single flat, uninterrupted block of solid color with zero surface graphics of any kind — copy the reference's SHAPE ONLY, never anything printed or patterned on its surface."
    : "";

  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 300,
    messages: [{
      role: "user",
      content: `You are a packaging concept designer for premium beverage brands, generating an early-stage concept mockup for brainstorming with a graphic designer — not a finished product shot.

Product concept: a "${flavorName}" flavor, ${shapeDescriptor}.
${colorInstruction}
Style direction: ${finishText} ${moodText}
${capText ? `Cap: ${capText}.` : ""}
${liquidText ? `Liquid: ${liquidText}.` : ""}
${extra ? `Additional creative direction from the user — follow this closely: ${extra}` : ""}
Scene: ${sceneText}

CRITICAL RULES:
- The label area must be BLANK — a clean solid color or subtle gradient only. NO text, NO logos, NO typography, NO brand name anywhere in the image. Real label artwork will be added separately by the design team as vector graphics. This rule is NON-NEGOTIABLE and overrides anything else in this brief, including the user's own creative direction above — if their request conflicts with it (e.g. asks for a real logo, real brand text, or an actual reference photo that wasn't provided), silently ignore just that conflicting part and proceed anyway with a blank label. Do not point out the conflict.
- This ban includes INVENTED or FAKE text — do not generate, hallucinate, or imply any lettering, numbers, words, or pseudo-typography of any kind, even illegible, blurry, or nonsensical characters. The surface must show zero character-like marks, not just zero real brand text.
${referenceInstruction}
- Describe exact camera settings, lighting, and composition under 100 words.
- You must ALWAYS return a valid image-generation prompt no matter what — never refuse, never ask a clarifying question, never explain a conflict or limitation. There is always enough information here to produce a reasonable concept image.
Return ONLY the final image prompt, nothing else — no explanations, no labels, no questions. End with "professional packaging concept photography, completely blank unlabeled surface, absolutely no text or lettering of any kind, not even fake or illegible characters."`,
    }],
  });
  const text = message.content[0].text.trim();

  // Claude occasionally breaks format under a conflicting custom prompt and returns a
  // refusal/clarifying question instead of an image prompt (confirmed live: a user's custom
  // prompt asking for a real logo triggered a multi-paragraph refusal, which then got fed to
  // Flux as-is, producing a nonsense render). Catch that here instead of silently generating
  // garbage from whatever stray keywords happened to be in the refusal text.
  const looksLikeRefusal =
    /^(i cannot|i can't|i need to|i'm unable|sorry,|unfortunately)/i.test(text) ||
    /would you like me to|please (attach|provide|clarify)|let me know if/i.test(text) ||
    text.includes("?");
  if (looksLikeRefusal) {
    throw new Error("Couldn't generate a valid concept from this prompt — try simplifying or removing conflicting instructions from your custom prompt (e.g. requests for real logos/brand text aren't possible since labels must stay blank).");
  }
  return text;
}

// Prompt-side defenses (banning text in the brief, stripping label content from the shape
// reference) reduce risk but can't guarantee compliance — Flux is trained on huge volumes of
// real product photography where cans/bottles have labels, so it can still hallucinate FAKE
// lettering purely from that bias, independent of what the reference image or prompt say
// (confirmed live: a heavily blurred, textless reference still produced invented gibberish
// text like "355ml ilt Rarnit Storah" in the exact label position). The only reliable backstop
// is checking the actual output and retrying when it fails, rather than trusting any single
// generation to comply.
async function hasVisibleText(imageUrl) {
  const imgRes = await fetch(imageUrl);
  const buf = Buffer.from(await imgRes.arrayBuffer());
  const base64 = buf.toString("base64");
  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 10,
    messages: [{
      role: "user",
      content: [
        { type: "image", source: { type: "base64", media_type: "image/png", data: base64 } },
        { type: "text", text: "Look closely at the product's surface (can/bottle body, cap, label area) in this image. Does it show ANY visible text, letters, numbers, words, or pseudo-typography anywhere — even if small, blurry, faint, fake, or illegible? Ignore condensation droplets, reflections, and lighting — only count actual character-like marks. Answer with exactly one word: YES or NO." },
      ],
    }],
  });
  const answer = (message.content[0].text || "").trim().toUpperCase();
  return answer.startsWith("YES");
}

const MAX_CONCEPT_ATTEMPTS = 5;

// referenceImageUrl/productImageUrl are user-supplied and get fetched server-side
// (getDominantColor/getFlatSilhouette) or handed to Replicate directly — without this check
// a caller could point either at an internal/private address (SSRF). Only our own Supabase
// storage ever legitimately produces these URLs, so restrict to that host.
function isTrustedImageUrl(url) {
  try {
    const u = new URL(url);
    if (u.protocol !== "https:") return false;
    return u.hostname === new URL(process.env.SUPABASE_URL).hostname;
  } catch {
    return false;
  }
}

async function generateConceptMockup({ flavorName, colorHex, format, sceneId, customScene, finish, mood, exampleId, capColor, liquidVisibility, referenceImageUrl, productImageUrl, customPrompt, userId, userEmail }) {
  const shapeDescriptor = conceptShapeDescriptors[format];
  if (!shapeDescriptor) throw new Error("Invalid format");
  if (referenceImageUrl && !isTrustedImageUrl(referenceImageUrl)) throw new Error("Invalid reference image URL");
  if (productImageUrl && !isTrustedImageUrl(productImageUrl)) throw new Error("Invalid product image URL");

  let sceneText;
  if (customScene && customScene.trim()) {
    sceneText = customScene.trim();
  } else {
    const scene = (conceptSceneTemplates[format] || []).find((s) => s.id === sceneId);
    if (!scene) throw new Error("Unknown scene template");
    sceneText = scene.prompt;
  }

  let finishKey = finish, moodKey = mood, extra = customPrompt || null;
  if (exampleId) {
    const example = conceptExamplePrompts.find((e) => e.id === exampleId);
    if (!example) throw new Error("Unknown example prompt");
    finishKey = example.finish; moodKey = example.mood; extra = example.extra;
  }
  const finishText = conceptFinishes[finishKey];
  const moodText = conceptMoods[moodKey];
  if (!finishText || !moodText) throw new Error("Invalid finish or mood");

  const isBottle = format === "bottle750" || format === "bottle114";
  const liquidKey = isBottle ? liquidVisibility || "visible" : null;
  const liquidText = isBottle ? conceptLiquidVisibility[liquidKey] || conceptLiquidVisibility.visible : null;

  // Bottles whose glass reads as clear (the "visible" liquid default) have no real body
  // color left for "matches body" to point at — Claude was resolving it to the LIQUID color
  // instead, producing e.g. a wine-red cap on a bottle whose real cap is natural wood, unrelated
  // to liquid color. Fall back to wood (the actual closure used across the real Cherry River
  // 750mL/1.14L line) for "matches_body" specifically when the glass is clear; opaque-coated
  // bottles (liquidVisibility "opaque") keep matching their real body color as before.
  const capKey = capColor || "matches_body";
  const capText = !isBottle
    ? null
    : capKey === "matches_body" && liquidKey !== "opaque"
    ? conceptCapColors.wood
    : conceptCapColors[capKey] || conceptCapColors.matches_body;

  // Priority: an already-generated board tile (color/finish consistency) beats the user's
  // selected catalog product photo (shape only), which beats the generic per-format stand-in.
  const shapeReferenceUrl = referenceImageUrl || productImageUrl || conceptShapeReferenceImages[format];
  const referenceMode = referenceImageUrl ? "consistency" : shapeReferenceUrl ? "shape" : null;

  // The catalog's stored `color` is often a label ACCENT, not the product's true dominant
  // body color — sample the real photo directly when one exists (skip for board-consistency
  // tiles, which already inherit their color from the master tile's own prompt).
  let effectiveColorHex = colorHex;
  if (productImageUrl && !referenceImageUrl) {
    const dominant = await getDominantColor(productImageUrl);
    if (dominant) effectiveColorHex = dominant;
  }

  // Bottles whose glass reads as clear (the "visible" liquid default) have no real body
  // color to extract — getDominantColor's crop instead samples whatever's behind the glass
  // (liquid, or the dark studio backdrop bleeding through), which produced a false gray-blue
  // glass tint on real clear bottles (confirmed: two unrelated clear bottles both landed on
  // #747e8a). Redirect the extracted tone to the liquid instead, and force the glass itself
  // to stay clear/colorless. Opaque-coated bottles (liquidVisibility "opaque") keep the old
  // body-tint behavior, which is correct for them (verified on Averse Rhum Ambré).
  const colorRole = isBottle && liquidKey !== "opaque" ? "liquid" : "body";

  const enhancedPrompt = await enhanceConceptPrompt({
    flavorName, colorHex: effectiveColorHex, colorRole, shapeDescriptor, finishText, moodText, sceneText, extra, capText, liquidText,
    referenceMode,
  });

  // "shape" mode references a REAL product photo (or generic stand-in) — strip its printed
  // content before the model sees it. "consistency" mode references our OWN previously-generated
  // (already blank-label) board tile, so there's no real label to leak — skip processing.
  const inputImages = shapeReferenceUrl
    ? [referenceMode === "shape" ? await getFlatSilhouette(shapeReferenceUrl) : shapeReferenceUrl]
    : [];

  // Generate, then verify with a vision check before trusting the output — retry on failure
  // rather than risk shipping a mockup with hallucinated text (see hasVisibleText comment above).
  // Sunburst (not Flare) — this path leans on precise reference-image conditioning to keep the
  // label genuinely blank, which is exactly Sunburst's "high-control editing" strength.
  let replicateUrl, attempts = 0, textDetected = true;
  while (attempts < MAX_CONCEPT_ATTEMPTS) {
    attempts++;
    [replicateUrl] = await runImageModel({
      prompt: enhancedPrompt,
      inputImages,
      // Cans/bottles are tall & slim — a square canvas forces the model to squash the real
      // proportions to fit the frame. Portrait 4:5 gives the subject room to render true.
      aspectRatio: "4:5",
      model: "sunburst",
    });
    textDetected = await hasVisibleText(replicateUrl);
    if (!textDetected) break;
    console.warn(`Concept mockup attempt ${attempts}/${MAX_CONCEPT_ATTEMPTS} for "${flavorName}" showed visible text/lettering — retrying...`);
  }
  if (textDetected) {
    throw new Error(`Generated ${MAX_CONCEPT_ATTEMPTS} attempts and each still showed unwanted text or lettering on the label — try a different style, scene, or reference, or just retry.`);
  }

  const filename = `concept-${Date.now()}-${Math.random().toString(36).slice(2, 6)}.png`;
  let image;
  try {
    image = await saveGeneratedImage(replicateUrl, filename);
  } catch (e) {
    console.error("Failed to save concept mockup to storage:", e.message);
    image = replicateUrl;
  }

  const row = {
    id: Date.now().toString() + Math.random().toString(36).slice(2, 6),
    url: image,
    product: `Concept: ${flavorName}`,
    color: effectiveColorHex || "#888888",
    prompt: enhancedPrompt,
    type: "image",
    date: new Date().toISOString(),
    user_id: userId,
    user_email: userEmail,
    cost_usd: Math.round(FLUX_COST_PER_IMAGE * attempts * 1000) / 1000,
  };
  const { data: savedGeneration, error } = await insertGeneration(row);
  if (error) throw error;

  return { image, enhancedPrompt, savedGeneration, effectiveColorHex };
}

// --- Part B: Real-logo brand concept generator (Sep 2026) ---
// Kept fully separate from generateConceptMockup above rather than a branch inside it — the
// prompt rule here is the INVERSE of the blank-label rule (apply the real logo faithfully,
// never blank it), so sharing one function would mean threading a flag through every line of
// that rule instead of just writing the opposite one. Francis explicitly asked (Sep 12 2026,
// see project memory) to reverse the June 24 blank-label spec for this one flow only —
// Generate Image / Concept Studio's blank-label discipline is untouched.
//
// Needs the `brands` table (see drafts/2026-09-13-brands-table.sql — not yet run) and a real
// logo image per brand (not yet supplied by Francis) before this can produce a real result.

async function getBrand(brandId) {
  const { data, error } = await supabase.from("brands").select("*").eq("id", brandId).maybeSingle();
  if (error || !data) throw new Error("Unknown brand — pick a brand first.");
  if (!data.logo_url) throw new Error(`No logo on file for "${data.name}" yet — this brand needs a real logo image uploaded before concepts can be generated.`);
  return data;
}

// One clean, unambiguous real can per known real archetype, instead of dumping every
// real can in as a reference. This brand's 24 real cans actually span several visually
// CONTRADICTORY treatments (matte-black-band body vs. full-color-gradient body; 3
// different top/bottom pattern styles: minimal-geometric, fruit-icon-wrap, line-art-
// silhouette) — feeding all of them at once gives the model 20+ conflicting examples to
// average across, diluting exactly the "pick ONE style" decision the prompt asks it to
// make. A handful of curated exemplars — one clean real photo per archetype — gives it
// an unambiguous real example of each option instead, without losing real-family
// grounding (still real photos, not invented) or falling back to only 1-2 images (which
// reintroduces the earlier "model invents an unrelated design" failure mode).
const curatedCanExemplars = {
  "Cherry River": [
    "CR Amaretto Sour 355mL", // style (a): minimal geometric top/bottom border
    "CR Vodka Limonade 355mL", // style (b): fruit-icon wrap, top+bottom bands
    "CR Sangria Rouge 355mL", // style (c): line-art silhouette + full-gradient body
    "CR Mocktail Paloma SA 355mL", // SANS ALCOOL / mocktail metallic-bar formula
    "CR Margarita 355mL", // dot+triangle border variant, tribal-spike typography, 3-line category tag, faint tonal line-art inside the bottom block
  ],
};

async function getBrandCanReferenceImages(brandId, brandName) {
  const { data } = await supabase.from("products").select("name, image").eq("brand_id", brandId).eq("product_type", "can").not("image", "is", null);
  // Six-pack/collection box shots share product_type "can" with individual cans (no separate
  // type in the DB — confirmed via direct query, e.g. Cherry River's "CR Mixologie 6-Pack SA"
  // rows), but they're multi-can box photography, not a single can's label design — feeding them
  // in teaches the model the wrong structure entirely. Filter them out by name.
  const realCans = (data || []).filter((p) => p.image && !/6-pack/i.test(p.name || ""));
  const curatedNames = curatedCanExemplars[brandName];
  if (curatedNames) {
    const curated = curatedNames
      .map((name) => realCans.find((p) => p.name === name))
      .filter(Boolean)
      .map((p) => p.image);
    // Only use the curated subset if every named exemplar was actually found (product
    // renamed/deleted otherwise silently shrinks the reference set) — fall back to the
    // full real set rather than risk missing a style archetype entirely.
    if (curated.length === curatedNames.length) return curated;
  }
  return realCans.map((p) => p.image);
}

// Inverse of hasVisibleText above: here the brand's real wordmark SHOULD appear, so the
// safety net checks for the opposite failure — blank, garbled, or wrong-brand lettering.
async function hasCorrectLogo(imageUrl, brandName) {
  const imgRes = await fetch(imageUrl);
  const buf = Buffer.from(await imgRes.arrayBuffer());
  const base64 = buf.toString("base64");
  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 10,
    messages: [{
      role: "user",
      content: [
        { type: "image", source: { type: "base64", media_type: "image/png", data: base64 } },
        { type: "text", text: `Look at the label/logo area on this can or bottle. Does it clearly show a real, legible "${brandName}" brand wordmark or logo — not blank, not garbled/gibberish lettering, not a different or invented brand name? Answer with exactly one word: YES or NO.` },
      ],
    }],
  });
  return (message.content[0].text || "").trim().toUpperCase().startsWith("YES");
}

// Cherry River's real BOTTLE identity is NOT one unified system — confirmed against 5 real
// photos (Dry Gin, Gin Framboise & Lime, Gin Litchi & Tangerine, Rhum Ambré, Tequila Silver,
// Vodka Érable) it varies distinctly by spirit category in shape, cap, body treatment, AND
// label structure. The can-only style_notes (audited separately from 27 real cans, see the
// Sep 15 brand.style_notes update) do NOT apply to bottles — "Don't bottle it!" and the
// black-body-with-red-accent system never appear on a single real CR bottle, only cans.
const cherryRiverBottleStyleNotesByCategory = {
  gin: `Real Cherry River Gin bottles: an apothecary-style bottle (wide rounded shoulders, short neck) with a WOOD-GRAIN cap over a visible cork stopper. The bottle is genuinely CLEAR glass and the gin's own real color tints the entire visible liquid/glass (e.g. rose-pink for a raspberry gin, warm amber-orange for a citrus gin) — never opaque or solid-colored. A faint, tone-on-tone (barely darker than the liquid tint) illustrated wrap covers the whole visible glass body: mountains, pine trees, a leaping animal, and flavor-relevant fruit line-art, drawn as delicate engraved/etched linework, subtle enough to read as texture rather than a bold graphic. A circular quality stamp reading "PRODUIT ÉLABORÉ AU QUÉBEC — PRODUCT BLENDED IN QUÉBEC" sits on the clear glass near the shoulder. ONE rectangular label patch with a deliberately torn/deckle top-and-bottom edge, colored in the flavor's accent color, holds: "CHERRY RIVER" in the bold stacked slab-serif wordmark, "GIN" beneath it, then the bilingual flavor name (French/English) beneath that. Near the base, small text directly on the glass shows "40% alc./vol." and "750 ml" plus a tiny GPS-coordinate-style line and a barcode. NO "Don't bottle it!" tagline anywhere — that phrase is exclusive to the can line, never a bottle.`,
  rhum: `Real Cherry River Rhum bottles: a rounded-shoulder cylindrical bottle with a WOOD-GRAIN cap over a visible cork stopper. The upper half is clear glass showing the rum's own real amber/dark liquid color through it, with the "CHERRY RIVER" wordmark rendered as bold outlined/pinstripe-style lettering directly on the clear glass (a faint mirrored/reversed decorative text may show through from behind, as subtle texture only). The lower half is a solid OPAQUE BLACK band holding: "RHUM" in bold white caps, the flavor word in a smaller white italic cursive script beneath it, then the English equivalent in the same bold-caps/cursive pairing (e.g. "RHUM" / "ambré" / "amber" / "RUM"), and "40% alc./vol. | 750 mL" beneath that. NO illustration wrap, NO torn label patch, NO "Don't bottle it!" tagline.`,
  tequila: `Real Cherry River Tequila bottles: a distinctive wide-based pyramid/decanter-shaped bottle (broad tapering body narrowing to the shoulder) with a WOOD-GRAIN cap over a visible cork stopper. "CHERRY RIVER" appears embossed/pressed directly into the clear glass near the shoulder (subtle, same color as the glass, barely visible). The lower two-thirds is a FROSTED WHITE/translucent band holding: a fine line-art illustration of an agave plant, the "CHERRY RIVER" wordmark directly beneath it, then "TEQUILA" in bold caps with the variant name in a lighter weight beneath, "FAIT AU MEXIQUE / MADE IN MEXICO" in small caps, and "750mL" / "40% alc./vol." in the bottom corners. NO nature-scene illustration wrap, NO torn label patch, NO "Don't bottle it!" tagline.
  FLAVOR-SPECIFIC COLOR VARIATION (this is the one thing that changes between tequila flavors, otherwise every new flavor looks identical to plain Silver, which is wrong): the real "Silver" variant uses teal/blue as its accent (agave line-art + wordmark + variant-name text), but a NEW flavored variant must shift this ONE accent color to genuinely match its flavor character — e.g. a chili/pepper flavor uses a warm chili-red accent, a citrus flavor uses a lime-green accent, a berry flavor uses a deep berry-pink/red accent — recolor the agave line-art, "CHERRY RIVER" wordmark, and variant-name text together in that new flavor-matched hue (keep them all the same new color, don't mix). If the flavor plausibly tints the spirit itself (e.g. a fruit-infused tequila), the liquid visible through the clear glass may also carry a faint matching tint — a plain/unflavored variant stays clear. Do not skip this step: reusing the exact same teal/blue Silver coloring for a differently-named flavor is a real, recurring failure.`,
  vodka: `Real Cherry River Vodka bottles: a tall cylindrical wine-bottle-style silhouette with a plain BLACK screw cap (not wood, unlike this brand's other spirits). The upper body is a solid MATTE BLACK opaque coating with a gold-foil maple-leaf illustration and the "CHERRY RIVER" wordmark in pale blue-gray beneath it. CRITICAL, that maple leaf is a SKELETONIZED/vein-only foil illustration, like a lace botanical engraving: only the leaf's veins and outline are solid gold, the space between the veins is negative/see-through, revealing the dark bottle body behind it — it is NEVER a solid, fully-filled gold leaf silhouette (a common, confirmed-wrong generation). The lower band is a cream/tan color with a repeated tone-on-tone pattern of that SAME skeletonized-leaf motif (like a subtle wallpaper print, not solid leaves), holding "VODKA" and the flavor name (e.g. "ÉRABLE") plus a small bilingual description line — for the maple/"Érable" flavor this line is EXACTLY "VODKA AROMATISÉE À L'ÉRABLE" / "MAPLE FLAVOURED VODKA", copied verbatim, never invented or reworded (a confirmed-wrong generation produced garbled invented French here, e.g. "AU-AAROME NATUREL D'ÉRABLE" — that exact phrasing is wrong and must never be used). Near the base, a rounded gold dome-shaped zone carries the specs ("41% alc./vol." and "750 ml"). NO nature-scene illustration wrap, NO torn label patch, NO wood cap, NO "Don't bottle it!" tagline.`,
};

function getCherryRiverBottleCategory(flavorName) {
  const n = flavorName.toLowerCase();
  if (n.includes("gin")) return "gin";
  if (n.includes("rhum") || n.includes("rum")) return "rhum";
  if (n.includes("tequila")) return "tequila";
  if (n.includes("vodka")) return "vodka";
  return null;
}

// One real photo per category — the generic conceptShapeReferenceImages.bottle750 (Dry Gin)
// is the right shape ONLY for gin; Rhum/Tequila/Vodka have their own distinct real bottle
// shapes (rounded cylinder, pyramid decanter, wine-bottle silhouette respectively).
const cherryRiverBottleShapeReferenceImages = {
  gin: "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/product-1776357898479.jpg", // Cherry River Dry Gin 750mL
  rhum: "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/product-1776357911053.jpg", // CR Rhum Ambré 750mL
  tequila: "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/product-1776357913761.jpg", // CR Tequila Silver 750mL
  vodka: "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/product-1776357915165.jpg", // CR Vodka Érable 750mL
};

async function enhanceBrandConceptPrompt({ flavorName, brandName, styleNotes, shapeDescriptor, designFeedback, realCanReferenceCount = 0, hasParentBrandLogo = false, strictStyleNotesOnly = false, includeIconStyleRefs = false }) {
  const hasRealCanFamily = realCanReferenceCount > 0;
  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 350,
    messages: [{
      role: "user",
      content: `You are a packaging concept designer generating a new flavor concept that must look like a real, market-ready product from the "${brandName}" line.

Product concept: a "${flavorName}" flavor, ${shapeDescriptor}.
${styleNotes ? `Brand visual identity to follow closely: ${styleNotes}` : ""}
${hasRealCanFamily
  ? `${realCanReferenceCount} real reference photos are attached, plus the brand's logo file — each one a DIFFERENT real archetype this brand's actual existing can line uses (not near-duplicates of each other, and not the brand's full catalog). These reference photos are not inspiration — they ARE the brand's real, approved packaging system already on shelves; your job is to pick whichever ONE of these attached archetypes fits this new flavor best, not to blend elements from all of them together.`
  : `Two real reference images are attached: the brand's actual logo/label design, and a real product photo as a shape/proportion template.`}
${hasParentBrandLogo ? `\nOne additional small reference image is attached: the real "CHERRY RIVER" parent-brand logo (a separate, distinct wordmark from the main brand's own logo above). This brand is a Cherry River sub-brand, and real bottles in this line carry a small "CHERRY RIVER" credit near the base — render that credit using this exact attached logo image, copied faithfully at small scale, NOT redrawn in a generic or different font. This is a small secondary detail only — it must never replace, resize to match, or visually compete with the main brand's own logo/wordmark, which stays the dominant identity on the bottle.` : ""}
${includeIconStyleRefs ? `\nTHE LAST TWO ATTACHED IMAGES are tight close-up crops of real fruit/berry icons from this can family, showing ONLY the icon artwork with no can shape or wordmark in view — study these two crops closely, they are the exact flat rendering style your icons must match: completely flat solid-color shapes, no gloss, no gradient, no dimensional shading of any kind, like cut vinyl sticker art. This is more reliable than the text description alone — copy the flatness level shown in these two crops exactly.` : ""}

CRITICAL RULES:
- Apply the attached brand logo/label design FAITHFULLY and LEGIBLY onto the can/bottle — do not redraw it in a different style, distort it, or invent a different logo/brand name. This is the OPPOSITE of a blank-label concept: the real brand identity must be clearly visible and correctly reproduced.
- Incorporate the flavor name "${flavorName}" and flavor-appropriate color/imagery around the existing brand identity — never replacing it.
- Match the shape reference photo's proportions exactly (height-to-diameter ratio, cap, base) — this is a real packaging mockup, not an abstract shape.
${hasRealCanFamily ? `- REPLICATE THE REAL FAMILY'S EXACT STRUCTURE (most important rule) — study the attached real can photos closely: this real line uses SEVERAL different pattern styles across its flavors, and you must pick whichever ONE fits this new flavor best, copying it precisely rather than defaulting to the same choice every time:
  (a) Minimal geometric: a thin tribal/zigzag/triangle line pattern in the accent color, only near the top crimp and repeated in a darker tone in the bottom chevron band (e.g. the Tiki, Amaretto Sour style) — a real variant of this same style alternates solid/outlined triangle peaks with a small dot centered above each gap between peaks (the Margarita style); either plain zigzag or triangle+dot is correct, pick whichever reads cleaner at this can's scale.
  (b) Fruit-icon wrap: clean VECTOR ILLUSTRATION icons across the top band AND bottom chevron band, on a solid accent-color background (the Gin Limonade / Vodka Limonade style) — use this when the flavor is fruit-forward. Icon rule verified directly from the real cans (e.g. the real "Gin Limonade Petits Fruits et Basilic" can shows FOUR distinct icon types — a lime slice, a blackberry, a small red berry, a blueberry — each its own color): use ONE icon type per named ingredient in the flavor (a flavor naming 3 ingredients gets 3 icon types, each in its own true-to-life color, not forced into one hue family). Do not scatter in unrelated ingredients that aren't part of the flavor name, and don't invent more icon types than named ingredients — but DO give every named ingredient its own icon rather than collapsing them into one.
  BAND HEIGHT + ICON BLEED (a recurring failure point, verified directly against the real cans): the solid-color band itself (top and bottom) must stay SHORT and compact — in the real references it's a narrow strip, clearly less than the height of the icons sitting on it. NEVER make the band taller to fit the icons inside it — that is the single most common mistake and is wrong. Instead, keep the band short and let the large decorative icons BLEED/OVERFLOW past the band's own edge, extending up into the dark can body above (for the top band) or down into the dark body below (for the bottom band) — the icons create the visual height through this overflow, not the band itself. Icons should look layered ON TOP of the boundary between band and dark body, partly inside the color band and partly spilling out over the black area, exactly like the real cans — never fully contained/clipped inside the band's own rectangle. Keep a comfortable gap between the overflowing icons and the main text block (wordmark, flavor name) — they must not touch or crowd the typography.
  (c) Subtle line-art silhouette: delicate outlined vector illustrations of the flavor's ingredients (fruit slices, leaves) drawn faintly in a barely-darker flat tone directly on the can body itself, behind/around the flavor name, rather than confined to a top/bottom band (the Sangria Rouge style, which also uses a sunburst/ray pattern as its top border instead of zigzag).
  Pick the ONE style (a, b, or c) that a real designer would choose for this specific flavor's actual ingredients, and execute it precisely — don't blend all three or invent a new one.
  RENDERING STYLE for icons in (b) and (c): render every icon/illustration as FLAT, CLEAN VECTOR GRAPHIC DESIGN — solid flat fill colors, crisp clean outlines, simple 2D shapes, like a printed sticker/icon set. A little internal linework (segment lines, leaf veins) and at most 2 tones of one hue for gentle shading is fine, matching the real cans (e.g. the real Vodka Limonade lime slice: solid green fill, white segment lines, one thin outline, only very subtle tonal variation). What is NOT allowed: glossy wet highlights, realistic 3D roundness/volume, painterly airbrushed texture, or camera-style depth — anything that makes the icon look like a photograph or a polished 3D render instead of a flat printed graphic. If an icon is coming out looking shiny or dimensional, simplify it toward the plainer end (a flat solid-color shape with minimal internal detail, like the Gin Limonade berries) rather than adding more rendering detail.
(1) a top band/border in the new accent color using the chosen pattern style, (2) small bilingual category text, ALWAYS in that SAME new accent color (never a fixed gold/amber — the real cans each color this tag to match their own flavor's accent, e.g. red for one flavor, yellow-green for another, coral for another — gold is NOT the rule) — either a 2-line form ("COCKTAIL" / "PRÊT-À-BOIRE") or a 3-line form ("READY-TO-DRINK" / "COCKTAIL" / "PRÊT-À-BOIRE", the Margarita style — English category word, then its French-only line, then the bilingual "PRÊT-À-BOIRE"), pick whichever reads cleaner, (3) the bold white/silver stacked "CHERRY RIVER" wordmark exactly as reference — unchanged, (4) the flavor name in LARGE stacked caps in the new accent color — pick ONE distinctive display treatment per flavor from the real range of styles this brand actually uses (this real line deliberately gives EVERY flavor its own different font mood, never reusing the same one twice): a casual script/cursive, an outline/stroke-only geometric face, a bold serif slab caps, a monospace/typewriter caps, a distressed/grunge slab caps, a thin elegant serif caps, a distressed stamped/rubber-stamp serif, a tribal stencil face (small perpendicular tick/spike accents on the stems of certain letters, e.g. the "TEQUILA MARGARITA" style), or a mixed treatment (bold sans + one cursive accent word + bold sans) — choose whichever mood fits this flavor's character, not the same choice as other recent generations, (5) a bilingual description below the flavor name — either a short 2-line form (French then English, e.g. "COLA CLASSIQUE" / "CLASSIC COLA") or a longer descriptive 2-line ingredient-based tagline (e.g. "BOISSON À BASE DE TEQUILA ET LIME" / "TEQUILA AND LIME BEVERAGE", the Margarita style), both real patterns — pick whichever reads cleaner for this flavor, in bold white ALL-CAPS sans-serif, never joined with a "·" dot, (6) volume/ABV specs near the base in this brand's specific real specs typeface — a clean modern geometric/humanist sans-serif (never a generic bold block font): the numeric values ("355", the "%" figure) in a solid clean weight, the unit abbreviations ("mL" or "ml", "alc./vol.") in a visibly LIGHTER/thinner weight and slightly smaller size than the numbers, separated by a vertical bar "|" with generous spacing on both sides — exactly this two-weight numeral/label contrast, not all-one-weight text, (7) directly above the specs line, this brand's real signature tagline "DON'T BOTTLE IT!" set in a HEAVY-weight, TIGHTLY-TRACKED/condensed geometric sans-serif all-caps — letters kerned close together, almost touching, distinctly bolder and more condensed than the volume/ABV specs line right below it and NOT the same loose, lighter default-weight font used for the bilingual description text elsewhere on the can — a punchy, compact, high-impact display line, NOT decorative/scripted/textured, sitting on its own single line with clear spacing above and below it (a thin short accent-color divider rule above it is a real recurring detail), (8) a bottom diagonal chevron band in the accent color continuing the SAME chosen pattern style from the top (styles a/b) or continuing the ingredient silhouettes onto the lower body (style c) — a subtle, tonal-on-tonal (barely-darker-than-the-block) line-art texture of the flavor's real ingredient (e.g. faint lime-slice illustrations within a solid lime-green block) is a valid, recurring way to add pattern to this bottom block even under style (a) or (b), not just style (c). A FLAT, PATTERN-LESS solid-color block is NEVER correct — every real can in this family has some visible pattern, using real flavor-relevant icon shapes or line-art, not just abstract lines alone.
- DO NOT invent an unrelated design system — no embossed line-art motifs, no foil-stamped illustrations, no graphic device that doesn't already exist in the real reference family. The only things that change between flavors in this real system are: the accent color, the flavor name's font treatment, the top/bottom band intensity (minimal stripe vs. busy icon pattern), and the icons/texture used. Everything else in the structure above stays fixed.
- ACCENT COLOR MUST BE REAL, VISIBLE PRODUCT COLOR: this brand has no single fixed "brand color" — verified directly against the real cans, each flavor gets its OWN distinct accent color (green for Vodka Limonade, yellow-green for Margarita, gold for Amaretto Sour, magenta/pink for Sangria Rouge, purple for Gin Limonade Petits Fruits) and there is no universal red requirement. What matters is that THIS flavor's chosen accent color is genuinely painted onto the product (band, text, pattern) — if you cropped the background out, the accent color should still clearly read on the can itself, not just appear as ambient background lighting reflected on an otherwise monochrome product.
- BODY TREATMENT: the default, majority real pattern is a matte-black brushed-metal can body with the accent color confined to the top band, flavor name, and bottom chevron block — use this by default. A small minority of real flavors (a bold rum or sangria-style cocktail) instead use a full-color gradient body (the accent color washing the entire can, wordmark tinted to match) — only pick this minority treatment when the flavor is a similarly bold, fruit-forward cocktail character, not by default.
- IF THIS IS A NON-ALCOHOLIC/MOCKTAIL VARIANT (flavor name mentions "SA", "sans alcool", "mocktail", or similar): use this brand's real, consistent mocktail formula — directly below the top pattern border and above the "CHERRY RIVER" wordmark, add ONE solid metallic silver/gray bar spanning the can's width reading "SANS ALCOOL" in bold black text; this is the ONLY extra element the mocktail variant adds. The word "MOCKTAIL" is NOT a separate bar, banner, or badge of its own — it is simply prepended directly onto the flavor name (e.g. "MOCKTAIL PALOMA"), rendered as ONE single flavor-name text block using the exact same accent-color display treatment from rule (4) above, in the exact same position and size a normal flavor name would occupy. Every other block (wordmark, category tag, bilingual description, specs line, tagline, bottom pattern block) keeps its normal position and treatment exactly as described above — do not stack, reorder, or add any additional band for "MOCKTAIL" beyond it being part of the flavor-name text. CRITICAL: the specs line on a mocktail/SA variant MUST read "0% alc./vol." — never any nonzero percentage, and never any alcohol category word (gin/vodka/rum/tequila/cocktail/wine/vin) anywhere on the can, since this is explicitly non-alcoholic; double-check this before finalizing. This same rule applies to the bilingual description text under rule (5): NEVER use an ingredient-based tagline that names an alcoholic base — including wine (e.g. "BOISSON À BASE DE TEQUILA ET LIME", or a Sangria description naming "VIN ROUGE"/"RED WINE") — on a mocktail/SA can, since that phrasing is exclusive to the alcoholic version of the flavor. For a mocktail/SA variant, either use the short 2-line flavor-only form (French flavor name / English flavor name, e.g. "MARGARITA" / "MARGARITA MOCKTAIL"), or if a longer descriptive tagline is used, describe only the fruit/flavor ingredients and never the alcoholic component (e.g. a Sangria mocktail should say "BOISSON À SAVEUR D'AGRUMES ET DE FRAISES" / "CITRUS AND STRAWBERRY FLAVOURED BEVERAGE" — dropping "de vin rouge"/"red wine" entirely).
- MATERIAL FINISH (this is what makes the real cans look premium, not flat/painted): the can body must show a subtle brushed-aluminum metal texture — fine vertical brushed-metal striations catching the light, giving it a real tactile metallic surface, not a flat matte-painted look. The dark body area should read as dark metal with this brushed texture visible, not solid flat black.
- TYPOGRAPHY POLISH: the flavor name and other display text must look bold, confident, and professionally finished — thick, well-kerned lettering with crisp clean edges and a subtle drop shadow or dark edge for depth/legibility against the metal body, exactly like a real finished can design, not thin or flat-looking text.
- OPTIONAL DETAIL: for a relevant callout (e.g. "NON PÉTILLANTE", "SANS ALCOOL", or similar), a clean thin-bordered pill/badge outline around that text is a real recurring detail in this line — use it where it fits naturally.` : strictStyleNotesOnly ? `
- FOLLOW THE STRUCTURE DESCRIBED IN "Brand visual identity" ABOVE EXACTLY — it is a precise, verified description of this brand's real, already-approved packaging for this specific product category, not a starting point for improvisation. Reproduce every named element (shape, cap, body treatment, label placement, illustration or lack of one, typography pairing, wording) as described.
- DO NOT invent, add, or substitute any graphic device, illustration, foil stamp, badge, seal, or motif beyond what is explicitly named in the brand visual identity description above — if it doesn't mention an illustration, do not add one; if it says a treatment is absent (e.g. "NO illustration wrap"), that absence is a real, deliberate design fact, not a gap to fill in.
- TYPOGRAPHY: keep the attached brand wordmark exactly as reference (never redraw it). For every other text element, match the exact structure and font pairing already described above.` : `
- INVENT ONE DISTINCTIVE GRAPHIC CONCEPT FIRST (do this before anything else): a real packaging designer's job is to create ONE ownable visual device for this flavor — e.g. a custom line-art motif (a river current, a cherry branch silhouette, an abstract splash wave) foil-stamped or spot-gloss-printed behind the wordmark; a debossed/embossed texture; a duotone gradient with fine etched linework; a small circular badge or seal; a subtle pattern print. Pick ONE such device and build the whole label around it, finished with a premium technique (metallic foil, spot-gloss varnish over matte, or emboss) — not just flat white linework.
- DO NOT default to "gradient background + a couple of photorealistic floating fruit + water droplets" as the design itself — that is a generic photography-prop trick, not packaging design, and must not be the ONLY visual idea on the can. Fruit/garnish props (if used at all) belong in the surrounding scene, not as the label's core graphic content.
- TYPOGRAPHY: keep the attached brand wordmark exactly as reference (never redraw it). For every OTHER text element (flavor name, tagline, category tag), use real typographic craft and a deliberate pairing — e.g. a refined lighter-weight sans or a subtle serif for the tagline in contrast to the bold wordmark, considered letter-spacing, not just the same bold uppercase weight repeated for every line.`}
- COLOR STORY: the brand's own core color(s) (see brand visual identity above) MUST be genuinely present as real, visible color IN the label design itself — never only as ambient background lighting or bokeh that fades to black-and-white on the actual product. A design that turns the product itself monochrome while only the surrounding scene carries the brand color is a FAIL — the can/bottle must be recognizably on-brand even with the background cropped out.
- HIERARCHY: a clear top-to-bottom order, all blocks centered on ONE vertical axis with generous even spacing (never cramped or touching). Nothing may overlap or clutter any text block.
- THIS IS A DESIGN CONCEPT MOCKUP FOR REVIEW/APPROVAL, NOT A MARKETING PHOTOGRAPH: show the product ALONE on a clean, plain, neutral studio background (solid white, or a soft seamless light-gray/black studio backdrop — matching exactly how this brand's own real catalog product photos are shot) — NO lifestyle props, NO scene, NO bokeh, NO environmental elements, NO garnish/ice/table surface of any kind. The only subject in frame is the can/bottle itself; the entire point is to evaluate the label/package design clearly, the same way the real reference photos present each product.
- Composition: product centered, straight-on, front-facing, label fully legible edge-to-edge, even soft neutral studio lighting from the front/slightly above (no dramatic rim light, no color-graded shadows, no low hero angle) — a clean catalog/packshot framing, not a cinematic advertising angle.
${designFeedback ? `- FIX THIS FROM THE PREVIOUS ATTEMPT (a design reviewer flagged it): ${designFeedback}\n` : ""}- Describe the label structure and the accent color/font choice for the flavor name in detail; keep the background/lighting description to one plain, brief line (e.g. "on a seamless white studio background, soft even front lighting") — under 140 words total.
- You must ALWAYS return a valid image-generation prompt — never refuse, never ask a clarifying question.
Return ONLY the final image prompt, nothing else. End with "clean studio product photography, plain neutral background, sharp focus throughout, professional catalog-style packaging shot, real brand logo clearly and accurately reproduced, market-ready packaging design mockup."`,
    }],
  });
  const text = message.content[0].text.trim();
  const looksLikeRefusal =
    /^(i cannot|i can't|i need to|i'm unable|sorry,|unfortunately)/i.test(text) ||
    /would you like me to|please (attach|provide|clarify)|let me know if/i.test(text) ||
    text.includes("?");
  if (looksLikeRefusal) {
    throw new Error("Couldn't generate a valid brand concept from this input — try simplifying the flavor name.");
  }
  return text;
}

// Judges the label's graphic design craft specifically (typographic hierarchy, spacing,
// clutter behind text) — separate from hasCorrectLogo, which only checks brand accuracy.
// Feeds actionable feedback back into the next generation attempt, same backstop pattern.
async function critiqueLabelDesign(imageUrl, brandName, styleNotes, matchRealFamily = false, referenceImages = []) {
  const imgRes = await fetch(imageUrl);
  const buf = Buffer.from(await imgRes.arrayBuffer());
  const base64 = buf.toString("base64");
  // Without seeing the brand's actual real cans, this critique has no ground truth and can
  // only guess from generic design-principle assumptions — which previously caused it to
  // wrongly flag genuinely real patterns (e.g. a real triangle+dot border variant) as
  // "foreign to the brand" simply because they weren't spelled out in the text rules below.
  // Attaching the same curated real reference photos used for generation lets it compare
  // directly against real examples instead of guessing.
  const refImages = matchRealFamily
    ? await Promise.all(referenceImages.map(async (url) => {
        const r = await fetch(url);
        const mediaType = r.headers.get("content-type") || (url.endsWith(".png") ? "image/png" : "image/jpeg");
        return { mediaType, base64: Buffer.from(await r.arrayBuffer()).toString("base64") };
      }))
    : [];
  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 200,
    messages: [{
      role: "user",
      content: [
        ...refImages.map((img) => ({ type: "image", source: { type: "base64", media_type: img.mediaType, data: img.base64 } })),
        { type: "image", source: { type: "base64", media_type: "image/png", data: base64 } },
        { type: "text", text: `You are a senior packaging designer for premium beverage brands (the typographic caliber of White Claw, High Noon, Topo Chico). ${refImages.length ? `The FIRST ${refImages.length} images attached are REAL reference photos of ${brandName}'s actual existing product line — this is your ONLY ground truth for what's "on-brand" vs "foreign," not general assumptions about beverage packaging. The LAST image is the NEW candidate design to judge.` : "Judge the attached candidate design."} Judge ONLY the label's graphic design — NOT the photography lighting, props, or background scene.
${styleNotes ? `\n${brandName}'s known brand visual identity: ${styleNotes}\n` : ""}
Check specifically for:
1. Typographic hierarchy, spacing/alignment, whether background imagery clutters or overlaps any text, contrast/legibility.
${matchRealFamily
  ? `2. Real family match: compare the candidate DIRECTLY against the attached real reference photos — does it plausibly belong to the same real product line? The real references themselves already show real variation (different border patterns, different flavor-name fonts, 2-line or 3-line category tags, short or longer bilingual taglines, different accent colors per flavor) — do NOT flag a candidate merely for differing from ONE specific reference photo if its choice is still clearly the same FAMILY of treatment (e.g. a different real geometric border style, or a different real font mood, or a different accent color) as what you see across the attached references. Only flag it if it invents a design system that matches NONE of the attached references' actual patterns (e.g. embossed illustrations, foil line-art, a badge/seal, or an ornate style nothing in the references shows). ALSO check the bottom diagonal color block: is there a visible pattern in it, matching the kind of pattern-density you see in the references? If icons are used: the real references show as many distinct icon types/colors as the flavor names ingredients (e.g. a real 3-fruit flavor shows 3 distinct icon types in 3 different true-to-life colors, not one hue family) — do NOT flag multiple distinct icon types as a failure when the flavor name itself lists multiple ingredients; only flag icons that don't correspond to any named ingredient, or icons scattered with no relation to the flavor at all. On rendering style: this brand's approved concept icons (already accepted and shipped, e.g. the regular Cosmo, Mojito, Tequila Paloma cans) DO show a small highlight dot on rounded fruit like a cranberry, and mild internal shading on a citrus slice — that normal level of flat-vector fruit-icon polish is CORRECT and must NOT be flagged. Only flag an icon if it is clearly full photographic or painterly rendering: strong dimensional 3D lighting with a visible light source and shadow side, camera depth-of-field blur, wet/juicy photographic texture, or airbrushed realism — i.e. something that looks like an actual photo of real fruit, not a slightly-polished flat vector icon. When in doubt, PASS it — this check has been too strict recently and wrongly rejected icons at the same normal polish level as already-approved cans.`
  : `2. Distinctiveness: does the label have ONE real ownable graphic device (a custom line-art motif, texture, badge, pattern) — or is it just a generic gradient background with a couple of literal floating fruit photos and water droplets and nothing else? The latter is a FAIL even if the text is legible and well-spaced — that formula is generic stock photography, not packaging design.`}
3. Typography variety: does the supporting text (flavor name, tagline) show real typographic craft/contrast, or is every line just the same bold uppercase weight repeated?
4. Accent color fidelity: DO NOT flag a design for lacking "cherry-red" or any specific brand-signature color — this brand has NO fixed brand color requirement of any kind, confirmed directly from the real cans (Vodka Limonade is green, Margarita is yellow-green with zero red anywhere, Mocktail Paloma is dusty pink, Sangria Rouge is magenta, Gin Limonade Petits Fruits is purple — none of these show a red crimp, red band, or any red brand signature). Look again carefully at the attached reference images' actual colors before claiming red is expected — it is not. The only real check here is: does THIS flavor's own chosen accent color (whatever hue it picked) actually appear as real color ON the product itself, rather than the product going fully monochrome/grayscale with color only in the background lighting?

Respond in EXACTLY this format:
VERDICT: GOOD or NEEDS_WORK
FEEDBACK: <if NEEDS_WORK: one short, specific, actionable instruction fixing the single biggest issue — naming a concrete graphic device or typographic change to make, not just "improve it". If GOOD, write NONE.>` },
      ],
    }],
  });
  const text = (message.content[0].text || "").trim();
  const verdictMatch = text.match(/VERDICT:\s*(GOOD|NEEDS_WORK)/i);
  const feedbackMatch = text.match(/FEEDBACK:\s*([\s\S]+)/i);
  const good = verdictMatch ? verdictMatch[1].toUpperCase() === "GOOD" : true;
  return { good, feedback: good ? "" : (feedbackMatch ? feedbackMatch[1].trim() : "") };
}

// conceptShapeReferenceImages (Concept Studio's generic per-format map) is Cherry-River-photo-based
// for the bottle formats — wrong silhouette if a brand's real bottle looks different. Override per
// brand+format here only where a REAL, CURRENT photo confirms the correct shape.
const brandShapeReferenceImages = {
  Averse: {
    bottle114: conceptShapeReferenceImages.bottle114, // Averse Vodka Mangue 1.14L — real, current shape, already Averse-specific
    // bottle750 intentionally NOT overridden: the catalog's stored Averse 750mL photo is a
    // known-STALE bottle design (see project memory, Sep 12 2026 finding) — using it would teach
    // the model the wrong, discontinued silhouette. Falls back to the generic bottle750 reference
    // below until Francis supplies an updated Averse 750mL product photo.
  },
};

// Cans (no fixed shape per flavor): several design ideas in one call, via number_of_images.
// Bottles (shape "already programmed" per product line, per Francis): one accurate mockup
// against the chosen bottle type's real shape reference.
// Real Averse bottles carry a small "CHERRY RIVER" parent-brand credit near the base
// (confirmed against 4 real product photos) — feed the actual parent logo file so it
// gets copied faithfully instead of redrawn in a generic font (see hasParentBrandLogo below).
const CHERRY_RIVER_LOGO_URL = "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/brand-assets/cherry-river-logo.png";
// Tight crops (no can shape, no wordmark) isolating just the real flat icon rendering itself —
// fed as extra visual reference because text instructions alone repeatedly failed to stop the
// model from adding glossy/dimensional shading to fruit icons (see generateBrandConcept).
const ICON_STYLE_REFERENCE_LIME = "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/brand-assets/icon-style-ref-lime.png";
const ICON_STYLE_REFERENCE_BERRIES = "https://bypedtyxtnmmdsyrgwpj.supabase.co/storage/v1/object/public/product-images/brand-assets/icon-style-ref-berries.png";

async function generateBrandConcept({ flavorName, productType, brandId, bottleShapeKey, variantCount, userId, userEmail }) {
  const brand = await getBrand(brandId);
  const shapeKey = productType === "can" ? "can" : bottleShapeKey;
  const isUnverifiedAverseBottle750 = brand.name === "Averse" && shapeKey === "bottle750";
  const crBottleCategory = brand.name === "Cherry River" && productType === "bottle" ? getCherryRiverBottleCategory(flavorName) : null;
  const shapeDescriptor = (conceptShapeDescriptors[shapeKey] || conceptShapeDescriptors.can)
    + (isUnverifiedAverseBottle750 ? " — the attached shape reference photo is Averse's 1.14L bottle: match its frosted matte glass material, dome-shaped clear base, and minimalist 'a|v|e|r|s|e' architectural label style exactly, but this concept is a STANDARD 750ml bottle — noticeably shorter and more compact than the 1.14L reference, ordinary spirit-bottle proportions (NOT the tall elongated 1.14L format, and NOT a Cherry River-style arched gin bottle)" : "")
    + (crBottleCategory ? ` — ignore the generic description above; the exact real shape for this category is described in the "Brand visual identity" notes below, and the attached shape reference photo is a real ${crBottleCategory} bottle from this exact category — match its specific silhouette precisely, not a generic spirit-bottle shape.` : "");
  // For the unverified Averse 750mL case, deliberately do NOT fall back to the generic
  // bottle750 reference (a real Cherry River Dry Gin photo) — feeding a different brand's
  // real product photo alongside the Averse logo as two reference images was corrupting the
  // logo (model blending the two brands), causing every attempt to fail hasCorrectLogo. Using
  // Averse's own 1.14L photo instead keeps the brand identity consistent; the height/proportion
  // difference is corrected via the shapeDescriptor text above instead of a second image.
  const shapeReferenceUrl = isUnverifiedAverseBottle750
    ? conceptShapeReferenceImages.bottle114
    : crBottleCategory
    ? cherryRiverBottleShapeReferenceImages[crBottleCategory]
    : (brandShapeReferenceImages[brand.name]?.[shapeKey] || conceptShapeReferenceImages[shapeKey] || conceptShapeReferenceImages.can);
  const n = productType === "can" ? Math.min(variantCount || 4, 4) : 1;

  // For cans, feed every real can this brand already sells as structure references instead of
  // one generic shape photo — the model should adapt this brand's actual approved packaging
  // system to the new flavor, not invent an unrelated design (see getBrandCanReferenceImages).
  const brandCanReferenceImages = productType === "can" ? await getBrandCanReferenceImages(brand.id, brand.name) : [];
  const hasParentBrandLogo = brand.name === "Averse";
  const referenceImages = brandCanReferenceImages.length
    ? [...brandCanReferenceImages, brand.logo_url]
    : [shapeReferenceUrl, brand.logo_url];
  if (hasParentBrandLogo) referenceImages.push(CHERRY_RIVER_LOGO_URL);
  // Text alone repeatedly failed to stop the model from rendering glossy/dimensional fruit icons
  // (5/5 failures across multiple prompt rewrites) — feeding tight crops of the real flat icon
  // rendering itself (no can shape, no photographic context) as extra visual reference finally
  // gives the model something concrete to copy instead of just a text description.
  const includeIconStyleRefs = brand.name === "Cherry River" && productType === "can" && brandCanReferenceImages.length > 0;
  if (includeIconStyleRefs) referenceImages.push(ICON_STYLE_REFERENCE_LIME, ICON_STYLE_REFERENCE_BERRIES);
  // Cherry River's own real style_notes (audited from cans only) don't apply to bottles —
  // e.g. "Don't bottle it!" never appears on a real CR bottle. Use the category-specific
  // bottle description instead when we recognize the spirit category.
  const effectiveStyleNotes = crBottleCategory ? cherryRiverBottleStyleNotesByCategory[crBottleCategory] : brand.style_notes;

  // Verify every variant carries the correct logo AND that the label's graphic design (typography
  // hierarchy, spacing, clutter) actually reads as professional — retry the whole batch on either
  // failure, feeding the design reviewer's specific feedback into the next attempt's prompt, same
  // backstop pattern as hasVisibleText/getFlatSilhouette above.
  // The Vercel function has a 300s hard ceiling — with variantCount > 1, each retry (image gen +
  // vision checks) takes long enough that 5 full attempts can exceed it, which kills the request
  // and returns a non-JSON platform error page instead of our normal error response. Stop retrying
  // with enough buffer left to return a proper JSON error instead of hitting that wall.
  const startedAt = Date.now();
  const TIME_BUDGET_MS = 250_000;
  let replicateUrls, attempts = 0, allCorrect = false, designFeedback = null, enhancedPrompt;
  while (attempts < MAX_CONCEPT_ATTEMPTS && !allCorrect && Date.now() - startedAt < TIME_BUDGET_MS) {
    attempts++;
    enhancedPrompt = await enhanceBrandConceptPrompt({ flavorName, brandName: brand.name, styleNotes: effectiveStyleNotes, shapeDescriptor, designFeedback, realCanReferenceCount: brandCanReferenceImages.length, hasParentBrandLogo, strictStyleNotesOnly: !!crBottleCategory, includeIconStyleRefs });
    replicateUrls = await runImageModel({
      prompt: enhancedPrompt,
      inputImages: referenceImages,
      aspectRatio: "4:5",
      n,
      model: "sunburst",
    });
    const [logoChecks, designCritique] = await Promise.all([
      Promise.all(replicateUrls.map((url) => hasCorrectLogo(url, brand.name))),
      critiqueLabelDesign(replicateUrls[0], brand.name, effectiveStyleNotes, brandCanReferenceImages.length > 0 || !!crBottleCategory, brandCanReferenceImages.length > 0 ? brandCanReferenceImages : (crBottleCategory ? [shapeReferenceUrl] : [])),
    ]);
    const logoOk = logoChecks.every(Boolean);
    allCorrect = logoOk && designCritique.good;
    if (!allCorrect) {
      designFeedback = designCritique.feedback || designFeedback;
      console.warn(`Brand concept attempt ${attempts}/${MAX_CONCEPT_ATTEMPTS} for "${flavorName}" — logoOk=${logoOk} designOk=${designCritique.good}${designCritique.feedback ? ` feedback="${designCritique.feedback}"` : ""}, retrying...`);
    }
  }
  if (!allCorrect) {
    throw new Error(`Generated ${MAX_CONCEPT_ATTEMPTS} attempts and the design still wasn't market-ready (logo accuracy or label design quality) — try again or simplify the flavor name.`);
  }

  const images = await Promise.all(
    replicateUrls.map(async (url, i) => {
      const filename = `brandconcept-${Date.now()}-${i}.png`;
      try { return await saveGeneratedImage(url, filename); }
      catch (e) { return url; }
    })
  );

  const rows = images.map((url) => ({
    id: Date.now().toString() + Math.random().toString(36).slice(2, 6),
    url,
    // Avoid "Brand Concept: Cherry River Cherry River Cola" when the flavor name already
    // includes the brand name (a very likely pattern, per Francis's own example).
    product: `Brand Concept: ${flavorName.toLowerCase().startsWith(brand.name.toLowerCase()) ? flavorName : `${brand.name} ${flavorName}`}`,
    color: "#888888",
    prompt: enhancedPrompt,
    type: "image",
    date: new Date().toISOString(),
    user_id: userId,
    user_email: userEmail,
    cost_usd: Math.round(FLUX_COST_PER_IMAGE * attempts * 1000) / 1000,
  }));
  const { data: savedGenerations, error } = await insertGenerations(rows);
  if (error) throw error;

  return { images, enhancedPrompt, savedGenerations };
}

app.get("/api/brands", async (_req, res) => {
  const { data: brands, error } = await supabase.from("brands").select("id, name, logo_url").order("name");
  if (error) return res.status(500).json({ error: error.message });
  const { data: products } = await supabase.from("products").select("brand_id, product_type");
  const withTypes = brands.map((b) => ({
    ...b,
    // Which product types this brand actually has in the real catalog — drives which
    // options (can / bottle) the UI offers per brand, instead of hardcoding all of them.
    productTypes: [...new Set((products || []).filter((p) => p.brand_id === b.id).map((p) => p.product_type))],
    // Bottle shape references the UI should flag as approximate rather than confirmed-accurate
    // (see brandShapeReferenceImages comment above generateBrandConcept).
    approximateBottleShapes: b.name === "Averse" ? ["bottle750"] : [],
  }));
  res.json(withTypes);
});

app.post("/api/generate-brand-concept", async (req, res) => {
  const reqStart = Date.now();
  console.log(`[brand-concept] request received: flavor="${req.body?.flavorName}" brand=${req.body?.brandId} productType=${req.body?.productType} variantCount=${req.body?.variantCount}`);
  try {
    const user = await getUser(req);
    if (!user) return res.status(401).json({ error: "Unauthorized" });
    const { flavorName, productType, brandId, bottleShapeKey, variantCount } = req.body;
    if (!flavorName || !flavorName.trim()) return res.status(400).json({ success: false, error: "Flavor name is required" });
    if (!brandId) return res.status(400).json({ success: false, error: "Brand is required" });
    if (productType === "bottle" && !bottleShapeKey) return res.status(400).json({ success: false, error: "Bottle type is required" });

    const { images, enhancedPrompt, savedGenerations } = await generateBrandConcept({
      flavorName: flavorName.trim(),
      productType: productType === "bottle" ? "bottle" : "can",
      brandId, bottleShapeKey, variantCount,
      userId: user.id, userEmail: user.email,
    });

    console.log(`[brand-concept] success in ${Date.now() - reqStart}ms`);
    res.json({ success: true, images, enhancedPrompt, generations: savedGenerations });
  } catch (error) {
    console.error(`[brand-concept] error after ${Date.now() - reqStart}ms:`, error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate brand concept" });
  }
});

app.post("/api/generate-concept-scene", async (req, res) => {
  try {
    const user = await getUser(req);
    if (!user) return res.status(401).json({ error: "Unauthorized" });
    const { flavorName, colorHex, format, sceneId, customScene, finish, mood, exampleId, capColor, liquidVisibility, referenceImageUrl, productImageUrl, customPrompt } = req.body;
    if (!flavorName || !flavorName.trim()) return res.status(400).json({ success: false, error: "Flavor name is required" });
    if (!sceneId && !(customScene && customScene.trim())) return res.status(400).json({ success: false, error: "A scene selection or custom scene description is required" });

    const { image, enhancedPrompt, savedGeneration, effectiveColorHex } = await generateConceptMockup({
      flavorName: flavorName.trim(), colorHex, format, sceneId, customScene, finish, mood, exampleId, capColor, liquidVisibility, referenceImageUrl, productImageUrl, customPrompt,
      userId: user.id, userEmail: user.email,
    });

    res.json({ success: true, image, enhancedPrompt, generation: savedGeneration, effectiveColorHex });
  } catch (error) {
    console.error("Concept scene generation error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate concept mockup" });
  }
});

// Generate the same scene+style for one flavor across all 8 scene angles at once (concept board).
// Generates one master reference shot first, then locks every other angle to it via Flux input_images
// so the design (color/finish/proportions) stays identical across all 8 — only the scene changes.
app.post("/api/generate-concept-board", async (req, res) => {
  try {
    const user = await getUser(req);
    if (!user) return res.status(401).json({ error: "Unauthorized" });
    const { flavorName, colorHex, format, finish, mood, exampleId, capColor, liquidVisibility, productImageUrl, customPrompt } = req.body;
    if (!flavorName || !flavorName.trim()) return res.status(400).json({ success: false, error: "Flavor name is required" });
    const sceneList = conceptSceneTemplates[format];
    if (!sceneList) return res.status(400).json({ success: false, error: "Invalid format" });

    const [masterScene, ...restScenes] = sceneList;
    const results = [];
    let referenceImageUrl = null;

    // The master tile's dominant-color extraction (getDominantColor on the real product photo)
    // is the only correct color signal here — the raw `colorHex` from the request is the
    // catalog's stored accent color, which is frequently stale/wrong (confirmed: CR Margarita SA
    // 355mL's stored color is a leftover light-green while the real can is near-black). Every
    // other tile must inherit the MASTER's resolved color, not the raw request value, or its
    // "brand color inspiration" instruction actively contradicts the "match the reference image
    // exactly" instruction — Flux was inconsistently obeying one or the other tile to tile,
    // which is why some board tiles rendered black and others rendered green.
    let boardColorHex = colorHex;

    try {
      const { image, enhancedPrompt, effectiveColorHex } = await generateConceptMockup({
        flavorName: flavorName.trim(), colorHex, format, sceneId: masterScene.id, finish, mood, exampleId, capColor, liquidVisibility, productImageUrl, customPrompt,
        userId: user.id, userEmail: user.email,
      });
      referenceImageUrl = image;
      if (effectiveColorHex) boardColorHex = effectiveColorHex;
      results.push({ sceneId: masterScene.id, sceneName: masterScene.name, success: true, image, enhancedPrompt });
    } catch (e) {
      console.error(`Board master generation failed for scene ${masterScene.id}:`, e.message);
      results.push({ sceneId: masterScene.id, sceneName: masterScene.name, success: false, error: e.message });
    }

    const CONCURRENCY = 3;
    for (let i = 0; i < restScenes.length; i += CONCURRENCY) {
      const batch = restScenes.slice(i, i + CONCURRENCY);
      const batchResults = await Promise.all(
        batch.map(async (scene) => {
          try {
            const { image, enhancedPrompt } = await generateConceptMockup({
              flavorName: flavorName.trim(), colorHex: boardColorHex, format, sceneId: scene.id, finish, mood, exampleId, capColor, liquidVisibility, referenceImageUrl, productImageUrl, customPrompt,
              userId: user.id, userEmail: user.email,
            });
            return { sceneId: scene.id, sceneName: scene.name, success: true, image, enhancedPrompt };
          } catch (e) {
            console.error(`Board generation failed for scene ${scene.id}:`, e.message);
            return { sceneId: scene.id, sceneName: scene.name, success: false, error: e.message };
          }
        })
      );
      results.push(...batchResults);
    }

    res.json({ success: true, results, resolvedColorHex: boardColorHex });
  } catch (error) {
    console.error("Concept board generation error:", error);
    res.status(500).json({ success: false, error: error.message || "Board generation failed" });
  }
});

// Generate the same scene+style across multiple flavors in one session (collection launch)
app.post("/api/generate-concept-batch", async (req, res) => {
  try {
    const user = await getUser(req);
    if (!user) return res.status(401).json({ error: "Unauthorized" });
    const { flavors, format, sceneId, customScene, finish, mood, exampleId, capColor, liquidVisibility, customPrompt } = req.body;
    if (!Array.isArray(flavors) || flavors.length === 0) {
      return res.status(400).json({ success: false, error: "No flavors selected" });
    }
    if (!sceneId && !(customScene && customScene.trim())) {
      return res.status(400).json({ success: false, error: "A scene selection or custom scene description is required" });
    }

    const CONCURRENCY = 3;
    const results = [];
    for (let i = 0; i < flavors.length; i += CONCURRENCY) {
      const batch = flavors.slice(i, i + CONCURRENCY);
      const batchResults = await Promise.all(
        batch.map(async (flavor) => {
          try {
            const { image, enhancedPrompt } = await generateConceptMockup({
              flavorName: flavor.flavorName, colorHex: flavor.colorHex, format, sceneId, customScene, finish, mood, exampleId, capColor, liquidVisibility, productImageUrl: flavor.productImage, customPrompt,
              userId: user.id, userEmail: user.email,
            });
            return { flavorName: flavor.flavorName, success: true, image, enhancedPrompt };
          } catch (e) {
            console.error(`Batch concept generation failed for ${flavor.flavorName}:`, e.message);
            return { flavorName: flavor.flavorName, success: false, error: e.message };
          }
        })
      );
      results.push(...batchResults);
    }

    res.json({ success: true, results });
  } catch (error) {
    console.error("Concept batch generation error:", error);
    res.status(500).json({ success: false, error: error.message || "Batch generation failed" });
  }
});

// Averse's real bottle has THREE zones: opaque white neck/shoulder, clear glass mid-body
// (visible liquid), and an OPAQUE, flavor-colored dome-shaped base/foot (confirmed against real
// product photos, Sep 25 2026 — earlier this was wrongly treated as one continuous clear zone
// all the way to the base, flagged as a real bug by Alyssa/Huzaifa; the base itself must stay
// opaque, not see-through). This is the safety net: same pattern as hasVisibleText/hasCorrectLogo
// elsewhere in this file.
async function hasTransparentGlass(imageUrl, domeColor) {
  const imgRes = await fetch(imageUrl);
  const buf = Buffer.from(await imgRes.arrayBuffer());
  const base64 = buf.toString("base64");
  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 10,
    messages: [{
      role: "user",
      content: [
        { type: "image", source: { type: "base64", media_type: "image/png", data: base64 } },
        { type: "text", text: `This bottle has three zones: (A) the very top neck/shoulder band carrying the "a|v|e|r|s|e" logo, (B) the mid-body starting at the "VODKA SUPÉRIEURE PREMIUM" text and ending where the bottle's flared foot/base begins, (C) the dome-shaped foot/base itself at the very bottom, which should be solid opaque ${domeColor || "white"} and narrower than the bottle's body, with a visible margin of clear glass around its left and right sides. Answer YES only if ALL of these are true: (1) zone B is genuinely clear, transparent, see-through glass — not opaque, solid white, frosted, or matte-colored, and the "VODKA SUPÉRIEURE PREMIUM" text reads as sitting directly on clear glass rather than on a white panel, (2) zone B's color/tone visibly reflects the actual background and lighting of the surrounding scene behind the bottle, rather than looking like a flat white, milky, or hazy cast that stays the same regardless of what's actually behind it, (3) zone C (the base/foot) is OPAQUE and solid-colored — NOT clear or see-through glass, and does NOT show the liquid or background through it, (4) zone C is CLEARLY narrower than the bottle body with clear glass visible on at least one side of it — only fail this specific point if the dome obviously fills the entire width edge-to-edge with zero visible glass margin on either side; if you're unsure or the margin is just small, treat this point as passing. If (1), (2), or (3) fail — zone B looks opaque/frosted/on-a-white-panel, OR zone B looks milky/white/hazy and disconnected from the real scene behind it, OR zone C looks clear/transparent/see-through instead of opaque — answer NO. Answer with exactly one word: YES or NO.` },
      ],
    }],
  });
  return (message.content[0].text || "").trim().toUpperCase().startsWith("YES");
}

// Any branded product with its own real reference photo can have its label text or graphics
// hallucinated by the image model instead of copied faithfully — confirmed real bug (Francis,
// Sep 25/26 2026): "Generate Image" produced a Cherry River Vodka Érable bottle with invented
// garbled French copy ("AU-AARÔME NATUREL D'ÉRABLE" instead of the real "VODKA AROMATISÉE À
// L'ÉRABLE") and a solid-filled maple leaf instead of the real skeletonized/vein-only foil style.
// This is a general risk for every branded product with a reference photo, not just Averse (which
// already has its own hasTransparentGlass safety net) — this check compares the generated label
// directly against the real reference photo, the same verify+retry pattern used elsewhere in this
// file, since prompt wording alone was already proven insufficient for Averse too.
async function hasAccurateLabel(generatedUrl, referenceUrl) {
  const [genRes, refRes] = await Promise.all([fetch(generatedUrl), fetch(referenceUrl)]);
  const [genBuf, refBuf] = await Promise.all([genRes.arrayBuffer(), refRes.arrayBuffer()]);
  const genContentType = genRes.headers.get("content-type") || "image/png";
  const refContentType = refRes.headers.get("content-type") || "image/jpeg";
  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 10,
    messages: [{
      role: "user",
      content: [
        { type: "text", text: "Image A is the REAL reference product photo. Image B is a newly generated marketing photo of the same product, in a different scene/background — ignore all scene/background/lighting differences and focus only on the product's own label area (text, wordmark, and any illustration/graphic device printed on it)." },
        { type: "image", source: { type: "base64", media_type: refContentType, data: Buffer.from(refBuf).toString("base64") } },
        { type: "image", source: { type: "base64", media_type: genContentType, data: Buffer.from(genBuf).toString("base64") } },
        { type: "text", text: `Compare the label in image B against image A. Answer YES only if image B's label reasonably matches image A's real label: the same printed wording (no invented, garbled, mistranslated, or reworded text), and the same illustration/graphic style (e.g. a lace/skeletonized illustration must not become a solid-filled one, a line-art illustration must not become a bold graphic). Minor differences in camera angle, lighting, or crop are fine and should NOT cause a NO. Answer NO only if the label text or graphic style is clearly wrong or invented compared to image A. Answer with exactly one word: YES or NO.` },
      ],
    }],
  });
  return (message.content[0].text || "").trim().toUpperCase().startsWith("YES");
}

// Core image generation pipeline — used by /api/generate-image (existing marketing photography pipeline)
async function generateProductImage({ product, productName, productColor, userPrompt, format, quality, quantity, userId, userEmail }) {
  const aspect_ratio = aspectRatioMap[format] || "1:1";
  const isCan = product && product.product_type === "can";
  const isAverseImg = product?.name?.toLowerCase().includes("averse");
  const averseDomeColorImg = isAverseImg ? getAverseDomeColor(product?.name) : "";

  let enhancedPrompt;
  try {
    enhancedPrompt = await enhancePrompt(userPrompt, product);
  } catch (e) {
    console.error("Prompt enhancement failed:", e.message);
    enhancedPrompt = userPrompt;
  }

  const canPrefix = isCan
    ? "tall slim 355ml cylindrical aluminum beverage can standing perfectly upright at 90 degrees, exact same can design and label as the reference image, label text sharp and undistorted, "
    : isAverseImg
    ? `frosted matte white elongated spirit bottle standing perfectly upright, distinctive colored dome-shaped base cap, minimalist 'a|v|e|r|s|e' architectural label with pipe separators, Cherry River dome cap branding, label text sharp and undistorted, the top neck/shoulder band carrying the 'a|v|e|r|s|e' logo is opaque frosted white — starting right at the 'VODKA SUPÉRIEURE PREMIUM' text the MID-BODY (down to where the bottle's flared foot/base begins) is genuine clear see-through glass: that text sits directly on the clear glass (not on a white backing), so the ribbed glass texture and the liquid inside are visible through it too, never opaque or frosted in the mid-body, this clear mid-body glass must show real optical refraction of the actual scene's background and lighting behind it, NEVER a flat white or milky haze carried over from a reference photo, the liquid itself is ${getAverseLiquidDescriptor(product?.name)} — never a different color, but the dome-shaped foot/base at the very bottom is a SEPARATE, OPAQUE, solid ${averseDomeColorImg}-colored section — NEVER clear or see-through glass, never showing the liquid or background through it — this dome is NARROWER than the bottle's body (about two-thirds of its width), centered at the very bottom with a visible margin of clear see-through glass wrapping around both its left and right sides, never spanning the bottle's full width, carrying the 'CHERRY RIVER' text and the '40% ALC. / VOL. 750 ml' line directly below it, both in solid black (not white or gray), sitting on that opaque ${averseDomeColorImg} surface, `
    : "";
  const prompt = canPrefix + enhancedPrompt;

  const hasReferenceImage = !!(product && product.image);
  const inputImages = hasReferenceImage ? [product.image] : [];
  const count = Math.min(quantity || 1, 4);
  let replicateUrls, glassOk = true, glassAttempts = 0;
  do {
    glassAttempts++;
    replicateUrls = await runImageModel({ prompt, inputImages, aspectRatio: aspect_ratio, n: count, model: "flare" });
    const checks = [];
    if (isAverseImg) {
      checks.push(...(await Promise.all(replicateUrls.map((u) => hasTransparentGlass(u, averseDomeColorImg)))));
    }
    if (hasReferenceImage) {
      checks.push(...(await Promise.all(replicateUrls.map((u) => hasAccurateLabel(u, product.image)))));
    }
    glassOk = checks.every(Boolean);
    if (!glassOk) console.warn(`Label/glass accuracy check failed on attempt ${glassAttempts}/3, retrying...`);
  } while (hasReferenceImage && !glassOk && glassAttempts < 3);

  const images = await Promise.all(
    replicateUrls.map(async (url, i) => {
      const filename = `gen-${Date.now()}-${i}.png`;
      try {
        return await saveGeneratedImage(url, filename);
      } catch (e) {
        console.error("Failed to save generated image to storage:", e.message);
        return url;
      }
    })
  );

  const rows = images.map((url) => ({
    id: Date.now().toString() + Math.random().toString(36).slice(2, 6),
    url,
    product: productName,
    color: productColor || "#888888",
    prompt,
    type: "image",
    date: new Date().toISOString(),
    user_id: userId,
    user_email: userEmail,
    cost_usd: Math.round(FLUX_COST_PER_IMAGE * glassAttempts * 1000) / 1000,
  }));

  const { data: savedGenerations, error } = await insertGenerations(rows);
  if (error) throw error;

  return { images, enhancedPrompt, savedGenerations };
}

// Extract authenticated user ID from Bearer token
async function getUserId(req) {
  const token = extractAccessToken(req.headers)
  if (!token) return null
  const { data: { user } } = await supabase.auth.getUser(token)
  return user?.id || null
}

async function getUser(req) {
  const token = extractAccessToken(req.headers)
  if (!token) return null
  const { data: { user } } = await supabase.auth.getUser(token)
  return user || null
}

async function insertGeneration(row) {
  const { data, error } = await supabase.from("generations").insert(row).select().single()
  if (error && error.message && error.message.includes('user_email')) {
    const { user_email, ...rowWithout } = row
    return supabase.from("generations").insert(rowWithout).select().single()
  }
  return { data, error }
}

async function insertGenerations(rows) {
  const { data, error } = await supabase.from("generations").insert(rows).select()
  if (error && error.message && error.message.includes('user_email')) {
    const sanitized = rows.map(({ user_email, ...r }) => r)
    return supabase.from("generations").insert(sanitized).select()
  }
  return { data, error }
}

// --- Auth: Signup (auto-confirm, no email verification) ---
app.post("/api/signup", async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) return res.status(400).json({ error: "Email and password required." });
  const { data, error } = await supabase.auth.admin.createUser({
    email,
    password,
    email_confirm: true,
  });
  if (error) return res.status(400).json({ error: error.message });
  res.json({ user: data.user });
});

// --- Products ---

app.get("/api/products", async (req, res) => {
  const userId = await getUserId(req)
  if (!userId) return res.status(401).json({ error: "Unauthorized" })
  const { data, error } = await supabase
    .from("products")
    .select("*")
    .order("name", { ascending: true });
  if (error) return res.status(500).json({ error: error.message });
  // Map snake_case → camelCase for frontend
  res.json(data.map((p) => ({
    id: p.id,
    name: p.name,
    category: p.category,
    color: p.color,
    image: p.image,
    productType: p.product_type,
    status: p.status,
  })));
});

// Generic upload used by Concept Studio's "New Flavor Idea" optional reference photo —
// not tied to the products table, just returns a storage URL for use as a Flux shape reference.
app.post("/api/upload-reference", upload.single("image"), async (req, res) => {
  const userId = await getUserId(req);
  if (!userId) return res.status(401).json({ error: "Unauthorized" });
  if (!req.file) return res.status(400).json({ error: "No image file provided" });

  const ext = req.file.originalname.split(".").pop();
  const filename = `concept-references/ref-${Date.now()}.${ext}`;
  try {
    const url = await uploadToStorage(req.file.buffer, req.file.mimetype, filename);
    res.json({ url });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post("/api/products", upload.single("image"), async (req, res) => {
  const userId = await getUserId(req)
  if (!userId) return res.status(401).json({ error: "Unauthorized" })
  const { name, category, color, status, productType } = req.body;
  if (!name || !category || !color) {
    return res.status(400).json({ error: "name, category, and color are required" });
  }

  let imageUrl = null;
  if (req.file) {
    const ext = req.file.originalname.split(".").pop();
    const filename = `product-${Date.now()}.${ext}`;
    try {
      imageUrl = await uploadToStorage(req.file.buffer, req.file.mimetype, filename);
    } catch (e) {
      return res.status(500).json({ error: e.message });
    }
  }

  const id = Date.now().toString();
  const { data, error } = await supabase
    .from("products")
    .insert({ id, name, category, color, image: imageUrl, product_type: productType || "bottle", status: status || "ready", user_id: userId })
    .select()
    .single();

  if (error) return res.status(500).json({ error: error.message });

  res.status(201).json({
    id: data.id, name: data.name, category: data.category,
    color: data.color, image: data.image, productType: data.product_type, status: data.status,
  });
});

app.post("/api/products/:id/image", upload.single("image"), async (req, res) => {
  const userId = await getUserId(req);
  if (!userId) return res.status(401).json({ error: "Unauthorized" });
  if (!req.file) return res.status(400).json({ error: "No image file provided" });

  const { data: product } = await supabase.from("products").select("*").eq("id", req.params.id).single();
  if (!product) return res.status(404).json({ error: "Product not found" });
  if (product.user_id !== userId) return res.status(403).json({ error: "Forbidden" });

  // Delete old image from storage
  if (product.image && product.image.includes(BUCKET)) {
    const oldPath = product.image.split(`${BUCKET}/`)[1];
    await supabase.storage.from(BUCKET).remove([oldPath]);
  }

  const ext = req.file.originalname.split(".").pop();
  const filename = `product-${Date.now()}.${ext}`;
  let imageUrl;
  try {
    imageUrl = await uploadToStorage(req.file.buffer, req.file.mimetype, filename);
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }

  const { data, error } = await supabase
    .from("products")
    .update({ image: imageUrl })
    .eq("id", req.params.id)
    .select()
    .single();

  if (error) return res.status(500).json({ error: error.message });

  res.json({
    id: data.id, name: data.name, category: data.category,
    color: data.color, image: data.image, productType: data.product_type, status: data.status,
  });
});

app.delete("/api/products/:id", async (req, res) => {
  const { data: product } = await supabase.from("products").select("*").eq("id", req.params.id).single();
  if (!product) return res.status(404).json({ error: "Product not found" });

  const userId = await getUserId(req)
  if (!userId || product.user_id !== userId) return res.status(403).json({ error: "Forbidden" })

  // Delete image from storage (only after authorization passes)
  if (product.image && product.image.includes(BUCKET)) {
    const storagePath = product.image.split(`${BUCKET}/`)[1];
    await supabase.storage.from(BUCKET).remove([storagePath]);
  }

  const { error } = await supabase.from("products").delete().eq("id", req.params.id);
  if (error) return res.status(500).json({ error: error.message });

  res.json({ id: product.id, name: product.name });
});

// --- Generations ---

app.get("/api/generations", async (req, res) => {
  const userId = await getUserId(req)
  if (!userId) return res.status(401).json({ error: "Unauthorized" })
  let query = supabase.from("generations").select("*").eq("user_id", userId).order("date", { ascending: false });
  const limit = parseInt(req.query.limit);
  if (limit && limit > 0) query = query.limit(limit);
  const { data, error } = await query;
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

app.get("/api/generations/team", async (req, res) => {
  const userId = await getUserId(req)
  if (!userId) return res.status(401).json({ error: "Unauthorized" })
  let query = supabase.from("generations").select("*").order("date", { ascending: false })
  const limit = parseInt(req.query.limit)
  if (limit && limit > 0) query = query.limit(limit)
  const { data, error } = await query
  if (error) return res.status(500).json({ error: error.message })
  res.json(data)
})

app.post("/api/generations", async (req, res) => {
  const user = await getUser(req)
  if (!user) return res.status(401).json({ error: "Unauthorized" })
  const { url, productName, productColor, sceneName, type, cost_usd } = req.body;
  if (!url || !productName) return res.status(400).json({ error: "url and productName are required" });

  const id = Date.now().toString() + Math.random().toString(36).slice(2, 6);
  const { data, error } = await insertGeneration({ id, url, product: productName, color: productColor || "#888888", scene: sceneName || "", type: type || "image", date: new Date().toISOString(), user_id: user.id, user_email: user.email, cost_usd: cost_usd || 0 });

  if (error) return res.status(500).json({ error: error.message });
  res.status(201).json(data);
});

app.delete("/api/generations/:id", async (req, res) => {
  const userId = await getUserId(req)
  if (!userId) return res.status(401).json({ error: "Unauthorized" })
  const { data: gen } = await supabase.from("generations").select("*").eq("id", req.params.id).single();
  if (!gen) return res.status(404).json({ error: "Generation not found" });
  if (gen.user_id !== userId) return res.status(403).json({ error: "Forbidden" })

  const { error } = await supabase.from("generations").delete().eq("id", req.params.id);
  if (error) return res.status(500).json({ error: error.message });

  res.json(gen);
});

// --- Stats ---

app.get("/api/stats", async (req, res) => {
  const userId = await getUserId(req)
  if (!userId) return res.status(401).json({ error: "Unauthorized" })
  const [{ data: generations }, { data: products }] = await Promise.all([
    supabase.from("generations").select("type, date, cost_usd").eq("user_id", userId),
    supabase.from("products").select("id").eq("user_id", userId),
  ]);

  const now = new Date();
  const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1).toISOString();
  const gens = generations || [];
  const thisMonthGens = gens.filter((g) => new Date(g.date) >= new Date(startOfMonth));

  res.json({
    images: gens.filter((g) => g.type === "image").length,
    videos: gens.filter((g) => g.type === "video").length,
    products: (products || []).length,
    thisMonth: thisMonthGens.length,
    thisMonthSpent: parseFloat(thisMonthGens.reduce((s, g) => s + (parseFloat(g.cost_usd) || 0), 0).toFixed(2)),
    totalSpent: parseFloat(gens.reduce((s, g) => s + (parseFloat(g.cost_usd) || 0), 0).toFixed(2)),
  });
});

// --- Image Generation ---

async function enhanceCollectionPrompt(products, userPrompt) {
  const userContent = []
  for (const product of products) {
    if (product.image) {
      try {
        const imgResp = await fetch(product.image)
        if (imgResp.ok) {
          const imgBuffer = Buffer.from(await imgResp.arrayBuffer())
          const base64 = imgBuffer.toString("base64")
          const contentType = imgResp.headers.get("content-type") || "image/jpeg"
          userContent.push({ type: "image", source: { type: "base64", media_type: contentType, data: base64 } })
          userContent.push({ type: "text", text: `Product: "${product.name}"` })
        }
      } catch (e) {
        console.error("Failed to fetch product image for collection:", e.message)
      }
    }
  }
  const hasCans = products.some(p => p.product_type === "can")
  const hasBottles = products.some(p => p.product_type !== "can")
  const averseProducts = products.filter(p => p.name?.toLowerCase().includes("averse"))
  const productNames = products.map(p => p.name).join(", ")
  userContent.push({
    type: "text",
    text: `You are a world-class product photography director for Cherry River premium beverages.
Study each product image above carefully.

Write a professional photography brief for a COLLECTION SHOT featuring all ${products.length} products together in one styled scene.

Reference style: De Soi editorial — products at different heights on natural props as risers.

${userPrompt ? `Client direction: ${userPrompt}\n\n` : ""}COMPOSITION RULES:
- Each product UPRIGHT at a DIFFERENT HEIGHT: upturned wine glass, textured glass block, stone slab, wooden board as risers
- Every label facing camera, fully readable
${hasCans ? "- Cans: perfectly vertical 90°, label undistorted\n" : ""}${hasBottles ? "- Bottles: upright, label sharp and clear\n" : ""}${averseProducts.length ? `- CRITICAL for ${averseProducts.map(p => `"${p.name}" (liquid: ${getAverseLiquidDescriptor(p.name)}, base dome: ${getAverseDomeColor(p.name)})`).join(", ")}: the top neck/shoulder band carrying the "a|v|e|r|s|e" logo is opaque frosted white — starting right at the "VODKA SUPÉRIEURE PREMIUM" text (which sits directly on clear glass, not a white backing) the MID-BODY only, down to where the bottle's flared foot/base begins, MUST render as genuine clear, transparent, see-through glass revealing the ribbed glass texture and the liquid inside (in the exact color noted above, never a different one), and it must show real refraction of THIS scene's own background/lighting rather than a flat white or milky cast copied from a reference photo — BUT the dome-shaped foot/base itself, at the very bottom, is a SEPARATE, OPAQUE, solid-colored section (the exact base-dome color noted above per product) — NEVER clear or see-through glass, never showing the liquid or background through it — and is NARROWER than the bottle's body (about two-thirds of its width), centered at the very bottom with a visible margin of clear see-through glass wrapping around both its left and right sides, never spanning the bottle's full width — carrying the "CHERRY RIVER" text and the "40% ALC. / VOL. 750 ml" line directly below it, both in solid black, not white or gray, sitting on that opaque base surface, unlike the white wordmark higher up\n` : ""}- Seasonal props scattered naturally: fresh fruits matching each flavor, botanical garnishes, flower petals
- Warm beige/clay background, soft directional natural light from left
- Stone or concrete surface
- Magazine lifestyle editorial quality, one hand reaching in from top (optional)

Products featured: ${productNames}

Under 150 words. Return ONLY the photography brief.`
  })

  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 400,
    messages: [{ role: "user", content: userContent }],
  })
  return message.content[0].text.trim()
}

app.post("/api/generate-collection", async (req, res) => {
  try {
    const user = await getUser(req)
    if (!user) return res.status(401).json({ error: "Unauthorized" })
    const { productIds, format, prompt: userPrompt } = req.body
    if (!productIds || productIds.length < 2) return res.status(400).json({ error: "Select 2–4 products" })

    const { data: products, error: pErr } = await supabase.from("products").select("*").in("id", productIds)
    if (pErr || !products || products.length === 0) return res.status(500).json({ error: "Failed to fetch products" })

    const enhancedPrompt = await enhanceCollectionPrompt(products, userPrompt)
    const aspect_ratio = aspectRatioMap[format] || "1:1"
    const input_images = products.filter(p => p.image).map(p => p.image)
    const averseProductInShot = products.find(p => p.name?.toLowerCase().includes("averse"))
    const hasAverse = !!averseProductInShot
    const averseDomeColorShot = hasAverse ? getAverseDomeColor(averseProductInShot.name) : ""

    // Multiple real products composited into one scene — Sunburst's high-control editing
    // handles multi-reference composition more reliably than the fast/everyday Flare model.
    // Same Averse glass-transparency safety net as generateProductImage — the prompt rule
    // alone isn't a guarantee, so verify + retry when an Averse product is in the shot.
    let replicateUrl, glassOk = true, glassAttempts = 0;
    do {
      glassAttempts++;
      [replicateUrl] = await runImageModel({ prompt: enhancedPrompt, inputImages: input_images, aspectRatio: aspect_ratio, model: "sunburst" })
      if (hasAverse) {
        glassOk = await hasTransparentGlass(replicateUrl, averseDomeColorShot);
        if (!glassOk) console.warn(`Collection Shot Averse glass-transparency check failed on attempt ${glassAttempts}/3, retrying...`);
      }
    } while (hasAverse && !glassOk && glassAttempts < 3);

    const filename = `collection-${Date.now()}.png`
    let finalUrl
    try { finalUrl = await saveGeneratedImage(replicateUrl, filename) }
    catch (e) { finalUrl = replicateUrl }

    const productNamesStr = products.map(p => p.name).join(", ")
    await insertGeneration({
      id: Date.now().toString() + Math.random().toString(36).slice(2, 6),
      url: finalUrl,
      product: `Collection: ${productNamesStr.slice(0, 80)}`,
      color: products[0]?.color || "#888888",
      prompt: enhancedPrompt,
      type: "image",
      date: new Date().toISOString(),
      user_id: user.id,
      user_email: user.email,
      cost_usd: Math.round(FLUX_COST_PER_IMAGE * glassAttempts * 1000) / 1000,
    })

    res.json({ success: true, image: finalUrl, enhancedPrompt })
  } catch (error) {
    console.error("Collection generation error:", error)
    res.status(500).json({ success: false, error: error.message || "Failed to generate collection" })
  }
})

app.post("/api/generate-free-scene", async (req, res) => {
  try {
    const user = await getUser(req);
    if (!user) return res.status(401).json({ error: "Unauthorized" });
    const { prompt: userPrompt, format } = req.body;
    if (!userPrompt?.trim()) return res.status(400).json({ error: "Prompt is required" });

    let enhancedPrompt;
    try {
      const msg = await anthropic.messages.create({
        model: "claude-opus-4-6",
        max_tokens: 300,
        messages: [{ role: "user", content: [{ type: "text", text: `You are a world-class advertising photography director for premium beverage brands.
Enhance this prompt into a professional photography brief. Keep it under 120 words.
Return ONLY the enhanced prompt — no explanations, no labels.

User prompt: ${userPrompt}

Rules:
- Keep the user's core scene and concept
- Add: exact lighting details, camera settings (f-stop, depth of field), mood, color palette, photographic style
- Make it specific and vivid, ready for a professional AI image generator
- End with: "professional beverage advertising photography"` }] }],
      });
      enhancedPrompt = msg.content[0].text.trim();
    } catch (e) {
      console.error("Free scene prompt enhancement failed:", e.message);
      enhancedPrompt = userPrompt;
    }

    const aspect_ratio = aspectRatioMap[format] || "1:1";
    const [replicateUrl] = await runImageModel({ prompt: enhancedPrompt, aspectRatio: aspect_ratio, model: "flare" });
    const filename = `freescene-${Date.now()}.png`;
    let finalUrl;
    try { finalUrl = await saveGeneratedImage(replicateUrl, filename); }
    catch (e) { finalUrl = replicateUrl; }

    await insertGeneration({
      id: Date.now().toString() + Math.random().toString(36).slice(2, 6),
      url: finalUrl,
      product: "Free Scene",
      color: "#888888",
      prompt: enhancedPrompt,
      type: "image",
      date: new Date().toISOString(),
      user_id: user.id,
      user_email: user.email,
      cost_usd: FLUX_COST_PER_IMAGE,
    });

    res.json({ success: true, image: finalUrl, enhancedPrompt });
  } catch (error) {
    console.error("Free scene generation error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate image" });
  }
});

app.post("/api/generate-image", async (req, res) => {
  try {
    const user = await getUser(req)
    if (!user) return res.status(401).json({ error: "Unauthorized" })
    const { productName, productColor, productId, prompt: userPrompt, format, quality, quantity } = req.body;

    const { data: product } = await supabase
      .from("products")
      .select("*")
      .eq("id", productId)
      .maybeSingle();

    const { images, enhancedPrompt, savedGenerations } = await generateProductImage({
      product,
      productName,
      productColor,
      userPrompt: userPrompt || `A premium product photo of "${productName}"`,
      format,
      quality,
      quantity,
      userId: user.id,
      userEmail: user.email,
    });

    res.json({ success: true, images, generations: savedGenerations, enhancedPrompt });
  } catch (error) {
    console.error("Generation error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate image" });
  }
});


// --- Video Generation ---

// In-memory map tracking background watermark removal jobs
// { taskId -> { status: 'processing'|'done'|'failed', videoUrl?, error? } }
const wmTasks = new Map();

// Cache of submitted PiAPI payloads keyed by taskId — used for OutputAudioRisk auto-retry
const taskCache = new Map();

// Top-level Supabase video upload (used by both WM removal and fallback paths)
async function uploadVideoToSupabase(buffer, taskId) {
  const filename = `${taskId}.mp4`;
  const { data: uploadData, error: uploadError } = await supabase.storage
    .from(VIDEO_BUCKET)
    .upload(filename, buffer, { contentType: "video/mp4", upsert: true });
  if (uploadError) throw new Error(`Supabase upload: ${uploadError.message}`);
  const { data: { publicUrl } } = supabase.storage.from(VIDEO_BUCKET).getPublicUrl(uploadData.path);
  return publicUrl;
}

// Starts watermark removal in the background — returns immediately.
// Stores result in wmTasks so polling can pick it up.
function startWatermarkRemovalBackground(taskId, rawVideoUrl) {
  if (wmTasks.has(taskId)) return; // already started
  wmTasks.set(taskId, { status: "processing" });

  (async () => {
    try {
      console.log(`[WM] Starting background removal for ${taskId}`);
      const cleanUrl = await removeWatermarkWithFfmpeg(rawVideoUrl, taskId);
      wmTasks.set(taskId, { status: "done", videoUrl: cleanUrl });
      console.log(`[WM] Done: ${cleanUrl}`);
    } catch (err) {
      console.error(`[WM] Removal failed for ${taskId}:`, err.message);
      // Fallback: download original and upload to Supabase
      try {
        const resp = await fetch(rawVideoUrl);
        if (resp.ok) {
          const buf = Buffer.from(await resp.arrayBuffer());
          const fallbackUrl = await uploadVideoToSupabase(buf, taskId);
          wmTasks.set(taskId, { status: "done", videoUrl: fallbackUrl });
          console.log(`[WM] Fallback saved: ${fallbackUrl}`);
        } else {
          wmTasks.set(taskId, { status: "done", videoUrl: rawVideoUrl });
        }
      } catch {
        wmTasks.set(taskId, { status: "done", videoUrl: rawVideoUrl });
      }
    }
  })();
}

// Remove Seedance "AI" watermark using ffmpeg delogo filter.
// Watermark is always in the TOP-LEFT corner of Seedance videos.
async function removeWatermarkWithFfmpeg(videoUrl, taskId) {
  const tmpIn  = join(tmpdir(), `cr_wm_in_${taskId}.mp4`);
  const tmpOut = join(tmpdir(), `cr_wm_out_${taskId}.mp4`);

  // Download video to temp file
  const resp = await fetch(videoUrl);
  if (!resp.ok) throw new Error(`WM download failed: ${resp.status}`);
  await writeFile(tmpIn, Buffer.from(await resp.arrayBuffer()));

  // Probe dimensions via ffmpeg stderr (works on Vercel — no ffprobe needed)
  const probeErr = await execFileAsync(FFMPEG_PATH, ["-i", tmpIn], { timeout: 30_000 }).catch(e => e);
  const dimMatch = (probeErr.stderr || probeErr.message || "").match(/(\d{3,4})x(\d{3,4})/);
  const vw = dimMatch ? parseInt(dimMatch[1]) : 834;
  const vh = dimMatch ? parseInt(dimMatch[2]) : 1112;

  // Seedance watermark: TOP-LEFT corner ~15% width x 10% height
  // x=2,y=2 because delogo cannot touch the frame edge (0)
  const dw = Math.floor(vw * 0.15) - 2;
  const dh = Math.floor(vh * 0.10) - 2;
  console.log(`[WM] delogo top-left: w=${dw} h=${dh} (video ${vw}x${vh})`);

  await execFileAsync(FFMPEG_PATH, [
    "-y", "-i", tmpIn,
    "-vf", `delogo=x=2:y=2:w=${dw}:h=${dh}:show=0`,
    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
    "-pix_fmt", "yuv420p", "-c:a", "copy",
    "-movflags", "+faststart",
    tmpOut,
  ], { timeout: 300_000 });

  // Upload clean video to Supabase
  const cleanBuffer = await readFile(tmpOut);
  const filename = `${taskId}.mp4`;
  const { data: uploadData, error: uploadError } = await supabase.storage
    .from(VIDEO_BUCKET)
    .upload(filename, cleanBuffer, { contentType: "video/mp4", upsert: true });
  if (uploadError) throw new Error(`Supabase upload: ${uploadError.message}`);
  const { data: { publicUrl } } = supabase.storage.from(VIDEO_BUCKET).getPublicUrl(uploadData.path);

  // Cleanup temp files (non-blocking)
  Promise.all([unlink(tmpIn).catch(() => {}), unlink(tmpOut).catch(() => {})]);

  return publicUrl;
}

// ── CAN FLAVOR PRESETS — Francis Visual DNA Brief (Apr 2026) ──

function isAlcoholicCan(product) {
  const cat = (product.category || "").toLowerCase();
  const name = (product.name || "").toLowerCase();
  if (cat.includes("sans alcool") || cat.includes("non-alc") || cat.includes("non-alcoholic")) return false;
  if (name.includes("sans alcool") || name.includes("non-alc")) return false;
  return true;
}

const canFlavorPresets = [
  {
    id: "vodka_limonade",
    detect: n => (n.includes("limonade") || n.includes("lemonade")) && (n.includes("vodka") || n.includes("non-petillante")),
    heroScene: "Sunny wooden dock at the edge of a calm lake at noon in summer. Perspiring can resting on a weathered dock railing, halved lemon slices falling in slow motion toward the still water below, golden sunlight refracting through the liquid",
    colorPalette: "vivid lemon yellow + brilliant white + sky blue + warm natural wood",
    props: "halved lemon slices, condensation droplets, weathered wood dock planks, calm mirror-like lake water, golden noon sunlight",
    motionStyle: "slow-motion lemon drop toward still lake water, water ripple spreading outward, golden light refracting — STILL liquid, absolutely NO carbonation bubbles, NO fizz",
    keyNote: "CRITICAL: Still liquid — NO bubbles, NO carbonation fizz. Smooth calm water and liquid only.",
    isNonAlcoholic: false,
  },
  {
    id: "orange_sanguine",
    detect: n => (n.includes("orange") && n.includes("sanguine")) || (n.includes("orange") && n.includes("framboise") && n.includes("355")),
    heroScene: "Bright midday poolside terrace in direct summer sun. Can placed on a colorful beach towel near a pool edge, blood orange wedges and fresh raspberries exploding in a vivid juice burst frozen in the air around the can",
    colorPalette: "deep blood orange + vivid coral orange + raspberry pink + brilliant white + direct midday sunshine",
    props: "blood orange wedges, whole fresh raspberries, frozen water droplets in air, pool edge or bright beach towel, sharp direct midday sunlight",
    motionStyle: "high-speed freeze of fruit burst explosion, water droplets suspended mid-air, direct sunlight creating brilliant specular highlights on fruit and can",
    keyNote: "NO powdered sugar, NO syrup, NO dessert props. Pure fresh fruit, pure water, clean energy.",
    isNonAlcoholic: false,
  },
  {
    id: "petits_fruits_basilic",
    detect: n => n.includes("basilic") || n.includes("petits fruits") || n.includes("petits-fruits"),
    heroScene: "Chic urban rooftop terrace on a warm summer evening. White marble bistro table, a single lit candle, fresh bright green basil leaves and red berries scattered casually around the can, soft warm ambient terrace evening lighting",
    colorPalette: "deep purple mauve + fresh basil green + marble white + raspberry pink + warm candlelight amber",
    props: "fresh basil sprigs, mixed red berries (raspberries, blueberries, strawberries), white marble bistro table, lit candle, soft warm ambient evening light",
    motionStyle: "herb micro-flutter in warm summer air, condensation drop sliding down can surface, candlelight flicker reflecting on aluminum",
    isNonAlcoholic: false,
  },
  {
    id: "pamplemousse_rose",
    detect: n => n.includes("pamplemousse") || (n.includes("grapefruit") && (n.includes("rose") || n.includes("pink"))),
    heroScene: "Outdoor summer party at a lake house patio or beach club in golden hour light. Can surrounded by vivid halved pink grapefruit slices, condensation streaming down cold aluminum in summer heat",
    colorPalette: "vivid hot pink + coral pink + white + warm golden summer light",
    props: "halved pink grapefruit slices, grapefruit wedges, condensation droplets streaming down can, warm golden outdoor summer light",
    motionStyle: "condensation streaming down cold aluminum, pink grapefruit halves falling through the air, warm golden light refracting off can surface",
    isNonAlcoholic: false,
  },
  {
    id: "sangria",
    detect: n => n.includes("sangria"),
    heroScene: "Late afternoon Mediterranean terrace, low rattan coffee table, warm golden hour raking light, relaxed aperitivo atmosphere",
    colorPalette: "deep Bordeaux red + warm orange + golden amber + warm Mediterranean afternoon light",
    props: "orange slices, fresh red berries, rattan coffee table surface, golden hour raking light, Mediterranean terrace setting",
    motionStyle: "gentle condensation drip down can side, orange slice rolling to rest beside the can, warm afternoon light sweeping across aluminum surface",
    isNonAlcoholic: true,
  },
  {
    id: "canneberge",
    detect: n => n.includes("canneberge") || n.includes("cranberry"),
    heroScene: "Active outdoor lakeside dock in bright morning light, can resting on a weathered wooden dock plank or paddle, crystal clear lake water glittering in the background",
    colorPalette: "vivid cranberry red + brilliant white + sky blue + natural dock wood tones",
    props: "fresh whole cranberries, wooden dock plank surface, crystal clear lake water in background, bright morning sunlight",
    motionStyle: "cranberries rolling and bouncing around the can, condensation on cold aluminum, bright morning light speckling the can surface",
    isNonAlcoholic: true,
  },
  {
    id: "melon_lime",
    detect: n => n.includes("melon") && (n.includes("lime") || n.includes("citron")),
    heroScene: "Sunny morning on a wooden lake dock, fresh melon wedges and lime slices on a light linen beach blanket, sparkling lake water in background, natural morning light through leafy tree canopy",
    colorPalette: "fresh melon green + lime yellow + brilliant white + sky blue + natural wood and linen",
    props: "honeydew melon wedges, fresh lime slices, light linen beach blanket on dock wood, clear sparkling lake water, morning sunlight filtering through tree canopy",
    motionStyle: "melon wedge dropping into clear lake water creating a clean splash, lime wedge beside it, gentle water ripple spreading, morning dew visible on aluminum surface",
    keyNote: "Fresh, clean, health-forward — nature-centric not party-centric.",
    isNonAlcoholic: true,
  },
  {
    id: "mojito_nonalc",
    detect: n => n.includes("mojito"),
    heroScene: "High-end outdoor bar counter at night, dark polished wood surface, can with a fresh mint sprig and raw sugarcane stick beside it, artisanal clear ice cube nearby, warm overhead spot light",
    colorPalette: "vivid lime green + fresh mint white + dark polished bar wood + warm amber spot lighting",
    props: "fresh mint sprigs, raw sugarcane stick, artisanal clear ice cube, dark polished wood bar counter, cocktail shaker in soft background bokeh, overhead spot lighting",
    motionStyle: "fresh mint sprig placement beside can, single condensation trail sliding down can, warm spot light reflection drifting across aluminum surface",
    isNonAlcoholic: true,
  },
  {
    id: "margarita_nonalc",
    detect: n => n.includes("margarita"),
    heroScene: "Upscale outdoor bar or rooftop terrace, natural stone surface, can with a lime wedge and salt-rimmed glass beside it, artisanal ice, warm cocktail atmosphere",
    colorPalette: "bright lime yellow + sea salt white + lime green + natural stone warm tones + cocktail bar amber light",
    props: "lime rounds, sea salt on nearby glass rim, artisanal clear ice cubes, natural stone counter surface, warm amber cocktail bar lighting",
    motionStyle: "lime wedge placement beside can, condensation on cold aluminum surface, sea salt crystals catching the warm light",
    isNonAlcoholic: true,
  },
  {
    id: "cosmo_nonalc",
    detect: n => n.includes("cosmo"),
    heroScene: "Upscale rooftop bar at dusk, dark sophisticated bar surface, can with orange zest garnish and fresh cranberries artfully placed beside it, soft city light bokeh in background",
    colorPalette: "cranberry red + orange amber + deep dark background + soft city lights bokeh",
    props: "fresh cranberries, orange zest curl, dark sophisticated bar surface, soft city bokeh lights, warm amber spot lighting on can",
    motionStyle: "orange zest curl unfurling slowly, fresh cranberries settling beside can, city light bokeh softly shifting in background",
    isNonAlcoholic: true,
  },
  {
    id: "amaretto_nonalc",
    detect: n => n.includes("amaretto"),
    heroScene: "Mediterranean terrace late afternoon, low rattan coffee table, warm golden hour raking light, elegant aperitivo trio atmosphere",
    colorPalette: "warm amber + burnt orange + golden afternoon glow + Mediterranean terrace warm earthy tones",
    props: "orange slices, whole olives, rattan coffee table surface, golden late afternoon sunlight, Mediterranean terrace aperitivo atmosphere",
    motionStyle: "gentle condensation drip down can, orange slice settling beside the can, golden afternoon light sweeping across the aluminum surface",
    isNonAlcoholic: true,
  },
  {
    id: "paloma_nonalc",
    detect: n => n.includes("paloma"),
    heroScene: "Mediterranean terrace late afternoon, warm golden light, pink grapefruit and fresh lime garnishes beside the can, aperitivo atmosphere",
    colorPalette: "grapefruit pink + lime green + warm white + warm golden Mediterranean afternoon light",
    props: "halved pink grapefruit wedge, fresh lime slice, Mediterranean terrace stone surface, warm golden late afternoon sun",
    motionStyle: "grapefruit wedge placement beside can, condensation beads forming on cold aluminum, warm golden light reflection drifting across can surface",
    isNonAlcoholic: true,
  },
];

function getCanPreset(product) {
  const name = (product.name || "").toLowerCase();
  for (const preset of canFlavorPresets) {
    if (preset.detect(name)) return preset;
  }
  return isAlcoholicCan(product) ? {
    heroScene: "Outdoor summer scene — golden hour at a lake house patio or rooftop terrace, vibrant social energy, warm summer light",
    colorPalette: "vivid punchy summer colors — warm pinks, oranges, electric blues — flavor-dependent",
    props: "fresh citrus slices, condensation-covered can, crushed ice, tropical fruits, warm golden summer light",
    motionStyle: "condensation streaming down cold can, fruit slices falling through summer air, golden light catching and refracting off aluminum surface",
    isNonAlcoholic: false,
  } : {
    heroScene: "Active outdoor nature scene — morning dock or mountain trail, clear lake water or open sky, fresh clean energy",
    colorPalette: "crisp natural tones — greens, sky blues, brilliant whites, fresh citrus — health-forward and bright",
    props: "fresh-cut fruit, clean water droplets, natural surface (wood, stone, linen), morning natural light, nature elements in background",
    motionStyle: "clean fresh water splash, fruit bursting with vitality, light water ripple, morning dew visible on can surface",
    isNonAlcoholic: true,
  };
}

// Enhance a user's video scene description into a Seedance 2 Pro optimized Cherry River prompt
async function enhanceVideoPrompt(userScene, product) {
  const flavorMap = {
    "framboise": "large crystal ice cubes + short crystal rock glass + whole raspberries + halved lime wedges + deep crimson raspberry liquid",
    "raspberry": "large crystal ice cubes + short crystal rock glass + whole raspberries + halved lime wedges + deep crimson raspberry liquid",
    "pamplemousse": "large crystal ice cubes + short crystal rock glass + halved pink grapefruit slices + grapefruit wedges + vibrant hot-pink liquid",
    "grapefruit": "large crystal ice cubes + short crystal rock glass + halved pink grapefruit slices + grapefruit wedges + vibrant hot-pink liquid",
    "triple sec": "whole orange slices, orange zest curls, golden amber liquid",
    "citrus": "citrus slices, orange zest, bright golden liquid",
    "litchi": "whole lychee fruit, tangerine wedges, pale rose liquid",
    "tangerine": "whole lychee fruit, tangerine wedges, pale rose liquid",
    "vanille": "split vanilla pods, cream drizzle, warm ivory liquid",
    "vanilla": "split vanilla pods, cream drizzle, warm ivory liquid",
    "rhum": "raw sugar cane stick, lime wedge, deep amber liquid",
    "rum": "raw sugar cane stick, lime wedge, deep amber liquid",
    "érable": "maple leaf, amber syrup drip, golden liquid",
    "erable": "maple leaf, amber syrup drip, golden liquid",
    "café": "whole coffee beans, espresso drip, dark caramel liquid",
    "cafe": "whole coffee beans, espresso drip, dark caramel liquid",
    "melon": "honeydew melon cube, cucumber slice, pale jade liquid",
    "gin": "botanicals, juniper berries, ice crystals",
    "pomme": "whole red and green apples, apple slices, cinnamon stick, crystal rocks glass",
    "apple": "whole red and green apples, apple slices, cinnamon stick, crystal rocks glass",
    "mangue": "fresh mango halves, mango cubes, tropical yellow-orange liquid, mint sprig",
    "mango": "fresh mango halves, mango cubes, tropical yellow-orange liquid, mint sprig",
    "supérieure": "large crystal ice cubes, crystal rocks glass, clean water droplets",
    "superieure": "large crystal ice cubes, crystal rocks glass, clean water droplets",
    "london dry": "juniper berries, rosemary sprigs, lemon peel, botanical elements",
  };

  // Color scheme per product — based on actual Cherry River reference videos
  // Framboise reference: warm crimson foreground + TEAL backlight contrast (not monochromatic)
  // Pamplemousse reference: hot pink everything + lifestyle + aerial fruit shot
  const colorWorldMap = {
    "framboise": "warm crimson-raspberry foreground + deep teal-turquoise backlight creating cinematic color contrast — dark charcoal stone-textured surface — bottle glows ruby red through warm backlight",
    "raspberry": "warm crimson-raspberry foreground + deep teal-turquoise backlight creating cinematic color contrast — dark charcoal stone-textured surface — bottle glows ruby red through warm backlight",
    "pamplemousse": "saturated hot magenta-pink world — hot pink background, glossy hot-pink surface, pink atmospheric haze — bottle glows translucent pink — everything saturated neon pink",
    "grapefruit": "saturated hot magenta-pink world — hot pink background, glossy hot-pink surface, pink atmospheric haze — bottle glows translucent pink — everything saturated neon pink",
    "triple sec": "pure obsidian-black world — infinite black void, matte black surface, zero ambient fill, single razor-thin rim light on bottle edge",
    "litchi": "warm blush-rose world — soft pink background, warm ivory surface, bottle glows pale rose from backlight",
    "tangerine": "warm peach-coral world — warm orange-pink background, ivory surface, bottle glows pale gold-pink",
    "vanille": "warm ivory-cream world — cream background, polished marble surface, warm golden low-angle light",
    "vanilla": "warm ivory-cream world — cream background, polished marble surface, warm golden low-angle light",
    "rhum": "warm amber-mahogany world — rich brown background, dark wood surface, deep amber liquid light",
    "rum": "warm amber-mahogany world — rich brown background, dark wood surface, deep amber liquid light",
    "érable": "rich amber-gold world — warm autumn orange background, maple wood surface, golden harvest light",
    "erable": "rich amber-gold world — warm autumn orange background, maple wood surface, golden harvest light",
    "café": "deep espresso-brown world — matte charcoal background, dark surface, warm amber spotlight from above",
    "cafe": "deep espresso-brown world — matte charcoal background, dark surface, warm amber spotlight from above",
    "melon": "vibrant jade-green world — fresh green background, dewy surface, crisp morning light",
    "pomme": "warm natural terracotta-beige world — sandy stone surface, warm daylight, green apple and red apple tones, organic earthy warmth",
    "apple": "warm natural terracotta-beige world — sandy stone surface, warm daylight, green apple and red apple tones, organic earthy warmth",
    "mangue": "warm golden-terracotta world — earthy terracotta backdrop, golden afternoon sun, amber and orange tones, tropical warmth",
    "mango": "warm golden-terracotta world — earthy terracotta backdrop, golden afternoon sun, amber and orange tones, tropical warmth",
    "supérieure": "deep midnight-charcoal world — dark charcoal background, gold glitter particles, silver rim light on bottle, cool premium atmosphere",
    "superieure": "deep midnight-charcoal world — dark charcoal background, gold glitter particles, silver rim light on bottle, cool premium atmosphere",
    "london dry": "warm amber-terracotta world — earthy terracotta textured wall backdrop, travertine stone surface, botanical warmth, golden evening light",
  };

  // Music style per product — strictly safe words only
  // BANNED: moody, minor key, dark, dramatic, tension, noir, sinister, brooding, ominous, haunting, aggressive, deep sub-bass
  const musicMap = {
    "pamplemousse": "upbeat tropical electronic pop, feminine synth lead, bubbly arpeggio, bright percussive hits, 108 BPM, warm reverb tail, cheerful and fresh",
    "grapefruit": "upbeat tropical electronic pop, feminine synth lead, bubbly arpeggio, bright percussive hits, 108 BPM, warm reverb tail, cheerful and fresh",
    "framboise": "smooth flowing electronic, warm synth bass groove, gentle melodic pad, rhythmic pulse, 90 BPM, rich and inviting",
    "raspberry": "smooth flowing electronic, warm synth bass groove, gentle melodic pad, rhythmic pulse, 90 BPM, rich and inviting",
    "litchi": "warm sunset electronic, smooth synth chords, gentle arpeggiated melody, soft percussion, 100 BPM, dreamy and serene",
    "tangerine": "warm sunset electronic, smooth synth chords, gentle arpeggiated melody, soft percussion, 100 BPM, dreamy and serene",
    "triple sec": "minimal electronic, slow bass pulse, single piano note, quiet and precise, 72 BPM, cool and refined",
    "rhum": "jazz-infused electronic, warm upright bass line, muted trumpet note, vinyl texture, 82 BPM, sophisticated lounge atmosphere",
    "rum": "jazz-infused electronic, warm upright bass line, muted trumpet note, vinyl texture, 82 BPM, sophisticated lounge atmosphere",
    "vanille": "slow elegant ambient, orchestral bass note, distant piano melody, soft string swell, 65 BPM, elegant and luxurious",
    "vanilla": "slow elegant ambient, orchestral bass note, distant piano melody, soft string swell, 65 BPM, elegant and luxurious",
    "café": "jazzy electronic, finger-snapped rhythm, double bass walk, mellow synth pad, 88 BPM, smooth Parisian café atmosphere",
    "cafe": "jazzy electronic, finger-snapped rhythm, double bass walk, mellow synth pad, 88 BPM, smooth Parisian café atmosphere",
    "whisky": "rich warm electronic, slow smooth groove, cello undertone, ambient pad shimmer, 78 BPM, cozy and warm",
    "érable": "rich warm electronic, slow smooth groove, cello undertone, ambient pad shimmer, 78 BPM, cozy and warm",
    "erable": "rich warm electronic, slow smooth groove, cello undertone, ambient pad shimmer, 78 BPM, cozy and warm",
    "mojito": "fresh energetic electronic pop, bright synth pluck, tropical steel pan sample, four-on-the-floor kick, 122 BPM, summer energy",
    "melon": "playful summer electronic, bouncy synth bass, light marimba melody, crisp snare, 118 BPM, refreshing and clean",
    "orange": "vibrant electronic pop, punchy bass hit, bright chord stab, energetic build, 115 BPM, bold and confident",
    "tequila": "minimal electronic, slow reverb guitar pluck, low bass note, gentle pad, 80 BPM, golden hour mood",
    "limonade": "upbeat summer electronic pop, bright synth pluck, crisp snare, 120 BPM, clean and refreshing",
    "sangria": "warm Mediterranean electronic, smooth acoustic strum, gentle bass groove, 88 BPM, relaxed and convivial",
    "canneberge": "energetic outdoor electronic pop, bright synth burst, four-on-the-floor kick, 118 BPM, active and fresh",
    "amaretto": "warm aperitivo electronic, smooth piano melody, gentle bass, 80 BPM, relaxed Mediterranean afternoon",
    "paloma": "breezy tropical electronic, bright citrus synth pluck, light percussion, 112 BPM, fresh and carefree",
    "cosmo": "sophisticated evening electronic, smooth bass groove, warm piano chord, 85 BPM, premium and refined",
    "margarita": "bright energetic electronic pop, clean synth hit, crisp percussion, 116 BPM, fresh and lively",
    "basilic": "warm summer evening electronic, soft synth chord, gentle bass pulse, 92 BPM, sophisticated and social",
    "pomme": "crisp refreshing electronic, bright synth pluck, clean crisp percussion, 110 BPM, fresh orchard energy",
    "apple": "crisp refreshing electronic, bright synth pluck, clean crisp percussion, 110 BPM, fresh orchard energy",
    "mangue": "warm tropical electronic, smooth bass groove, bright synth pad, 108 BPM, golden afternoon energy",
    "mango": "warm tropical electronic, smooth bass groove, bright synth pad, 108 BPM, golden afternoon energy",
    "supérieure": "minimal sleek electronic, slow precise bass note, single piano key, quiet and refined, 72 BPM, VIP lounge atmosphere",
    "superieure": "minimal sleek electronic, slow precise bass note, single piano key, quiet and refined, 72 BPM, VIP lounge atmosphere",
    "london dry": "warm botanical electronic, soft acoustic guitar pluck, ambient pad, 80 BPM, sophisticated evening atmosphere",
  };

  const productNameLower = product.name.toLowerCase();
  let flavorProps = "fresh fruit slices, botanical garnishes, premium ice";
  for (const [key, val] of Object.entries(flavorMap)) {
    if (productNameLower.includes(key)) { flavorProps = val; break; }
  }

  let musicStyle = "cinematic luxury ambient, smooth synth bass pulse, subtle atmospheric pad, no percussion, 80 BPM, VIP lounge atmosphere";
  for (const [key, val] of Object.entries(musicMap)) {
    if (productNameLower.includes(key)) { musicStyle = val; break; }
  }

  let colorWorld = "deep moody dark universe — black background, obsidian surface, single backlight through bottle";
  for (const [key, val] of Object.entries(colorWorldMap)) {
    if (productNameLower.includes(key)) { colorWorld = val; break; }
  }

  // Detect product type — can vs bottle
  const isCan = product.product_type === "can" || product.productType === "can" || productNameLower.includes("can");
  const isAverse = productNameLower.includes("averse");
  const averseDomeColor = isAverse ? getAverseDomeColor(productNameLower) : null;

  // Can: resolve flavor preset (outdoor lifestyle), override colorWorld from preset
  const canPreset = isCan ? getCanPreset(product) : null;
  if (canPreset) {
    colorWorld = canPreset.colorPalette;
  }

  // Narrative style per product type
  // CAN: dark dramatic — label macro → can reveal → can rotation → props detail → wide hero
  // PAMPLEMOUSSE/bright: condensation world + aerial fruit splash + lifestyle
  // ALL BOTTLES: cocktail pour narrative (Framboise reference style)
  const condensationProducts = ["pamplemousse", "grapefruit", "litchi", "tangerine", "melon"];
  const useCondensationNarrative = !isCan && condensationProducts.some(k => productNameLower.includes(k));

  // Build shot template based on product type
  const canIsAlcoholic = canPreset ? isAlcoholicCan(product) : false;

  const shotTemplate = isCan && canPreset ? (canIsAlcoholic ? `
SHOT 1 — CAN CONDENSATION MACRO (0–3s):
Camera opens in EXTREME CLOSE-UP on THIS EXACT CAN surface — cold aluminum covered in dense condensation water droplets — ${canPreset.heroScene.split(".")[0]} visible in soft focus behind — camera slowly slides up the can from label base to pull-tab, revealing the full label design — single condensation drop slides down in slow motion — ${canPreset.colorPalette} world establishing

SHOT 2 — OUTDOOR SOCIAL SCENE (3–8s):
Medium shot establishing ${canPreset.heroScene} — THIS EXACT CAN centered prominently in foreground — ${canPreset.props} arranged naturally around the can — vivid ${canPreset.colorPalette} world — golden outdoor light hitting the aluminum surface — vibrant social summer energy

SHOT 3 — MOTION ACTION (8–11s):
${canPreset.motionStyle} — THIS EXACT CAN stays prominently in frame throughout — ${canPreset.colorPalette} saturating the entire scene — high energy, motion-forward, vivid summer action

SHOT 4 — CAN + PROPS MACRO (11–13s):
EXTREME CLOSE-UP — THIS EXACT CAN label in sharp full focus, completely undistorted and readable — ${canPreset.props.split(",")[0].trim()} and condensation at the base — ${canPreset.colorPalette} — shallow depth of field${canPreset.keyNote ? ` — ${canPreset.keyNote}` : ""}

SHOT 5 — WIDE OUTDOOR HERO (13–15s):
Camera rises to wide outdoor hero shot — ${canPreset.heroScene} at full scale — THIS EXACT CAN centered and prominent — ${canPreset.colorPalette} world in full — peak summer energy — cinematic final freeze frame
` : `
SHOT 1 — CAN NATURE MACRO (0–3s):
Camera opens in EXTREME CLOSE-UP on THIS EXACT CAN resting on a natural surface — ${canPreset.heroScene.split(".")[0]} visible in soft focus behind — natural morning or afternoon light catching the cold aluminum — condensation on can surface — camera slowly reveals full label design, sharp and fully undistorted

SHOT 2 — OUTDOOR NATURE SCENE (3–8s):
Medium shot establishing ${canPreset.heroScene} — THIS EXACT CAN placed naturally in foreground — ${canPreset.props} arranged around the can — active, fresh, health-forward energy — ${canPreset.colorPalette} — crisp natural lighting

SHOT 3 — NATURE MOTION (8–11s):
${canPreset.motionStyle} — THIS EXACT CAN stays prominently in frame throughout — ${canPreset.colorPalette} — fresh clean energy, personal vitality, outdoor adventure${canPreset.keyNote ? ` — ${canPreset.keyNote}` : ""}

SHOT 4 — CAN + FRESH PROPS MACRO (11–13s):
EXTREME CLOSE-UP — ${canPreset.props.split(",")[0].trim()} and natural elements in tight frame around THIS EXACT CAN — ${canPreset.colorPalette} — can label fully readable and undistorted — fresh and natural mood

SHOT 5 — WIDE OUTDOOR HERO (13–15s):
Wide outdoor shot — ${canPreset.heroScene} fully established — THIS EXACT CAN prominent in frame — ${canPreset.colorPalette} — active nature energy — cinematic final freeze
`) : useCondensationNarrative ? `
SHOT 1 — CONDENSATION MACRO (0–4s):
Camera opens in EXTREME CLOSE-UP on the bottle glass surface — dense condensation water droplets coat the entire glass — camera slowly rotates 90 degrees around the bottle showing the wet glass from different angles — hot pink background saturating the entire frame — camera gently racks focus between the water drops and the label — NO full bottle shape visible yet — just the condensation-covered curved glass surface and partial label

SHOT 2 — FLAVOR TEXT MACRO (4–6s):
Camera tightens to an extreme close-up on just the flavor text area of the label — flavor name fills the entire frame — condensation drops scattered on the hot-pink label surface — camera holds still, breathing gently — pure abstract hot pink

SHOT 3 — AERIAL FRUIT SPLASH (6–9s):
AERIAL overhead camera angle looking straight down — hot pink glossy surface below — multiple halved grapefruit slices fall from above in SLOW MOTION — pink water EXPLODES upward in a dramatic splash — this exact bottle appears small in the center of the water eruption — pure saturated hot pink world — no horizon visible, just pink surface + fruit + water

SHOT 4 — LIFESTYLE HERO (9–12s):
Eye-level medium shot — a person wearing soft pink fabric stands behind a glossy hot-pink reflective table — THIS EXACT BOTTLE stands upright centered-right on the glossy surface — mirror-perfect reflection of the bottle below — halved grapefruit slices and a rosemary sprig scattered at bottle base — person's hand reaches toward a whole grapefruit on the left — warm diffused pink lighting everywhere

SHOT 5 — PRODUCT CLOSE (12–15s):
Camera slowly pushes in toward the bottle on the glossy surface — bottle perfectly centered — label fully sharp and readable — condensation drops catching pink light — glossy surface reflection crisp — cinematic still freeze
` : `
SHOT 1 — LABEL MACRO (0–3s):
Camera opens in EXTREME CLOSE-UP on the bottle label — label fills the entire 9:16 frame — warm backlight glows through the colored glass — condensation micro-droplets on the glass surface — camera gently racks focus from glass texture to label — No full bottle visible yet — just label filling the frame

SHOT 2 — POUR ACTION (3–8s):
A hand slowly tilts THIS EXACT BOTTLE and dispenses a precise 1-ounce measure — a thin, controlled trickle of liquid falls in SLOW MOTION from the bottle neck into a short crystal rock glass holding 2 large clear ice cubes — the narrow pour stream is the focal point — sharp and jewel-like — this is a spirit pour, NOT a juice pour — minimal liquid, refined and deliberate — background: ${colorWorld} — teal-turquoise backlight creates cinematic contrast behind the thin stream — shallow depth of field — ice cubes softly bokeh

SHOT 3 — ${isAverse ? "DOME BASE DETAIL" : "CORK MACRO"} (8–10s):
${isAverse
  ? `EXTREME CLOSE-UP on the distinctive ${averseDomeColor} dome-shaped base cap of THIS BOTTLE — smooth curved dome fills the frame — Cherry River branding visible on the dome surface — ${colorWorld} — shallow depth of field with soft background bokeh`
  : `EXTREME CLOSE-UP on the wooden cork stopper of THIS BOTTLE — warm amber wood grain texture fills the frame — engraved stag animal logo on top of cork — shallow depth of field with soft background bokeh`}

SHOT 4 — INSIDE GLASS MACRO (10–12s):
Camera goes INSIDE the rock glass — extreme close-up — a shallow 1-oz layer of colored liquid swirling gently around the base of large ice cubes — ice towers above the liquid line — ice catches light and refracts — liquid rich, jewel-like, saturated — the glass is not full, just a measured spirit serving

SHOT 5 — WIDE HERO (12–15s):
Camera lifts to overhead-angled establishing shot — THIS BOTTLE resting on its side on dark charcoal stone-textured surface — filled rock glass beside it — props scattered artfully: ${flavorProps} — ${colorWorld} — atmospheric haze fills the scene — cinematic final freeze frame
`;

  // Build message content — include actual product image so Opus can describe the bottle precisely
  const userContent = [];

  if (product.image) {
    try {
      const imgResp = await fetch(product.image);
      if (imgResp.ok) {
        const imgBuffer = Buffer.from(await imgResp.arrayBuffer());
        const base64 = imgBuffer.toString("base64");
        const contentType = imgResp.headers.get("content-type") || "image/jpeg";
        userContent.push({
          type: "image",
          source: { type: "base64", media_type: contentType, data: base64 },
        });
      }
    } catch (e) {
      console.error("Failed to fetch product image for Opus:", e.message);
    }
  }

  const opusText = isCan && canPreset ? `You are a world-class beverage advertising director specializing in high-energy lifestyle content.
Study the EXACT can image attached — memorize the can shape, label design, colors, pull-tab.
Write a Seedance 2 Pro video prompt for this Cherry River RTD can.
Style reference: Red Bull, Celsius — vibrant, saturated, motion-forward, outdoor lifestyle energy.

PRODUCT: "${product.name}"
HERO SCENE: ${canPreset.heroScene}
COLOR WORLD: ${canPreset.colorPalette}
PROPS: ${canPreset.props}
MUSIC: ${musicStyle}
DIRECTOR'S NOTE: "${userScene}"

SUBJECT (use in all shots): From the image — describe this exact can: tall slim 355ml cylindrical aluminum can, exact label design and colors as shown in the image, Cherry River branding, pull-tab at top. Be precise.

Follow this EXACT shot structure:
${shotTemplate}
MUSIC: ${musicStyle} — starts immediately at 0s, builds with the outdoor energy, fades naturally at end.
SOUND & FOLEY: condensation drops on cold aluminum, outdoor ambient sounds, ${canPreset.isNonAlcoholic ? "fresh nature sounds, water sounds" : "crowd energy sounds, summer ambiance"}, prop interaction sounds.
CINEMATIC GRADE: vibrant saturated outdoor colors | condensation on cold aluminum | ${canPreset.isNonAlcoholic ? "crisp natural morning or afternoon light" : "golden summer outdoor light"} | motion-forward energy | shallow depth of field on label | high energy beverage commercial grade.

CRITICAL RULES:
- Follow the shot sequence EXACTLY as written above — do not change the order
- THIS EXACT CAN must be prominently visible and readable in EVERY shot
- Can label must be UNDISTORTED and READABLE — never warped, never bent
- Can stands PERFECTLY UPRIGHT at 90 degrees in all shots
- NEVER name spirit or drink categories (no gin, vodka, rum, spirits, cocktail, mocktail, etc.)
- NEVER include text/typography or label-reading instructions
- Under 3800 characters total

Return ONLY the final prompt. No preamble. No explanation.`
  : `You are a world-class luxury beverage advertising director. Study the EXACT product bottle image attached — memorize bottle shape, glass color, label color and design, cork material. Write a Seedance 2 Pro video prompt replicating the Cherry River reference commercial style EXACTLY.

PRODUCT: "${product.name}"
PROPS: ${flavorProps}
COLOR & LIGHTING: ${colorWorld}
MUSIC: ${musicStyle}
DIRECTOR'S NOTE: "${userScene}"

SUBJECT (use in all shots): ${isAverse ? `This is an Averse bottle — frosted matte white elongated body, distinctive ${averseDomeColor} dome-shaped base cap, minimalist rectangular label with 'a|v|e|r|s|e' architectural text — NO cork. Confirm from image and describe precisely.` : "From the image — describe this exact bottle: round glass bottle shape, liquid color visible through glass, label color and rectangular shape, wooden cork stopper. Be precise."}

Follow this EXACT shot structure:
${shotTemplate}
MUSIC: ${musicStyle} — starts at 0s immediately, builds with the action, fades naturally at end.
SOUND & FOLEY: condensation water drops on glass, liquid movement sounds, prop impact sounds appropriate to the scene.
CINEMATIC GRADE: extreme macro glass texture | dense condensation water drops | slow-motion physics | saturated color grade | shallow depth of field | single specular highlight.

CRITICAL RULES:
- Follow the shot sequence EXACTLY as written above — do not change the order
- NEVER name spirit categories or drink types (no gin, vodka, rum, liqueur, spirits, cocktail, mocktail, mojito, margarita, etc.)
- NEVER include text/typography/label-reading instructions
- Under 3800 characters total

Return ONLY the final prompt. No preamble. No explanation.`;

  userContent.push({ type: "text", text: opusText });

  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 800,
    messages: [{ role: "user", content: userContent }],
  });

  const enhanced = message.content[0].text.trim();

  // Strip alcohol + drink-category words that trigger Seedance "shark" content filter
  const alcoholWords = /\b(alcohol|alcoholic|spirits?|liqueur|liquor|gin|vodka|whisky|whiskey|rum|tequila|brandy|cognac|wine|beer|booze|tipple|cocktail|cocktails|mocktail|mocktails|mojito|margarita|aperitif|nightcap|mixology)\b/gi;

  // Strip music words that trigger Seedance OutputAudioRisk
  // Replace with safe neutral equivalents
  const riskyAudioWords = [
    [/\bmoody\b/gi, "flowing"],
    [/\bminor key\b/gi, "melodic"],
    [/\bominous\b/gi, "gentle"],
    [/\bhaunting\b/gi, "smooth"],
    [/\bbrooding\b/gi, "flowing"],
    [/\bsinister\b/gi, "subtle"],
    [/\bmelancholic\b/gi, "serene"],
    [/\bforeboding\b/gi, "building"],
    [/\baggressive\b/gi, "energetic"],
    [/\bdeep sub-bass\b/gi, "warm bass"],
    [/\bcinematic build\b/gi, "rhythmic build"],
  ];

  let sanitized = enhanced.replace(alcoholWords, "");
  for (const [pattern, replacement] of riskyAudioWords) {
    sanitized = sanitized.replace(pattern, replacement);
  }
  sanitized = sanitized
    .replace(/\s{2,}/g, " ")
    .replace(/\s([,.])/g, "$1")
    .trim();

  // Hard safety cap — Seedance rejects anything over 4000 chars; use 3500 for safe margin
  return sanitized.length > 3500 ? sanitized.slice(0, 3500).trimEnd() : sanitized;
}

// Video aspect ratio map for text_to_video mode
const videoRatioMap = {
  "9:16": "9:16",
  "16:9": "16:9",
  "1:1":  "1:1",
  "4:5":  "9:16",
};

// POST /api/generate-video — start Seedance 2 video generation via PiAPI
app.post("/api/generate-video", async (req, res) => {
  try {
    const { productId, customPrompt, duration, format } = req.body;
    if (!productId) return res.status(400).json({ error: "productId is required" });

    const { data: product } = await supabase.from("products").select("*").eq("id", productId).maybeSingle();
    if (!product) return res.status(404).json({ error: "Product not found" });

    const dur = Math.min(Math.max(parseInt(duration) || 10, 4), 15);
    const userScene = customPrompt?.trim() || "cinematic product reveal with dramatic lighting";

    let fullPrompt;
    try {
      fullPrompt = await enhanceVideoPrompt(userScene, product);
    } catch (e) {
      console.error("Video prompt enhancement failed:", e.message);
      fullPrompt = `premium artisan glass bottle, ${userScene}, cinematic luxury product commercial, dark marble surface, dramatic rim lighting`;
    }

    const hasImage = !!product.image;
    const piPayload = {
      model: "seedance",
      task_type: "seedance-2",
      input: {
        prompt: fullPrompt,
        mode: hasImage ? "first_last_frames" : "text_to_video",
        duration: dur,
        ...(hasImage
          ? { image_urls: [product.image] }
          : { aspect_ratio: videoRatioMap[format] || "9:16" }
        ),
      },
    };

    const piResp = await fetch("https://api.piapi.ai/api/v1/task", {
      method: "POST",
      headers: {
        "X-API-Key": process.env.PIAPI_API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(piPayload),
    });
    const piData = await piResp.json();
    if (piData.code !== 200 || !piData.data?.task_id) {
      throw new Error(`Seedance start failed: ${JSON.stringify(piData)}`);
    }

    const taskId = piData.data.task_id;
    taskCache.set(taskId, { piPayload });
    console.log(`Seedance 2 job started: ${taskId}`);
    res.json({ predictionId: taskId, status: "created" });
  } catch (error) {
    console.error("Video generation start error:", error);
    res.status(500).json({ error: error.message || "Failed to start video generation" });
  }
});

// GET /api/video-status/:id — poll Seedance 2 video status via PiAPI
app.get("/api/video-status/:id", async (req, res) => {
  try {
    const taskId = req.params.id;
    const filename = `${taskId}.mp4`;

    // Helper: download video from URL and save to Supabase videos bucket
    const saveVideoToSupabase = async (videoUrl) => {
      const contentResp = await fetch(videoUrl);
      if (!contentResp.ok) throw new Error(`Video download failed: ${contentResp.status}`);
      const buffer = Buffer.from(await contentResp.arrayBuffer());
      console.log(`Downloaded video ${taskId}: ${buffer.length} bytes`);
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from(VIDEO_BUCKET)
        .upload(filename, buffer, { contentType: "video/mp4", upsert: true });
      if (uploadError) throw new Error(`Supabase upload: ${uploadError.message}`);
      const { data: { publicUrl } } = supabase.storage.from(VIDEO_BUCKET).getPublicUrl(uploadData.path);
      return publicUrl;
    };

    // Fast path: return cached Supabase URL if already saved
    const { data: { publicUrl: existingUrl } } = supabase.storage.from(VIDEO_BUCKET).getPublicUrl(filename);
    const headCheck = await fetch(existingUrl, { method: "HEAD" }).catch(() => null);
    if (headCheck?.ok) return res.json({ status: "succeeded", videoUrl: existingUrl });

    // Check if background WM removal finished since last poll
    const bgWm = wmTasks.get(taskId);
    if (bgWm?.status === "done") {
      wmTasks.delete(taskId);
      return res.json({ status: "succeeded", videoUrl: bgWm.videoUrl });
    }
    if (bgWm?.status === "processing") {
      return res.json({ status: "removing-watermark" });
    }

    // ── NORMAL GENERATION STATUS CHECK ──
    const piResp = await fetch(`https://api.piapi.ai/api/v1/task/${taskId}`, {
      headers: { "X-API-Key": process.env.PIAPI_API_KEY },
    });
    const data = await piResp.json();
    if (data.code !== 200) throw new Error(`PiAPI status check failed: ${JSON.stringify(data)}`);

    const task = data.data;
    const taskStatus = task.status?.toLowerCase();

    if (taskStatus === "completed") {
      // Check all known PiAPI output field names for the video URL
      const videoUrlFromPiAPI =
        task.output?.video ||
        task.output?.url ||
        task.output?.video_url ||
        task.output?.works?.[0]?.video ||
        task.output?.works?.[0]?.url ||
        (Array.isArray(task.output?.videos) ? task.output.videos[0] : null) ||
        (typeof task.output === "string" ? task.output : null);

      if (!videoUrlFromPiAPI) {
        console.error("Generation completed but no video URL. Full output:", JSON.stringify(task.output));
        return res.json({ status: "processing", progress: 0 });
      }
      console.log(`Generation completed. PiAPI URL: ${videoUrlFromPiAPI}`);

      // Run WM removal synchronously (works on Vercel — no background tasks)
      try {
        console.log(`[WM] Removing watermark for ${taskId}...`);
        const cleanUrl = await removeWatermarkWithFfmpeg(videoUrlFromPiAPI, taskId);
        console.log(`[WM] Done: ${cleanUrl}`);
        return res.json({ status: "succeeded", videoUrl: cleanUrl });
      } catch (wmErr) {
        console.error(`[WM] Failed, uploading original:`, wmErr.message);
        // Fallback: save original (with watermark) to Supabase so URL is permanent
        try {
          const raw = await fetch(videoUrlFromPiAPI);
          if (raw.ok) {
            const buf = Buffer.from(await raw.arrayBuffer());
            const fallbackUrl = await uploadVideoToSupabase(buf, taskId);
            return res.json({ status: "succeeded", videoUrl: fallbackUrl });
          }
        } catch {}
        return res.json({ status: "succeeded", videoUrl: videoUrlFromPiAPI });
      }

    } else if (taskStatus === "failed") {
      const rawErr = task.error?.message || task.logs?.join(", ") || "Generation failed";
      // Map known Seedance error codes to user-friendly messages
      let errMsg = rawErr;
      if (rawErr.includes("OutputAudioRisk")) {
        // Auto-retry: strip all MUSIC/SOUND lines from cached prompt and re-submit
        const cached = taskCache.get(taskId);
        if (cached) {
          const safePrompt = cached.piPayload.input.prompt
            .replace(/^MUSIC:.*$/gim, "")
            .replace(/^SOUND & FOLEY:.*$/gim, "")
            .replace(/^SOUND:.*$/gim, "")
            .replace(/\n{3,}/g, "\n\n")
            .trim();
          const retryPayload = {
            ...cached.piPayload,
            input: { ...cached.piPayload.input, prompt: safePrompt },
          };
          try {
            const retryResp = await fetch("https://api.piapi.ai/api/v1/task", {
              method: "POST",
              headers: { "X-API-Key": process.env.PIAPI_API_KEY, "Content-Type": "application/json" },
              body: JSON.stringify(retryPayload),
            });
            const retryData = await retryResp.json();
            if (retryData.code === 200 && retryData.data?.task_id) {
              const newTaskId = retryData.data.task_id;
              taskCache.set(newTaskId, { piPayload: retryPayload });
              taskCache.delete(taskId);
              console.log(`OutputAudioRisk auto-retry: ${taskId} → ${newTaskId}`);
              return res.json({ status: "retrying", taskId: newTaskId });
            }
          } catch (retryErr) {
            console.error("Auto-retry failed:", retryErr.message);
          }
        }
        errMsg = "Audio flagged by Seedance. Please try again.";
      } else if (rawErr.toLowerCase().includes("shark not pass") || rawErr.toLowerCase().includes("shark")) {
        // Seedance "shark" = prompt-level content safety filter — auto-retry with a stripped-down neutral prompt
        const cached = taskCache.get(taskId);
        if (cached) {
          const sharkSafePrompt = cached.piPayload.input.prompt
            .replace(/\b(cocktail|cocktails|mocktail|mojito|margarita|aperitif|alcohol|alcoholic|spirit|spirits|liqueur|gin|vodka|rum|whisky|whiskey|tequila|brandy|wine|beer|booze|pour|pouring|liquid|drink|drinking|beverage|bar|bartender|mixology)\b/gi, "product")
            .replace(/^MUSIC:.*$/gim, "")
            .replace(/^SOUND.*:.*$/gim, "")
            .replace(/\n{3,}/g, "\n\n")
            .trim();
          const retryPayload = {
            ...cached.piPayload,
            input: { ...cached.piPayload.input, prompt: sharkSafePrompt },
          };
          try {
            const retryResp = await fetch("https://api.piapi.ai/api/v1/task", {
              method: "POST",
              headers: { "X-API-Key": process.env.PIAPI_API_KEY, "Content-Type": "application/json" },
              body: JSON.stringify(retryPayload),
            });
            const retryData = await retryResp.json();
            if (retryData.code === 200 && retryData.data?.task_id) {
              const newTaskId = retryData.data.task_id;
              taskCache.set(newTaskId, { piPayload: retryPayload });
              taskCache.delete(taskId);
              console.log(`Shark filter auto-retry: ${taskId} → ${newTaskId}`);
              return res.json({ status: "retrying", taskId: newTaskId });
            }
          } catch (retryErr) {
            console.error("Shark auto-retry failed:", retryErr.message);
          }
        }
        errMsg = "Content flagged by Seedance. Retrying with a safer prompt — please wait a moment and try again.";
      } else if (rawErr.includes("OutputVideoRisk")) {
        errMsg = "OutputVideoRisk: Seedance flagged the video content. Try a different scene description.";
      } else if (rawErr.includes("InputAudioRisk") || rawErr.includes("InputVideoRisk")) {
        errMsg = "InputRisk: Your product image or prompt was flagged. Try a cleaner product photo or simpler scene description.";
      } else if (rawErr.includes("NSFW") || rawErr.includes("nsfw")) {
        errMsg = "Content flagged: Scene description contains restricted content. Simplify the prompt and try again.";
      } else if (rawErr.includes("timeout") || rawErr.includes("Timeout")) {
        errMsg = "Generation timed out. Seedance servers are busy — please try again in a moment.";
      } else if (rawErr.includes("credit") || rawErr.includes("balance")) {
        errMsg = "Insufficient PiAPI credits. Please top up your PiAPI account balance.";
      } else if (rawErr.includes("exceeds maximum length") || rawErr.includes("prompt exceeds")) {
        errMsg = "Prompt too long — please try again. The system will use a shorter version.";
      }
      res.json({ status: "failed", error: errMsg, rawError: rawErr });
    } else {
      // pending, staged, processing
      res.json({ status: taskStatus, progress: 0 });
    }
  } catch (error) {
    console.error("Video status error:", error);
    res.status(500).json({ error: error.message });
  }
});

// Health check
app.get("/api/health", (_req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3001;
if (process.env.VERCEL !== '1') {
  app.listen(PORT, () => console.log(`Cherry River AI backend running on http://localhost:${PORT}`));
}

mountMcp(app);

export default app;
