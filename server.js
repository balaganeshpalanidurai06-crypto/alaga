// server.js - AURA AI v6.0
// Vision chain: Groq Vision → OpenRouter Vision → Gemini Vision → BLIP (no key)
// Chat  chain:  Groq → OpenRouter → Gemini

const express = require('express');
const cors    = require('cors');
const multer  = require('multer');
const path    = require('path');
const fs      = require('fs');
const axios   = require('axios');
const sharp   = require('sharp');

require('dotenv').config();

const PORT           = process.env.PORT || 5000;
const GROQ_API_KEY   = process.env.GROQ_API_KEY;
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// ── Uploads ──────────────────────────────────────────────────────────────────
const uploadDir = './uploads';
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename:    (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname)),
});
const upload = multer({ storage, limits: { fileSize: 10 * 1024 * 1024 } });

console.log('\n╔══════════════════════════════════════╗');
console.log('║       AURA AI SERVER v6.0            ║');
console.log('╚══════════════════════════════════════╝');
console.log('\n🔑 Keys:');
console.log(`  Groq       : ${GROQ_API_KEY   ? '✅ (Chat + Vision!)' : '❌ → console.groq.com (FREE, 2 min)'}`);
console.log(`  OpenRouter : ${OPENROUTER_KEY ? '✅' : '❌ → openrouter.ai/keys'}`);
console.log(`  Gemini     : ${GEMINI_API_KEY ? '✅' : '❌ → aistudio.google.com'}\n`);

// ════════════════════════════════════════════════════════════════════════════
//  CHAT PROVIDERS
// ════════════════════════════════════════════════════════════════════════════

// ─── Groq Chat ───────────────────────────────────────────────────────────────
async function tryGroqChat(messages) {
  if (!GROQ_API_KEY) return null;
  const models = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'llama3-70b-8192', 'gemma2-9b-it'];
  for (const model of models) {
    try {
      console.log(`  [Groq Chat] ${model}`);
      const res = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        { model, messages, max_tokens: 2048, temperature: 0.7 },
        { headers: { Authorization: `Bearer ${GROQ_API_KEY}`, 'Content-Type': 'application/json' }, timeout: 20000 }
      );
      const text = res.data?.choices?.[0]?.message?.content;
      if (text) { console.log(`  [Groq Chat] ✅`); return { text, provider: `Groq/${model}` }; }
    } catch (e) {
      const msg = e.response?.data?.error?.message || e.message;
      console.log(`  [Groq Chat] ❌ ${msg.substring(0, 80)}`);
      if (msg.includes('Invalid API Key') || msg.includes('401')) break;
    }
  }
  return null;
}

// ─── OpenRouter Chat ─────────────────────────────────────────────────────────
async function tryOpenRouterChat(messages) {
  if (!OPENROUTER_KEY) return null;
  const models = [
    'meta-llama/llama-3.3-70b-instruct:free',
    'meta-llama/llama-3.1-8b-instruct:free',
    'google/gemma-3-27b-it:free',
    'mistralai/mistral-7b-instruct:free',
    'qwen/qwen3-8b:free',
    'deepseek/deepseek-r1:free',
    'microsoft/phi-4:free',
  ];
  for (const model of models) {
    try {
      console.log(`  [OR Chat] ${model}`);
      const res = await axios.post(
        'https://openrouter.ai/api/v1/chat/completions',
        { model, messages, max_tokens: 2048, temperature: 0.7 },
        {
          headers: {
            Authorization: `Bearer ${OPENROUTER_KEY}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://aura-ai.local',
            'X-Title': 'AURA AI',
          },
          timeout: 25000,
        }
      );
      const text = res.data?.choices?.[0]?.message?.content;
      if (text) { console.log(`  [OR Chat] ✅`); return { text, provider: `OpenRouter/${model}` }; }
    } catch (e) {
      const msg = e.response?.data?.error?.message || e.message;
      console.log(`  [OR Chat] ❌ ${msg.substring(0, 80)}`);
      if (msg.includes('User not found') || msg.includes('401')) break;
    }
  }
  return null;
}

// ─── Gemini Chat ─────────────────────────────────────────────────────────────
async function tryGeminiChat(text) {
  if (!GEMINI_API_KEY) return null;
  const models = ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash'];
  for (const model of models) {
    for (const ver of ['v1beta', 'v1']) {
      try {
        console.log(`  [Gemini Chat] ${model}`);
        const res = await axios.post(
          `https://generativelanguage.googleapis.com/${ver}/models/${model}:generateContent?key=${GEMINI_API_KEY}`,
          { contents: [{ parts: [{ text }] }] },
          { timeout: 25000 }
        );
        const reply = res.data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (reply) { console.log(`  [Gemini Chat] ✅`); return { text: reply, provider: `Gemini/${model}` }; }
      } catch (e) {
        const msg = e.response?.data?.error?.message || e.message;
        console.log(`  [Gemini Chat] ❌ ${msg.substring(0, 80)}`);
        if (msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED')) break;
      }
    }
  }
  return null;
}

async function callAI(messages) {
  console.log('\n🤖 Chat chain...');
  let r;
  r = await tryGroqChat(messages);       if (r) return r;
  r = await tryOpenRouterChat(messages); if (r) return r;
  r = await tryGeminiChat(messages.map(m => m.content).join('\n')); if (r) return r;
  return null;
}

// ════════════════════════════════════════════════════════════════════════════
//  VISION PROVIDERS
// ════════════════════════════════════════════════════════════════════════════

// 1️⃣  Groq Vision — Llama 4 Scout/Maverick (multimodal, replacing deprecated llama-3.2-vision)
async function tryGroqVision(base64Image, question) {
  if (!GROQ_API_KEY) return null;
  const visionModels = [
    'meta-llama/llama-4-scout-17b-16e-instruct',   // Llama 4 Scout — vision support ✅
    'meta-llama/llama-4-maverick-17b-128e-instruct', // Llama 4 Maverick — vision support ✅
  ];
  const prompt = question?.trim() || 'Describe this image in detail. What do you see? Include objects, people, colors, setting, and any text visible.';

  for (const model of visionModels) {
    try {
      console.log(`  [Groq Vision] ${model}`);
      const res = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model,
          messages: [{
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Image}` } },
            ],
          }],
          max_tokens: 1024,
          temperature: 0.7,
        },
        {
          headers: { Authorization: `Bearer ${GROQ_API_KEY}`, 'Content-Type': 'application/json' },
          timeout: 30000,
        }
      );
      const text = res.data?.choices?.[0]?.message?.content;
      if (text) { console.log(`  [Groq Vision] ✅`); return { text, provider: `Groq Vision/${model}` }; }
    } catch (e) {
      const msg = e.response?.data?.error?.message || e.message;
      console.log(`  [Groq Vision] ❌ ${msg.substring(0, 100)}`);
      if (msg.includes('Invalid API Key') || msg.includes('401')) break;
    }
  }
  return null;
}

// 2️⃣  OpenRouter Vision — only models with confirmed vision endpoints
async function tryOpenRouterVision(base64Image, question) {
  if (!OPENROUTER_KEY) return null;
  const visionModels = [
    'qwen/qwen2.5-vl-7b-instruct:free',          // Qwen VL — confirmed working ✅
    'qwen/qwen2.5-vl-72b-instruct:free',          // Qwen VL large ✅
    'microsoft/phi-4-multimodal-instruct:free',   // Phi-4 multimodal ✅
    'google/gemini-2.0-flash-exp:free',           // Gemini flash free ✅
    'mistralai/pixtral-12b:free',                 // Pixtral vision ✅
  ];
  const prompt = question?.trim() || 'Describe this image in detail.';

  for (const model of visionModels) {
    try {
      console.log(`  [OR Vision] ${model}`);
      const res = await axios.post(
        'https://openrouter.ai/api/v1/chat/completions',
        {
          model,
          messages: [{
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Image}` } },
            ],
          }],
          max_tokens: 1024,
        },
        {
          headers: {
            Authorization: `Bearer ${OPENROUTER_KEY}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://aura-ai.local',
            'X-Title': 'AURA AI',
          },
          timeout: 30000,
        }
      );
      const text = res.data?.choices?.[0]?.message?.content;
      if (text) { console.log(`  [OR Vision] ✅`); return { text, provider: `OpenRouter Vision/${model}` }; }
    } catch (e) {
      const msg = e.response?.data?.error?.message || e.message;
      console.log(`  [OR Vision] ❌ ${msg.substring(0, 80)}`);
      if (msg.includes('User not found') || msg.includes('401')) break;
    }
  }
  return null;
}

// 3️⃣  Gemini Vision — correct model names for current API
async function tryGeminiVision(base64Image, question) {
  if (!GEMINI_API_KEY) return null;
  const prompt = question?.trim() || 'Describe this image in detail.';
  // Only v1beta works for vision; use gemini-2.0 models only
  const models = [
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash-preview-04-17', // latest
    'gemini-2.5-pro-preview-03-25',
  ];

  for (const model of models) {
    try {
      console.log(`  [Gemini Vision] ${model}`);
      const res = await axios.post(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`,
        {
          contents: [{
            parts: [
              { text: prompt },
              { inline_data: { mime_type: 'image/jpeg', data: base64Image } },
            ],
          }],
        },
        { timeout: 30000 }
      );
      const text = res.data?.candidates?.[0]?.content?.parts?.[0]?.text;
      if (text) { console.log(`  [Gemini Vision] ✅`); return { text, provider: `Gemini Vision/${model}` }; }
    } catch (e) {
      const msg = e.response?.data?.error?.message || e.message;
      console.log(`  [Gemini Vision] ❌ ${msg.substring(0, 80)}`);
      if (msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED')) break; // all gemini models will fail, stop
    }
  }
  return null;
}

// 4️⃣  BLIP (FREE, NO KEY) — HuggingFace public inference
async function tryBLIPVision(imageBuffer, question) {
  const blipModels = [
    'Salesforce/blip-image-captioning-large',
    'Salesforce/blip-image-captioning-base',
    'nlpconnect/vit-gpt2-image-captioning',
  ];

  for (const model of blipModels) {
    try {
      console.log(`  [BLIP] ${model} (no auth)`);

      // Try 1: Send raw bytes
      let caption = null;
      try {
        const res = await axios.post(
          `https://api-inference.huggingface.co/models/${model}`,
          imageBuffer,
          { headers: { 'Content-Type': 'application/octet-stream' }, timeout: 35000 }
        );
        caption = Array.isArray(res.data) ? res.data[0]?.generated_text : res.data?.generated_text;
      } catch {
        // Try 2: Send base64 JSON
        const res2 = await axios.post(
          `https://api-inference.huggingface.co/models/${model}`,
          { inputs: imageBuffer.toString('base64') },
          { headers: { 'Content-Type': 'application/json' }, timeout: 35000 }
        );
        caption = Array.isArray(res2.data) ? res2.data[0]?.generated_text : res2.data?.generated_text;
      }

      if (caption && caption.trim().length > 3) {
        console.log(`  [BLIP] ✅ "${caption.substring(0, 60)}"`);

        // Expand caption with AI if we have any working chat provider
        const expandPrompt = question?.trim()
          ? `Image analysis result: "${caption}"\nUser question: "${question}"\nAnswer the question based on what's visible in the image. Be detailed and helpful.`
          : `An AI analyzed an image and described it as: "${caption}"\nExpand this into a detailed, natural 3-4 sentence description of what's in the image.`;

        const expanded = await callAI([{ role: 'user', content: expandPrompt }]);
        return {
          text: expanded?.text || `Image shows: ${caption}`,
          provider: 'BLIP (free) + AI',
        };
      }
    } catch (e) {
      const errMsg = String(e.response?.data?.error || e.message);
      console.log(`  [BLIP] ❌ ${errMsg.substring(0, 80)}`);

      if (errMsg.includes('loading')) {
        console.log(`  [BLIP] ⏳ Model loading, waiting 10s...`);
        await new Promise(r => setTimeout(r, 10000));
        try {
          const retry = await axios.post(
            `https://api-inference.huggingface.co/models/${model}`,
            imageBuffer,
            { headers: { 'Content-Type': 'application/octet-stream' }, timeout: 35000 }
          );
          const cap = Array.isArray(retry.data) ? retry.data[0]?.generated_text : retry.data?.generated_text;
          if (cap?.trim().length > 3) {
            console.log(`  [BLIP] ✅ (after wait)`);
            return { text: `Image shows: ${cap}`, provider: 'BLIP (free)' };
          }
        } catch {}
      }
    }
  }
  return null;
}

// ════════════════════════════════════════════════════════════════════════════
//  ROUTES
// ════════════════════════════════════════════════════════════════════════════

// ── Vision rate limit tracker (per IP, 2 min cooldown) ───────────────────────
const visionLastCall = new Map(); // ip → timestamp
const VISION_COOLDOWN_MS = 2 * 60 * 1000; // 2 minutes

// ── VISION ────────────────────────────────────────────────────────────────────
app.post('/api/vision', upload.single('image'), async (req, res) => {
  const { question } = req.body;
  if (!req.file) return res.status(400).json({ error: 'No image uploaded' });

  // ── Cooldown check ──────────────────────────────────────────────────────────
  const ip = req.ip || req.connection.remoteAddress || 'unknown';
  const lastCall = visionLastCall.get(ip) || 0;
  const elapsed  = Date.now() - lastCall;
  const remaining = VISION_COOLDOWN_MS - elapsed;

  if (remaining > 0) {
    try { fs.unlinkSync(req.file.path); } catch {}
    const secs = Math.ceil(remaining / 1000);
    return res.status(429).json({
      cooldown: true,
      remainingSeconds: secs,
      response: `COOLDOWN:${secs}`,
    });
  }

  console.log(`\n👁️ Vision: "${question || '(no question)'}"`);

  try {
    const imageBuffer = await sharp(req.file.path)
      .resize(800, 800, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 85 })
      .toBuffer();
    try { fs.unlinkSync(req.file.path); } catch {}

    const base64Image = imageBuffer.toString('base64');

    // Chain: Groq Vision → OpenRouter Vision → Gemini Vision → BLIP (no key)
    let result = null;

    result = await tryGroqVision(base64Image, question);       if (result) { visionLastCall.set(ip, Date.now()); return res.json({ response: result.text, provider: result.provider }); }
    result = await tryOpenRouterVision(base64Image, question); if (result) { visionLastCall.set(ip, Date.now()); return res.json({ response: result.text, provider: result.provider }); }
    result = await tryGeminiVision(base64Image, question);     if (result) { visionLastCall.set(ip, Date.now()); return res.json({ response: result.text, provider: result.provider }); }
    result = await tryBLIPVision(imageBuffer, question);       if (result) { visionLastCall.set(ip, Date.now()); return res.json({ response: result.text, provider: result.provider }); }

    res.json({ response: '⚠️ Vision analysis temporarily unavailable. Please try again in a moment.' });

  } catch (err) {
    console.error('❌ Vision Error:', err.message);
    res.status(500).json({ response: 'Server error during image analysis. Please try again.' });
  }
});

// ── STREAMING CHAT ────────────────────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: 'Message required' });
  console.log(`\n📨 [Stream] "${message.substring(0, 60)}"`);

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const result = await callAI([{ role: 'user', content: message }]);

  if (!result) {
    res.write(
      '⚠️ Chat is unavailable — your API keys need refresh.\n\n' +
      'Get FREE key: https://console.groq.com (2 min)\n' +
      'Add to .env → restart: node server.js'
    );
    return res.end();
  }

  for (let i = 0; i < result.text.length; i++) {
    res.write(result.text[i]);
    if (i % 15 === 0) await new Promise(r => setTimeout(r, 4));
  }
  res.end();
});

// ── SIMPLE CHAT ───────────────────────────────────────────────────────────────
app.post('/api/chat/simple', async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: 'Message required' });
  const result = await callAI([{ role: 'user', content: message }]);
  if (!result) return res.status(503).json({ error: 'All providers unavailable.' });
  res.json({ response: result.text, provider: result.provider });
});

// ── TEST ──────────────────────────────────────────────────────────────────────
app.get('/api/test', async (req, res) => {
  const chatResult = await callAI([{ role: 'user', content: "Reply with: 'AURA online'" }]);
  res.json({
    status: 'OK',
    server: 'AURA AI v6.0',
    chat:   chatResult ? `✅ ${chatResult.provider}` : '❌ All chat providers failed',
    vision: {
      groq:       GROQ_API_KEY   ? '✅ Groq Vision ready (llama-3.2-vision)' : '❌ No key',
      openrouter: OPENROUTER_KEY ? '✅ OpenRouter Vision ready' : '❌ No key',
      gemini:     GEMINI_API_KEY ? '✅ Gemini Vision ready' : '❌ No key',
      blip:       '✅ FREE BLIP (always available, no key needed)',
    },
    keys: {
      groq:       GROQ_API_KEY   ? '✅' : '❌ → console.groq.com',
      openrouter: OPENROUTER_KEY ? '✅' : '❌ → openrouter.ai/keys',
      gemini:     GEMINI_API_KEY ? '✅' : '❌ → aistudio.google.com',
    },
  });
});

// ── HEALTH ────────────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => res.json({ status: 'OK', uptime: `${Math.floor(process.uptime())}s`, version: '6.0' }));

// ── HISTORY ───────────────────────────────────────────────────────────────────
let tempHistory = [];
app.get('/api/history', (req, res) => res.json(tempHistory.slice(-50).reverse()));
app.get('/api/all-history', (req, res) =>
  res.json(tempHistory.slice(-50).reverse().map(h => ({
    ...h,
    formattedTime: new Date(h.timestamp).toLocaleString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: true,
    }),
  })))
);
app.post('/api/save-history', (req, res) => {
  const { type, question, answer } = req.body;
  tempHistory.push({ id: Date.now(), type, question, answer, timestamp: new Date().toISOString() });
  if (tempHistory.length > 200) tempHistory = tempHistory.slice(-200);
  res.json({ success: true });
});
app.delete('/api/history/:id', (req, res) => { tempHistory = tempHistory.filter(h => h.id !== parseInt(req.params.id)); res.json({ success: true }); });
app.delete('/api/all-history', (req, res) => { tempHistory = []; res.json({ success: true }); });
app.get('/', (req, res) => res.json({ name: 'AURA AI', version: '6.0.0' }));

// ── START ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀 AURA AI v6.0 → http://localhost:${PORT}`);
  console.log(`\n👁️  Vision models: Groq Vision + OpenRouter + Gemini + BLIP (free)`);
  console.log(`\n⚡ Get FREE Groq key (Chat + Vision both work):`);
  console.log(`   https://console.groq.com`);
  console.log(`\n✅ Test: http://localhost:${PORT}/api/test\n`);
});