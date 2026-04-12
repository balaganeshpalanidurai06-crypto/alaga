const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const OpenAI = require('openai');
const Groq = require('groq-sdk');
const sharp = require('sharp');

require('dotenv').config();

const PORT = process.env.PORT || 5000;

// API KEYS
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;

// OpenRouter
const openrouter = new OpenAI({
  apiKey: OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
});

// Groq
const groq = new Groq({ apiKey: GROQ_API_KEY });

const app = express();
app.use(cors());
app.use(express.json());

let tempHistory = [];
let historyIdCounter = 1;

// Upload setup
const uploadDir = './uploads';
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
});

const upload = multer({ storage });

// MODELS
const GROQ_MODELS = [
  "llama-3.1-8b-instant",
  "mixtral-8x7b-32768"
];

// ✅ WORKING VISION MODELS (OpenRouter)
const VISION_MODELS = [
  "openai/gpt-4o-mini",       // BEST (fast + stable)
  "google/gemini-flash-1.5",  // backup
];

// ---------------- CHAT ----------------
app.post('/api/chat', async (req, res) => {
  const { message } = req.body;

  res.setHeader("Content-Type", "text/event-stream");

  for (const model of GROQ_MODELS) {
    try {
      const completion = await groq.chat.completions.create({
        model,
        messages: [
          { role: "system", content: "You are AURA AI assistant." },
          { role: "user", content: message }
        ],
        stream: true,
      });

      for await (const chunk of completion) {
        const text = chunk.choices[0]?.delta?.content;
        if (text) res.write(text);
      }

      res.end();
      return;
    } catch (err) {
      console.log(`❌ Groq ${model} failed`);
    }
  }

  res.write("AI unavailable");
  res.end();
});

// ---------------- VISION ----------------
app.post('/api/vision', upload.single('image'), async (req, res) => {
  const { question } = req.body;
  const imageFile = req.file;

  if (!imageFile) return res.status(400).json({ error: "No image" });

  try {
    const buffer = await sharp(imageFile.path)
      .resize(800, 800, { fit: 'inside' })
      .jpeg({ quality: 80 })
      .toBuffer();

    fs.unlinkSync(imageFile.path);

    const base64 = `data:image/jpeg;base64,${buffer.toString('base64')}`;
    const prompt = question || "Describe this image";

    let result = null;

    for (const model of VISION_MODELS) {
      try {
        console.log("Trying:", model);

        const response = await openrouter.chat.completions.create({
          model,
          messages: [{
            role: "user",
            content: [
              { type: "text", text: prompt },
              { type: "image_url", image_url: { url: base64 } }
            ]
          }]
        });

        result = response.choices[0].message.content;
        console.log("✅ Success:", model);
        break;

      } catch (err) {
        console.log("❌ Failed:", model, err.message);
      }
    }

    if (!result) {
      result = "Vision AI overloaded. Try again later.";
    }

    res.json({ response: result });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Vision failed" });
  }
});

// ---------------- HISTORY ----------------
app.get('/api/all-history', (req, res) => {
  res.json(tempHistory.reverse());
});

app.post('/api/save-history', (req, res) => {
  const { type, question, answer } = req.body;

  const record = {
    id: `tmp_${historyIdCounter++}`,
    type,
    question,
    answer,
    timestamp: new Date()
  };

  tempHistory.push(record);
  res.json({ success: true });
});

// ---------------- START ----------------
app.listen(PORT, () => {
  console.log(`🚀 Server running on ${PORT}`);
});