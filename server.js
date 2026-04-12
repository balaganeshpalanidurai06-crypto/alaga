const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const Groq = require('groq-sdk');
const sharp = require('sharp');

require('dotenv').config();

const PORT = process.env.PORT || 5000;

const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

// Clients
const groq = new Groq({ apiKey: GROQ_API_KEY });

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Temp history
let tempHistory = [];
let historyIdCounter = 1;

// Upload setup
const uploadDir = './uploads';
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) =>
    cb(null, Date.now() + path.extname(file.originalname))
});

const upload = multer({ storage });

// Models
const GROQ_MODELS = [
  "llama-3.1-8b-instant",
  "mixtral-8x7b-32768"
];

/* ================= CHAT ================= */
app.post('/api/chat', async (req, res) => {
  const { message } = req.body;

  if (!message) return res.status(400).send("Message required");

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
      console.log(`❌ Chat ${model} failed`);
    }
  }

  res.write("⚠️ AI unavailable");
  res.end();
});

/* ================= VISION (GEMINI 2.5 FLASH) ================= */
app.post('/api/vision', upload.single('image'), async (req, res) => {
  const { question } = req.body;
  const imageFile = req.file;

  if (!imageFile) {
    return res.status(400).json({ error: "No image uploaded" });
  }

  try {
    // Resize & optimize image
    const buffer = await sharp(imageFile.path)
      .resize(800, 800, { fit: 'inside' })
      .jpeg({ quality: 80 })
      .toBuffer();

    // Delete temp file
    fs.unlinkSync(imageFile.path);

    const base64Image = buffer.toString("base64");
    const prompt = question || "Describe this image in detail.";

    // 🔥 Gemini API call
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`,
      {
        contents: [
          {
            parts: [
              { text: prompt },
              {
                inline_data: {
                  mime_type: "image/jpeg",
                  data: base64Image
                }
              }
            ]
          }
        ]
      }
    );

    const result =
      response.data?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "No response from Gemini";

    res.json({ response: result });

  } catch (err) {
    console.error("❌ Gemini Vision Error:", err.response?.data || err.message);

    res.json({
      response: "⚠️ Gemini vision failed. Try again."
    });
  }
});

/* ================= HISTORY ================= */
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

  if (tempHistory.length > 200) {
    tempHistory = tempHistory.slice(-200);
  }

  res.json({ success: true });
});

/* ================= HEALTH ================= */
app.get('/api/health', (req, res) => {
  res.json({
    status: "OK",
    message: "AURA AI running with Gemini Vision + Groq Chat",
    historyCount: tempHistory.length
  });
});

/* ================= START ================= */
app.listen(PORT, () => {
  console.log(`🚀 AURA Server running on ${PORT}`);
  console.log(`🤖 Groq Chat : ${GROQ_API_KEY ? "OK" : "Missing"}`);
  console.log(`👁️ Gemini Vision : ${GEMINI_API_KEY ? "OK" : "Missing"}`);
});