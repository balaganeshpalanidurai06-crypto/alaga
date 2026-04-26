import { useState, useEffect, useRef } from "react";
import axios from "axios";
import Webcam from "react-webcam";

const API_URL = "https://alaga.onrender.com";

const FEATURE_INFO = {
  voice: {
    icon: "🎙️", label: "AI VOICE", sub: "ASSISTANT", color: "#4aff9e",
    rows: [
      ["MODE",     "Continuous Voice Recognition"],
      ["LANGUAGE", "Tamil / English / Multi-lang"],
      ["RESPONSE", "Streaming AI Answer"],
      ["OUTPUT",   "Real-time Speech Synthesis"],
      ["STORAGE",  "Temporary Memory"],
      ["STATUS",   "ONLINE ● ACTIVE"],
    ]
  },
  chatbot: {
    icon: "💬", label: "CHATBOT", sub: "ASSISTANT", color: "#38bdf8",
    rows: [
      ["MODE",   "Multi-turn Conversation"],
      ["CODE",   "Syntax Highlight (Python/JS)"],
      ["STREAM", "Live Token Streaming"],
      ["EXPORT", "Download Chat as .txt"],
      ["COPY",   "One-click Clipboard Copy"],
      ["STATUS", "ONLINE ● ACTIVE"],
    ]
  },
  lense: {
    icon: "📷", label: "AI LENSE", sub: "VISION", color: "#f472b6",
    rows: [
      ["MODE",     "Image Analysis (Vision AI)"],
      ["INPUT",    "Camera Capture / Gallery"],
      ["QUERY",    "Ask anything about image"],
      ["OUTPUT",   "Detailed AI Description"],
      ["COOLDOWN", "2 min between analyses"],
      ["STATUS",   "ONLINE ● ACTIVE"],
    ]
  },
  news: {
    icon: "📰", label: "AI NEWS", sub: "READER", color: "#facc15",
    rows: [
      ["MODE",       "Live News Feed"],
      ["CATEGORIES", "7 Topics (Tech/Sports/...)"],
      ["SOURCE",     "India & World Headlines"],
      ["LISTEN",     "Text-to-Speech Readout"],
      ["CACHE",      "3-min Smart Cache"],
      ["STATUS",     "ONLINE ● ACTIVE"],
    ]
  }
};

const DEFAULT_SETTINGS = {
  speechLang: 'en-US',
  voiceRate: 1.0,
  voicePitch: 1.1,
  appearance: 'dark',
  notifSound: true,
};

const COOLDOWN_SECS = 120; // 2 minutes

const App = () => {
  const [allHistory, setAllHistory] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [currentText, setCurrentText] = useState("");
  const [loading, setLoading] = useState(false);
  const [finalSentences, setFinalSentences] = useState([]);
  const [aiAnswer, setAiAnswer] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [waveValues, setWaveValues] = useState(Array(30).fill(4));
  const [activeFeature, setActiveFeature] = useState(null);
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);

  const [settings, setSettings] = useState(() => {
    try { const s = localStorage.getItem('aura_settings'); return s ? { ...DEFAULT_SETTINGS, ...JSON.parse(s) } : DEFAULT_SETTINGS; }
    catch { return DEFAULT_SETTINGS; }
  });
  const [showSettings, setShowSettings] = useState(false);
  const [settingsTab, setSettingsTab] = useState('general');

  const [weatherData, setWeatherData] = useState({ temp: "27°C", description: "Mostly cloudy", location: "Chennai" });
  const [showWeatherSearch, setShowWeatherSearch] = useState(false);
  const [searchCity, setSearchCity] = useState("");
  const [weatherLoading, setWeatherLoading] = useState(false);

  const [news, setNews] = useState([]);
  const [newsLoading, setNewsLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState("general");
  const [readingNews, setReadingNews] = useState(false);
  const [newsCache, setNewsCache] = useState({});
  const [lastFetchTime, setLastFetchTime] = useState({});

  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [visionQuestion, setVisionQuestion] = useState("");
  const [visionAnswer, setVisionAnswer] = useState("");
  const [visionLoading, setVisionLoading] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [cameraPhoto, setCameraPhoto] = useState(null);

  // ── COOLDOWN STATE ─────────────────────────────────────────────────────────
  const [cooldownSecs, setCooldownSecs] = useState(0); // 0 = ready, >0 = waiting
  const cooldownRef = useRef(null);

  const startCooldown = (secs = COOLDOWN_SECS) => {
    setCooldownSecs(secs);
    clearInterval(cooldownRef.current);
    cooldownRef.current = setInterval(() => {
      setCooldownSecs(prev => {
        if (prev <= 1) { clearInterval(cooldownRef.current); return 0; }
        return prev - 1;
      });
    }, 1000);
  };

  const cooldownPercent = Math.round(((COOLDOWN_SECS - cooldownSecs) / COOLDOWN_SECS) * 100);
  const cooldownMins = Math.floor(cooldownSecs / 60);
  const cooldownSecsRem = cooldownSecs % 60;
  const cooldownActive = cooldownSecs > 0;

  // ──────────────────────────────────────────────────────────────────────────

  const recognitionRef = useRef(null);
  const waveAnimRef = useRef(null);
  const fileInputRef = useRef(null);
  const webcamRef = useRef(null);

  const backgroundVideoUrl = "/Iron Man Avengers Infinity War Version Marvel Desktop Live Wallpaper-1078p.mp4";

  const categories = [
    { id: "general", name: "📰 Top News", emoji: "📰" },
    { id: "technology", name: "💻 Technology", emoji: "💻" },
    { id: "business", name: "💰 Business", emoji: "💰" },
    { id: "entertainment", name: "🎬 Entertainment", emoji: "🎬" },
    { id: "sports", name: "⚽ Sports", emoji: "⚽" },
    { id: "science", name: "🔬 Science", emoji: "🔬" },
    { id: "health", name: "🏥 Health", emoji: "🏥" },
  ];

  const updateSettings = (key, value) => {
    const ns = { ...settings, [key]: value };
    setSettings(ns);
    localStorage.setItem('aura_settings', JSON.stringify(ns));
  };

  const handleFeatureClick = (feature) => {
    setIsTransitioning(true);
    setTimeout(() => { setActiveFeature(feature); setTimeout(() => setIsTransitioning(false), 600); }, 600);
  };

  const handleBackToMenu = () => {
    setIsTransitioning(true);
    setTimeout(() => {
      setActiveFeature(null); setChatMessages([]); setAiAnswer(""); setIsListening(false);
      stopReading(); clearImage(); closeCamera();
      if (recognitionRef.current) { recognitionRef.current.stop(); recognitionRef.current = null; }
      setTimeout(() => setIsTransitioning(false), 600);
    }, 600);
  };

  const clearAllHistory = async () => {
    if (!window.confirm('Clear ALL your conversation history? This cannot be undone.')) return;
    try {
      await axios.delete(`${API_URL}/api/all-history`);
      loadHistory();
      alert('✅ History cleared!');
    } catch { alert('Failed to clear history.'); }
  };

  const fetchNewsFromMultipleSources = async (category) => {
    const categoryMap = { general:"top-headlines", technology:"technology", business:"business", entertainment:"entertainment", sports:"sports", science:"science", health:"health" };
    const apiCategory = categoryMap[category] || "top-headlines";
    try {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 8000);
      const res = await axios.get(`https://api.allorigins.win/raw?url=${encodeURIComponent(`https://inshorts.deta.dev/news?category=${apiCategory === "top-headlines" ? "all" : apiCategory}`)}`, { signal: controller.signal });
      clearTimeout(tid);
      if (res.data?.data) {
        const d = res.data.data.slice(0,15).map(i=>({ title:i.title||"News Update", description:i.content||i.description||"Read more", publishedAt:new Date().toISOString(), source:{name:i.source||"News Source"}, url:i.url||"#" }));
        if (d.length >= 10) return d;
      }
    } catch {}
    return getCategorySpecificNews(category);
  };

  const getCategorySpecificNews = (cat) => {
    const m = {
      general: [
        {title:"India's GDP grows at 8.2% in Q4",description:"Economy shows strong recovery momentum",source:{name:"Economic Times"},publishedAt:new Date().toISOString()},
        {title:"New space mission announced by ISRO",description:"India's next lunar mission scheduled",source:{name:"The Hindu"},publishedAt:new Date().toISOString()},
        {title:"Digital India reaches 1 billion users",description:"Internet penetration grows rapidly",source:{name:"Times of India"},publishedAt:new Date().toISOString()},
      ],
      technology: [
        {title:"5G rollout completes across major cities",description:"High-speed internet now available",source:{name:"TechCrunch"},publishedAt:new Date().toISOString()},
        {title:"Indian AI startup raises $50M funding",description:"New developments in generative AI",source:{name:"YourStory"},publishedAt:new Date().toISOString()},
      ],
      business: [
        {title:"Sensex hits all-time high of 75,000",description:"Markets rally on positive data",source:{name:"Business Standard"},publishedAt:new Date().toISOString()},
        {title:"RBI keeps repo rate at 6.5%",description:"Inflation under control",source:{name:"Financial Express"},publishedAt:new Date().toISOString()},
      ],
      entertainment: [
        {title:"Blockbuster crosses 1000cr worldwide",description:"Indian cinema dominates box office",source:{name:"Filmfare"},publishedAt:new Date().toISOString()},
        {title:"OTT platforms release 50+ originals",description:"Content boom continues",source:{name:"Bollywood Hungama"},publishedAt:new Date().toISOString()},
      ],
      sports: [
        {title:"India wins cricket series vs Australia",description:"Historic victory in away conditions",source:{name:"ESPN Cricinfo"},publishedAt:new Date().toISOString()},
        {title:"Chess grandmaster wins world championship",description:"India's rising star makes history",source:{name:"Sportstar"},publishedAt:new Date().toISOString()},
      ],
      science: [
        {title:"ISRO Aditya-L1 captures first images",description:"Solar study reveals new insights",source:{name:"Space.com"},publishedAt:new Date().toISOString()},
        {title:"New species found in Western Ghats",description:"Biodiversity hotspot yields findings",source:{name:"Nature India"},publishedAt:new Date().toISOString()},
      ],
      health: [
        {title:"Ayushman Bharat covers 500M families",description:"World's largest health scheme",source:{name:"WHO India"},publishedAt:new Date().toISOString()},
        {title:"New cancer treatment center opens",description:"Advanced care in tier-2 cities",source:{name:"Medical News"},publishedAt:new Date().toISOString()},
      ],
    };
    return m[cat] || m.general;
  };

  const fetchNews = async (category = "general") => {
    const now = Date.now(), cached = newsCache[category], lastFetch = lastFetchTime[category] || 0;
    if (cached && (now - lastFetch) < 180000) { setNews(cached); return; }
    setNewsLoading(true);
    try {
      const data = await fetchNewsFromMultipleSources(category);
      const final = data.length >= 10 ? data.slice(0,15) : [...data, ...getCategorySpecificNews(category).slice(0, 10 - data.length)];
      setNews(final);
      setNewsCache(p => ({...p, [category]: final}));
      setLastFetchTime(p => ({...p, [category]: now}));
      await axios.post(`${API_URL}/api/save-history`, { type:"news", question:`📰 Read ${category} news`, answer:`Fetched ${final.length} articles` });
    } catch { setNews(getCategorySpecificNews(category)); }
    setNewsLoading(false);
  };

  const preloadAllNews = async () => {
    for (const cat of ["general","technology","business","entertainment","sports","science","health"]) {
      try {
        const d = await fetchNewsFromMultipleSources(cat);
        if (d?.length > 0) { setNewsCache(p => ({...p, [cat]: d.slice(0,15)})); setLastFetchTime(p => ({...p, [cat]: Date.now()})); }
      } catch {}
    }
  };

  const speakNews = (title, desc) => {
    if (readingNews) { window.speechSynthesis.cancel(); setReadingNews(false); return; }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(`${title}. ${desc}`);
    u.lang = settings.speechLang; u.rate = settings.voiceRate;
    u.onstart = () => setReadingNews(true); u.onend = () => setReadingNews(false);
    window.speechSynthesis.speak(u);
  };
  const stopReading = () => { window.speechSynthesis.cancel(); setReadingNews(false); };

  const formatDate = (ds) => {
    try {
      const d = new Date(ds), now = new Date(), diff = Math.floor((now - d) / 60000);
      if (diff < 1) return "Just now"; if (diff < 60) return `${diff}m ago`;
      if (diff < 1440) return `${Math.floor(diff/60)}h ago`;
      return d.toLocaleDateString('en-IN');
    } catch { return "Recently"; }
  };

  const getWeatherDescription = (code) => {
    const m = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Foggy",51:"Light drizzle",53:"Moderate drizzle",61:"Slight rain",63:"Moderate rain",65:"Heavy rain",71:"Slight snow",73:"Moderate snow",75:"Heavy snow",80:"Slight showers",81:"Moderate showers",82:"Violent showers",95:"Thunderstorm"};
    return m[code] || "Unknown";
  };

  const fetchWeather = async (city) => {
    if (!city.trim()) return;
    setWeatherLoading(true);
    try {
      const geo = await axios.get(`https://geocoding-api.open-meteo.com/v1/search?name=${city}&count=1&language=en&format=json`);
      if (!geo.data.results?.length) { alert(`City "${city}" not found!`); setWeatherLoading(false); return; }
      const loc = geo.data.results[0];
      const wx = await axios.get(`https://api.open-meteo.com/v1/forecast?latitude=${loc.latitude}&longitude=${loc.longitude}&current_weather=true&temperature_unit=celsius`);
      setWeatherData({ temp: `${Math.round(wx.data.current_weather.temperature)}°C`, description: getWeatherDescription(wx.data.current_weather.weathercode), location: loc.name });
      setShowWeatherSearch(false); setSearchCity("");
    } catch { alert("Failed to fetch weather"); }
    setWeatherLoading(false);
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/all-history`);
      setAllHistory(response.data || []);
    } catch { setAllHistory([]); }
    setHistoryLoading(false);
  };

  const deleteHistory = async (id) => {
    if (!window.confirm("Delete this conversation?")) return;
    try { await axios.delete(`${API_URL}/api/history/${id}`); loadHistory(); } catch {}
  };

  useEffect(() => { loadHistory(); }, []);
  useEffect(() => { if (showHistory) loadHistory(); }, [showHistory]);
  useEffect(() => {
    if (isListening || isSpeaking) {
      waveAnimRef.current = setInterval(() => setWaveValues(Array(30).fill(0).map(() => Math.random() * 40 + 4)), 120);
    } else { clearInterval(waveAnimRef.current); setWaveValues(Array(30).fill(4)); }
    return () => clearInterval(waveAnimRef.current);
  }, [isListening, isSpeaking]);
  useEffect(() => { if (activeFeature === "news") fetchNews(selectedCategory); }, [activeFeature, selectedCategory]);
  useEffect(() => { const t = setTimeout(() => preloadAllNews(), 2000); return () => clearTimeout(t); }, []);
  useEffect(() => { return () => clearInterval(cooldownRef.current); }, []);

  const speakText = (text) => {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = settings.speechLang; u.rate = settings.voiceRate; u.pitch = settings.voicePitch;
    u.onstart = () => setIsSpeaking(true); u.onend = () => setIsSpeaking(false);
    window.speechSynthesis.speak(u);
  };

  const askAI = async (question) => {
    if (!question.trim()) return;
    setAiAnswer("Thinking...");
    try {
      const res = await fetch(`${API_URL}/api/chat`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ message:question }) });
      const reader = res.body.getReader(), dec = new TextDecoder("utf-8"); let bot = "";
      while (true) { const { done, value } = await reader.read(); if (done) break; bot += dec.decode(value); setAiAnswer(bot); }
      await axios.post(`${API_URL}/api/save-history`, { type:"voice", question, answer:bot });
      loadHistory(); speakText(bot);
    } catch { setAiAnswer("Sorry, AI connection failed!"); }
  };

  const escapeHtml = t => t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");

  const highlightPythonSyntax = (code) => {
    const kw = ['def','class','import','from','return','if','else','elif','for','while','try','except','finally','with','as','pass','break','continue','lambda','True','False','None','and','or','not','is','in'];
    const bi = ['print','len','range','str','int','float','list','dict','set','tuple','bool','type','input','open','sum','min','max','sorted','enumerate','zip','map','filter'];
    let h = code;
    kw.forEach(k => { h = h.replace(new RegExp(`\\b(${k})\\b`, 'g'), `<span class="code-keyword">$1</span>`); });
    bi.forEach(b => { h = h.replace(new RegExp(`\\b(${b})\\b(?=\\()`, 'g'), `<span class="code-builtin">$1</span>`); });
    h = h.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, m => `<span class="code-string">${m}</span>`);
    h = h.replace(/(#.*$)/gm, m => `<span class="code-comment">${m}</span>`);
    h = h.replace(/\b(\d+)\b/g, m => `<span class="code-number">${m}</span>`);
    return h;
  };

  const formatCodeBlocks = (text) => {
    let f = text;
    f = f.replace(/```python\n([\s\S]*?)```/g, (_, c) => `<pre class="code-block python"><code>${highlightPythonSyntax(escapeHtml(c))}</code></pre>`);
    f = f.replace(/```(\w*)\n([\s\S]*?)```/g, (_, l, c) => `<pre class="code-block ${l||'code'}"><code>${escapeHtml(c)}</code></pre>`);
    f = f.replace(/`([^`]+)`/g, (_, c) => `<code class="inline-code">${escapeHtml(c)}</code>`);
    f = f.replace(/\n/g, '<br>');
    return f;
  };

  const askChatbot = async (message) => {
    if (!message.trim()) return;
    setChatMessages(p => [...p, { role:"user", content:message }]);
    setChatInput(""); setLoading(true); setIsTyping(true);
    const isCode = ["code","python","javascript","java","html","css","example","function"].some(w => message.toLowerCase().includes(w));
    try {
      const res = await fetch(`${API_URL}/api/chat`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ message }) });
      const reader = res.body.getReader(), dec = new TextDecoder("utf-8"); let full = "";
      setChatMessages(p => [...p, { role:"assistant", content:"", isStreaming:true, isCode }]);
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        full += dec.decode(value);
        let disp = full;
        if (isCode && (full.includes("```") || full.includes("def ") || full.includes("class "))) disp = formatCodeBlocks(full);
        setChatMessages(p => { const m = [...p]; const l = m.length-1; if (m[l]?.role==="assistant") m[l] = {...m[l], content:disp, isStreaming:true, isCode}; return m; });
      }
      let final = full;
      if (isCode && (full.includes("```") || full.includes("def ") || full.includes("class "))) final = formatCodeBlocks(full);
      setChatMessages(p => { const m = [...p]; const l = m.length-1; if (m[l]?.role==="assistant") m[l] = {...m[l], content:final, isStreaming:false, isCode}; return m; });
      await axios.post(`${API_URL}/api/save-history`, { type:"chat", question:message, answer:full });
      loadHistory();
    } catch { setChatMessages(p => [...p, { role:"assistant", content:"Sorry, AI connection failed!", isError:true }]); }
    setLoading(false); setIsTyping(false);
  };

  const copyToClipboard = (text) => {
    const d = document.createElement('div'); d.innerHTML = text;
    navigator.clipboard.writeText(d.textContent || d.innerText);
    alert("Copied!");
  };

  const downloadChat = () => {
    const t = chatMessages.map(m => { const d=document.createElement('div'); d.innerHTML=m.content; return `${m.role==="user"?"You":"AI"}: ${d.textContent||d.innerText}`; }).join("\n\n");
    const blob = new Blob([t], { type:"text/plain" });
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = `chat_${new Date().toISOString().slice(0,19)}.txt`; a.click(); URL.revokeObjectURL(url);
  };

  const createRecognition = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { alert("Speech Recognition not supported"); return null; }
    const r = new SR();
    r.continuous = true; r.interimResults = true; r.lang = settings.speechLang;
    r.onresult = (e) => {
      let interim = "", final = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const txt = e.results[i][0].transcript;
        e.results[i].isFinal ? (final += txt) : (interim += txt);
      }
      setCurrentText(interim);
      if (final.trim()) { setFinalSentences(p => [...p, final]); askAI(final); }
    };
    r.onerror = e => console.log("Speech Error:", e);
    return r;
  };

  const handleMic = async () => {
    if (loading) return; setLoading(true);
    try {
      if (isListening) {
        recognitionRef.current?.stop(); recognitionRef.current = null;
        setIsListening(false); setCurrentText(""); setFinalSentences([]); setAiAnswer(""); window.speechSynthesis.cancel();
      } else {
        const rec = createRecognition(); if (!rec) return;
        recognitionRef.current = rec; rec.start(); setIsListening(true); setAiAnswer("");
      }
    } catch (err) { console.error(err); }
    setTimeout(() => setLoading(false), 300);
  };

  const handleImageUpload = (e) => {
    const f = e.target.files[0];
    if (f) {
      setSelectedImage(f);
      const r = new FileReader(); r.onloadend = () => setImagePreview(r.result); r.readAsDataURL(f);
      setVisionAnswer(""); setShowCamera(false); setCameraPhoto(null);
    }
  };

  const capturePhoto = () => {
    const src = webcamRef.current.getScreenshot();
    if (src) {
      fetch(src).then(r => r.blob()).then(blob => {
        const f = new File([blob], "camera-photo.jpg", { type:"image/jpeg" });
        setSelectedImage(f); setImagePreview(src); setVisionAnswer(""); setShowCamera(false); setCameraPhoto(src);
      });
    }
  };

  const startCamera = () => { setShowCamera(true); setSelectedImage(null); setImagePreview(null); setVisionAnswer(""); setVisionQuestion(""); };
  const closeCamera = () => setShowCamera(false);

  // ── VISION SUBMIT with cooldown handling ──────────────────────────────────
  const handleVisionSubmit = async () => {
    if (!selectedImage) { alert("Please capture or select an image first!"); return; }
    if (cooldownActive) return; // blocked during cooldown

    setVisionLoading(true);
    setVisionAnswer("🔍 Analyzing image...");

    const fd = new FormData();
    fd.append("image", selectedImage);
    fd.append("question", visionQuestion || "Describe this image in detail");

    try {
      const res = await axios.post(`${API_URL}/api/vision`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
        validateStatus: () => true, // don't throw on 429
      });

      if (res.status === 429 || res.data?.cooldown) {
        // Server says still in cooldown
        const secs = res.data?.remainingSeconds || COOLDOWN_SECS;
        startCooldown(secs);
        setVisionAnswer("");
        setVisionLoading(false);
        return;
      }

      const answer = res.data?.response || "Could not analyze image.";
      setVisionAnswer(answer);

      // Start cooldown after successful analysis
      startCooldown(COOLDOWN_SECS);

      await axios.post(`${API_URL}/api/save-history`, {
        type: "vision",
        question: visionQuestion || "Analyzed image",
        answer,
      });
      loadHistory();
      speakText(answer);
    } catch {
      setVisionAnswer("Failed to analyze image. Please try again.");
    }
    setVisionLoading(false);
  };

  const clearImage = () => {
    setSelectedImage(null); setImagePreview(null); setVisionAnswer(""); setVisionQuestion(""); setCameraPhoto(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const displayedFeature = activeFeature || hoveredFeature;
  const info = displayedFeature ? FEATURE_INFO[displayedFeature] : null;

  const SkeletonLoader = () => (
    <div style={S.skeletonContainer}>
      {[1,2,3,4,5].map(i => (
        <div key={i} style={S.skeletonCard}>
          <div style={S.skeletonHeader}><div style={S.skeletonSource}/><div style={S.skeletonTime}/></div>
          <div style={S.skeletonTitle}/><div style={S.skeletonLine}/><div style={S.skeletonLineShort}/>
        </div>
      ))}
    </div>
  );

  // ── COOLDOWN DISPLAY COMPONENT ─────────────────────────────────────────────
  const CooldownBanner = () => (
    <div style={S.cooldownBanner}>
      <div style={S.cooldownTop}>
        <span style={S.cooldownIcon}>⏳</span>
        <span style={S.cooldownLabel}>NEXT ANALYSIS AVAILABLE IN</span>
        <span style={S.cooldownTime}>
          {cooldownMins}:{cooldownSecsRem.toString().padStart(2, '0')}
        </span>
      </div>
      {/* Progress bar */}
      <div style={S.cooldownBarBg}>
        <div style={{ ...S.cooldownBarFill, width: `${cooldownPercent}%` }} />
      </div>
      <div style={S.cooldownSub}>
        AI vision models need a short rest between calls. You can upload a new image now and analyze when ready.
      </div>
    </div>
  );

  return (
    <div style={S.root}>
      {isTransitioning && <div style={S.fadeOverlay}/>}
      <video autoPlay loop muted playsInline style={S.backgroundVideo}>
        <source src={backgroundVideoUrl} type="video/mp4"/>
      </video>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        *{margin:0;padding:0;box-sizing:border-box;}
        body{margin:0;padding:0;overflow:hidden;}
        @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.1)}}
        @keyframes glow{0%,100%{box-shadow:0 0 20px rgba(74,255,158,.3)}50%{box-shadow:0 0 40px rgba(74,255,158,.6)}}
        @keyframes slideIn{from{transform:translateX(-100%)}to{transform:translateX(0)}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
        @keyframes shimmer{0%{background-position:-200px 0}100%{background-position:200px 0}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
        @keyframes btnBorderBlink{0%,100%{border-color:rgba(74,255,158,0.25);box-shadow:0 0 6px rgba(74,255,158,0.08);}50%{border-color:rgba(74,255,158,0.85);box-shadow:0 0 18px rgba(74,255,158,0.35),0 0 32px rgba(74,255,158,0.15);}}
        @keyframes liteFloat{0%,100%{transform:translateY(0px);}50%{transform:translateY(-5px);}}
        @keyframes overlayFade{0%{opacity:0;}30%{opacity:1;}70%{opacity:1;}100%{opacity:0;}}
        @keyframes slideInR{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}
        @keyframes modalFadeIn{from{opacity:0;transform:translate(-50%,-48%)}to{opacity:1;transform:translate(-50%,-50%)}}
        @keyframes cooldownPulse{0%,100%{opacity:1;}50%{opacity:0.6;}}
        @keyframes barFlow{0%{background-position:0% 50%;}100%{background-position:100% 50%;}}
        .feat-btn{background:rgba(0,0,0,0.45);border:1.5px solid rgba(74,255,158,0.25);border-radius:16px;padding:18px;text-align:center;transition:background .25s,transform .2s;cursor:pointer;color:#fff;animation:btnBorderBlink 2s ease-in-out infinite,liteFloat 4s ease-in-out infinite;}
        .feat-btn:hover{background:rgba(74,255,158,0.12);transform:scale(1.04) translateY(-4px);}
        .nav-btn-hover:hover{background:rgba(74,255,158,0.15)!important;border-color:rgba(74,255,158,0.5)!important;}
        .sett-tab-hover:hover{background:rgba(255,255,255,0.08)!important;}
        .code-block{background:rgba(0,0,0,0.4);border-radius:8px;padding:12px;margin:8px 0;overflow-x:auto;font-family:'Fira Code',monospace;font-size:12px;border-left:3px solid #4aff9e;}
        .code-block code{font-family:'Fira Code',monospace;white-space:pre-wrap;word-break:break-word;}
        .inline-code{background:rgba(0,0,0,0.3);padding:2px 6px;border-radius:4px;font-family:'Fira Code',monospace;font-size:12px;color:#4aff9e;}
        .code-keyword{color:#ff79c6;font-weight:bold;}
        .code-builtin{color:#8be9fd;}
        .code-string{color:#f1fa8c;}
        .code-comment{color:#6272a4;font-style:italic;}
        .code-number{color:#bd93f9;}
        input[type=range]{accent-color:#4aff9e;cursor:pointer;}
        ::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:transparent;}::-webkit-scrollbar-thumb{background:rgba(74,255,158,0.3);border-radius:4px;}
        select option { background: #1a1a2e !important; color: #ffffff !important; }
        select { color-scheme: dark; }
        .analyze-btn-ready{background:rgba(74,255,158,0.2);border:1px solid rgba(74,255,158,0.3);color:#4aff9e;cursor:pointer;}
        .analyze-btn-ready:hover{background:rgba(74,255,158,0.35);}
        .analyze-btn-cooldown{background:rgba(255,180,0,0.12);border:1px solid rgba(255,180,0,0.3);color:#ffb400;cursor:not-allowed;opacity:0.7;}
      `}</style>

      <div style={S.topLeftBrand}>
        <div>
          <div style={S.brandMain}>AURA</div>
          <div style={S.brandSub}>ADVANCED ROBOTIC ASSISTANT</div>
        </div>
      </div>

      <div style={S.topRight}>
        {info ? (
          <div style={{...S.systemChip, borderColor:`${info.color}55`}}>
            <span style={{fontSize:20}}>{info.icon}</span>
            <div style={{display:"flex",flexDirection:"column",gap:1}}>
              <span style={{fontSize:10,fontFamily:"'Orbitron',sans-serif",color:info.color,letterSpacing:1.5,fontWeight:700}}>{info.label} — {info.sub}</span>
              <span style={{fontSize:8,color:"rgba(255,255,255,0.5)",letterSpacing:1}}>SYSTEM ACTIVATION</span>
            </div>
            <div style={{width:7,height:7,borderRadius:"50%",background:info.color,boxShadow:`0 0 8px ${info.color}`,animation:"pulse 1.5s infinite"}}/>
          </div>
        ) : (
          <div style={S.statusChip}>
            <div style={S.statusDot}/>
            <span style={{fontFamily:"'Orbitron',sans-serif",fontSize:10,color:"#4aff9e",letterSpacing:2}}>SYSTEM READY</span>
          </div>
        )}
      </div>

      {info && (
        <div style={S.infoPanel}>
          <div style={{...S.infoPanelHeader, borderColor:`${info.color}44`}}>
            <span style={{fontSize:16}}>{info.icon}</span>
            <span style={{fontSize:11,fontFamily:"'Orbitron',sans-serif",color:info.color,letterSpacing:1,fontWeight:700}}>{info.label}</span>
          </div>
          <table style={S.infoTable}>
            <tbody>
              {info.rows.map(([k,v],i) => (
                <tr key={i} style={{background:i%2===0?"rgba(255,255,255,0.03)":"transparent"}}>
                  <td style={{...S.infoTd,color:info.color,fontFamily:"'Orbitron',sans-serif",fontSize:8,letterSpacing:1,whiteSpace:"nowrap",paddingRight:10}}>{k}</td>
                  <td style={{...S.infoTd,color:"rgba(255,255,255,0.85)",fontSize:11}}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showHistory && (
        <>
          <div style={S.historyBackdrop} onClick={() => setShowHistory(false)}/>
          <div style={S.historySidebar}>
            <div style={S.historyHeader}>
              <div><span>📜 HISTORY</span><div style={{fontSize:10,color:"rgba(74,255,158,0.7)",marginTop:3}}>Temporary (max 50)</div></div>
              <button onClick={() => setShowHistory(false)} style={S.closeHistoryBtn}>✕</button>
            </div>
            <div style={S.historyListSidebar}>
              {historyLoading ? (<div style={S.noHistory}>Loading...</div>) : allHistory.length === 0 ? (
                <div style={S.noHistory}><div style={{fontSize:32,marginBottom:10}}>📭</div>No conversations yet</div>
              ) : (
                allHistory.map(item => (
                  <div key={item.id} style={S.historyItemSidebar}>
                    <div style={S.historyContent}>
                      <div style={S.historyTypeIcon}>{item.type==="voice"?"🎤 Voice":item.type==="news"?"📰 News":item.type==="vision"?"📷 Vision":"💬 Chat"}</div>
                      <div style={S.historyQuestion}>{item.question?.substring(0,55)}...</div>
                      {item.answer && <div style={S.historyAnswer}>{item.answer?.substring(0,65)}...</div>}
                      <div style={S.historyTimestamp}>{item.formattedTime || new Date(item.timestamp).toLocaleString()}</div>
                    </div>
                    <button onClick={() => deleteHistory(item.id)} style={S.deleteBtnSidebar}>🗑️</button>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}

      {showWeatherSearch && (
        <>
          <div style={S.modalBackdrop} onClick={() => setShowWeatherSearch(false)}/>
          <div style={S.weatherModal}>
            <div style={S.weatherModalHeader}><span>🌤️ Search Weather</span><button onClick={() => setShowWeatherSearch(false)} style={S.modalCloseBtn}>✕</button></div>
            <div style={S.weatherModalBody}>
              <input type="text" value={searchCity} onChange={e=>setSearchCity(e.target.value)} onKeyPress={e=>e.key==="Enter"&&fetchWeather(searchCity)} placeholder="Enter city name" style={S.weatherSearchInput} autoFocus/>
              <button onClick={() => fetchWeather(searchCity)} style={S.weatherSearchBtn} disabled={weatherLoading}>{weatherLoading?"⏳ Searching...":"🔍 Get Weather"}</button>
              <div style={S.weatherExample}>Examples: Chennai, Coimbatore, Madurai, Ooty</div>
            </div>
          </div>
        </>
      )}

      {showSettings && (
        <>
          <div style={S.modalBackdrop} onClick={() => setShowSettings(false)}/>
          <div style={S.settingsModal}>
            <div style={S.settingsSidebar}>
              <div style={S.settingsSidebarTitle}><span style={{fontFamily:"'Orbitron',sans-serif",fontSize:12,color:"#4aff9e",letterSpacing:1}}>AURA</span></div>
              <button onClick={() => setShowSettings(false)} style={S.settingsCloseBtn}>✕</button>
              {[{key:'general',label:'General',icon:'⚙️'},{key:'notifications',label:'Notifications',icon:'🔔'},{key:'data',label:'Data Controls',icon:'🗄️'}].map(({key,label,icon}) => (
                <button key={key} className="sett-tab-hover" onClick={() => setSettingsTab(key)} style={{...S.settingsTabBtn,...(settingsTab===key?S.settingsTabActive:{})}}>
                  <span style={{fontSize:15}}>{icon}</span><span>{label}</span>
                </button>
              ))}
            </div>
            <div style={S.settingsContent}>
              <button onClick={() => setShowSettings(false)} style={S.settingsContentClose}>✕</button>
              {settingsTab === 'general' && (
                <>
                  <h2 style={S.settingsTitle}>General</h2>
                  <div style={S.settingsDivider}/>
                  {[
                    { label:"Appearance", desc:"Choose your display theme", control:(<select value={settings.appearance} onChange={e=>updateSettings('appearance',e.target.value)} style={S.settingSelect}><option value="dark">Dark</option><option value="system">System</option></select>) },
                    { label:"Spoken Language", desc:"Language for speech recognition", control:(<select value={settings.speechLang} onChange={e=>updateSettings('speechLang',e.target.value)} style={{...S.settingSelect,background:"#0d1117",color:"#fff",minWidth:160}}><option value="en-US">English (US)</option><option value="en-GB">English (UK)</option><option value="ta-IN">Tamil (India)</option><option value="hi-IN">Hindi (India)</option><option value="te-IN">Telugu (India)</option><option value="ml-IN">Malayalam (India)</option><option value="kn-IN">Kannada (India)</option><option value="bn-IN">Bengali (India)</option></select>) },
                    { label:"Voice Speed", desc:`Playback speed: ${settings.voiceRate}x`, control:(<div style={{display:"flex",alignItems:"center",gap:10}}><input type="range" min="0.5" max="2" step="0.1" value={settings.voiceRate} onChange={e=>updateSettings('voiceRate',parseFloat(e.target.value))} style={{width:120}}/><span style={{color:"#4aff9e",fontSize:12,minWidth:30}}>{settings.voiceRate}x</span></div>) },
                    { label:"Voice Pitch", desc:`Pitch level: ${settings.voicePitch}`, control:(<div style={{display:"flex",alignItems:"center",gap:10}}><input type="range" min="0.5" max="2" step="0.1" value={settings.voicePitch} onChange={e=>updateSettings('voicePitch',parseFloat(e.target.value))} style={{width:120}}/><span style={{color:"#4aff9e",fontSize:12,minWidth:30}}>{settings.voicePitch}</span></div>) },
                  ].map(({label,desc,control},i) => (
                    <div key={i} style={S.settingRow}><div><div style={S.settingLabel}>{label}</div><div style={S.settingDesc}>{desc}</div></div>{control}</div>
                  ))}
                </>
              )}
              {settingsTab === 'notifications' && (
                <>
                  <h2 style={S.settingsTitle}>Notifications</h2>
                  <div style={S.settingsDivider}/>
                  {[{label:"AI Response Sound",desc:"Play a sound when AI responds",key:"notifSound"},{label:"Voice Feedback",desc:"Speak AI responses aloud automatically",key:"autoSpeak"}].map(({label,desc,key}) => (
                    <div key={key} style={S.settingRow}>
                      <div><div style={S.settingLabel}>{label}</div><div style={S.settingDesc}>{desc}</div></div>
                      <div style={{...S.toggle,background:settings[key]!==false?"#4aff9e":"rgba(255,255,255,0.2)"}} onClick={() => updateSettings(key,settings[key]===false)}>
                        <div style={{...S.toggleKnob,transform:settings[key]!==false?"translateX(20px)":"translateX(0)"}}/>
                      </div>
                    </div>
                  ))}
                </>
              )}
              {settingsTab === 'data' && (
                <>
                  <h2 style={S.settingsTitle}>Data Controls</h2>
                  <div style={S.settingsDivider}/>
                  <div style={S.settingRow}><div><div style={S.settingLabel}>Conversation History</div><div style={S.settingDesc}>Delete all conversations</div></div><button onClick={clearAllHistory} style={S.dangerBtn}>Clear History</button></div>
                  <div style={S.settingRow}><div><div style={S.settingLabel}>Reset Settings</div><div style={S.settingDesc}>Restore all settings to default values</div></div><button onClick={() => { localStorage.removeItem('aura_settings'); setSettings(DEFAULT_SETTINGS); }} style={S.dangerBtn}>Reset</button></div>
                  <div style={{...S.settingRow,flexDirection:"column",alignItems:"flex-start",gap:8}}><div style={S.settingLabel}>History</div><div style={{...S.settingDesc,fontSize:11}}>Temporary storage (max 50 items) | Records: {allHistory.length}</div></div>
                </>
              )}
            </div>
          </div>
        </>
      )}

      <div style={S.main}>
        <div style={S.hero}>
          {!activeFeature && !isTransitioning && (
            <div style={{...S.featureGrid, position:"absolute", left:60, top:"50%", transform:"translateY(-50%)", marginTop:0}}>
              {[{k:"voice",label:"AI VOICE",val:"ASSISTANT",ic:"🎙️"},{k:"chatbot",label:"CHATBOT",val:"ASSISTANT",ic:"💬"},{k:"lense",label:"AI LENSE",val:"VISION",ic:"📷"},{k:"news",label:"AI NEWS",val:"READER",ic:"📰"}].map(({k,label,val,ic}) => (
                <button key={k} className="feat-btn" onClick={() => handleFeatureClick(k)} onMouseEnter={() => setHoveredFeature(k)} onMouseLeave={() => setHoveredFeature(null)} style={{animationDelay:k==="chatbot"?"0.3s":k==="lense"?"0.6s":k==="news"?"0.9s":"0s"}}>
                  <div style={{fontSize:22,marginBottom:4}}>{ic}</div>
                  <div style={S.featureLabel}>{label}</div>
                  <div style={S.featureValue}>{val}</div>
                </button>
              ))}
            </div>
          )}

          {activeFeature === "voice" && !isTransitioning && (
            <div style={S.voiceSection}>
              <div style={S.waveContainer}>{waveValues.map((h,i) => (<div key={i} style={{...S.waveBar, height:`${Math.max(4,h)}px`, backgroundColor:isSpeaking?"#4aff9e":isListening?"#3a86ff":"rgba(255,255,255,0.5)"}}/>))}</div>
              <button onClick={handleMic} disabled={loading} style={{...S.micButton,...(isListening?S.micActive:{}),...(loading?S.micLoading:{})}}>{loading?"⟳":isListening?"⏹️":"🎤"}</button>
              <div style={S.micLabel}>{isListening?"🎙️ LISTENING...":isSpeaking?"🔊 SPEAKING...":"🎤 TAP MIC TO SPEAK"}</div>
              {(aiAnswer||isListening||isSpeaking)&&(
                <div style={S.aiResponseContainer}>
                  <div style={S.aiResponseHeader}><span style={S.aiResponseIcon}>✨</span><span style={S.aiResponseTitle}>AURA AI</span></div>
                  <div style={S.aiResponseContent}>
                    {aiAnswer?<div style={S.aiResponseText}>{aiAnswer}</div>:isListening?<div style={S.listeningIndicator}><span style={S.listeningText}>🎙️ Listening...</span></div>:<div style={S.listeningIndicator}><span style={S.listeningText}>🔊 Speaking...</span></div>}
                  </div>
                </div>
              )}
              <button onClick={handleBackToMenu} style={S.backToMenuBtn}>← Back to Menu</button>
            </div>
          )}

          {activeFeature === "chatbot" && !isTransitioning && (
            <div style={S.chatbotSection}>
              <div style={S.chatHeader}><span style={S.chatHeaderTitle}>💬 AI CHAT</span>{chatMessages.length>0&&<button onClick={downloadChat} style={S.downloadChatBtn}>📥 Download</button>}</div>
              <div style={S.chatMessages}>
                {chatMessages.length===0?(<div style={S.emptyChat}>Start a conversation with AI...</div>):(
                  chatMessages.map((msg,idx)=>(
                    <div key={idx} style={{...S.chatBubble,...(msg.role==="user"?S.userBubble:S.aiBubble)}}>
                      <div style={S.chatBubbleHeader}><strong>{msg.role==="user"?"You":"AI"}</strong>{msg.role==="assistant"&&msg.content&&!msg.isError&&<button onClick={()=>copyToClipboard(msg.content)} style={S.copyBtn}>📋</button>}</div>
                      <div style={S.chatBubbleContent} dangerouslySetInnerHTML={{__html:msg.content}}/>
                      {msg.isStreaming&&<span style={S.typingCursor}>|</span>}
                    </div>
                  ))
                )}
                {loading&&!chatMessages.some(m=>m.isStreaming)&&<div style={S.typingIndicator}><span>●</span><span>●</span><span>●</span></div>}
              </div>
              <div style={S.chatInputContainer}>
                <input type="text" value={chatInput} onChange={e=>setChatInput(e.target.value)} onKeyPress={e=>e.key==="Enter"&&askChatbot(chatInput)} placeholder="Type your message..." style={S.chatInput}/>
                <button onClick={()=>askChatbot(chatInput)} style={S.sendBtn}>Send</button>
              </div>
              <button onClick={handleBackToMenu} style={S.backToMenuBtn}>← Back to Menu</button>
            </div>
          )}

          {activeFeature === "lense" && !isTransitioning && (
            <div style={S.lenseSection}>
              <div style={S.lenseHeader}>
                <span style={S.lenseIcon}>📷</span>
                <span style={S.lenseTitle}>AI LENSE</span>
                <span style={S.lenseSubtitle}>Capture or upload an image and ask anything</span>
              </div>

              {/* Action buttons — always visible */}
              <div style={S.actionButtons}>
                <button onClick={startCamera} style={S.cameraBtn}>📷 Camera</button>
                <label htmlFor="imageUpload" style={S.galleryBtn}>📁 Gallery</label>
                <input type="file" ref={fileInputRef} accept="image/*" onChange={handleImageUpload} style={S.fileInput} id="imageUpload"/>
              </div>

              {/* Camera view */}
              {showCamera && (
                <div style={S.cameraContainer}>
                  <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" videoConstraints={{facingMode:"environment"}} style={S.cameraPreview}/>
                  <div style={S.cameraButtons}>
                    <button onClick={capturePhoto} style={S.captureBtn}>📸 Capture</button>
                    <button onClick={closeCamera} style={S.closeCameraBtn}>✕ Close</button>
                  </div>
                </div>
              )}

              {/* Image preview */}
              {imagePreview && !showCamera && (
                <div style={S.imagePreviewContainer}>
                  <img src={imagePreview} alt="Preview" style={S.imagePreview}/>
                  <button onClick={clearImage} style={S.clearImageBtn}>✕ Clear</button>
                </div>
              )}

              {/* Question + Analyze button */}
              {imagePreview && !showCamera && (
                <div style={S.questionArea}>
                  <input
                    type="text"
                    value={visionQuestion}
                    onChange={e => setVisionQuestion(e.target.value)}
                    onKeyPress={e => e.key === "Enter" && !cooldownActive && handleVisionSubmit()}
                    placeholder={cooldownActive ? `⏳ Wait ${cooldownMins}:${cooldownSecsRem.toString().padStart(2,'0')} to analyze again...` : "Ask about this image..."}
                    style={{...S.lenseInput, opacity: cooldownActive ? 0.6 : 1}}
                    disabled={cooldownActive}
                  />
                  <button
                    onClick={handleVisionSubmit}
                    disabled={visionLoading || cooldownActive}
                    className={cooldownActive ? "analyze-btn-cooldown" : "analyze-btn-ready"}
                    style={{...S.analyzeBtn, ...(cooldownActive ? {background:"rgba(255,180,0,0.12)",borderColor:"rgba(255,180,0,0.3)",color:"#ffb400",cursor:"not-allowed"} : {})}}
                  >
                    {visionLoading ? "⏳ Analyzing..." : cooldownActive ? `⏳ ${cooldownMins}:${cooldownSecsRem.toString().padStart(2,'0')}` : "🔍 Analyze"}
                  </button>
                </div>
              )}

              {/* ── COOLDOWN BANNER ─────────────────────────────────────────── */}
              {cooldownActive && <CooldownBanner />}

              {/* Vision result */}
              {(visionAnswer || visionLoading) && !showCamera && (
                <div style={S.visionResponseContainer}>
                  <div style={S.visionResponseHeader}><span>✨</span><span style={S.visionResponseTitle}>AI ANALYSIS</span></div>
                  <div style={S.visionResponseContent}>
                    {visionLoading
                      ? <div style={S.loadingDots}><span>●</span><span>●</span><span>●</span></div>
                      : <div style={S.visionResponseText}>{visionAnswer}</div>
                    }
                  </div>
                </div>
              )}

              {!imagePreview && !showCamera && (
                <div style={S.noImageMessage}>
                  <span>📷 Tap Camera to take a photo</span>
                  <span>📁 Tap Gallery to upload</span>
                </div>
              )}

              <button onClick={handleBackToMenu} style={S.backToMenuBtn}>← Back to Menu</button>
            </div>
          )}

          {activeFeature === "news" && !isTransitioning && (
            <div style={{...S.newsSection,...(info?{marginRight:300,maxWidth:680}:{})}}>
              <div style={S.newsHeader}><span style={S.newsIcon}>📰</span><span style={S.newsTitle}>LATEST NEWS</span><span style={S.newsCount}>{news.length} articles</span><button onClick={()=>fetchNews(selectedCategory)} style={S.newsRefreshBtn} disabled={newsLoading}>{newsLoading?"⟳":"🔄"}</button>{readingNews&&<button onClick={stopReading} style={S.newsStopBtn}>⏹️ Stop</button>}</div>
              <div style={S.categoryContainer}>{categories.map(cat=>(<button key={cat.id} onClick={()=>setSelectedCategory(cat.id)} style={{...S.categoryBtn,...(selectedCategory===cat.id?S.categoryActive:{})}}>{cat.emoji} {cat.name}</button>))}</div>
              <div style={S.newsList}>
                {newsLoading?<SkeletonLoader/>:news.length===0?(<div style={S.noNews}><span>📭</span><span>No news available</span><button onClick={()=>fetchNews(selectedCategory)} style={S.retryBtn}>Try Again</button></div>):(
                  news.map((item,idx)=>(
                    <div key={idx} style={S.newsCard}>
                      <div style={S.newsCardHeader}><span style={S.newsSource}>{item.source?.name||"AURA News"}</span><span style={S.newsTime}>{formatDate(item.publishedAt)}</span></div>
                      <div style={S.newsTitleText}>{item.title}</div>
                      <div style={S.newsDescription}>{item.description}</div>
                      <div style={S.newsActions}><button onClick={()=>speakNews(item.title,item.description)} style={S.newsListenBtn}>🔊 Listen</button></div>
                    </div>
                  ))
                )}
              </div>
              <div style={S.newsFooter}><span>📡 {news.length}+ latest news • Real-time</span></div>
              <button onClick={handleBackToMenu} style={S.backToMenuBtn}>← Back to Menu</button>
            </div>
          )}
        </div>
      </div>

      {!activeFeature && !isTransitioning && (
        <div style={S.weatherCardFixed}>
          <span style={S.temp}>{weatherData.temp}</span>
          <span style={S.weatherDesc}>{weatherData.description}</span>
          <span style={S.locationText}>{weatherData.location}</span>
          <button onClick={() => setShowWeatherSearch(true)} style={S.searchIconBtn}>🔍</button>
        </div>
      )}

      <div style={S.bottomNav}>
        <button className="nav-btn-hover" onClick={() => setShowHistory(!showHistory)} style={S.navBtn}>📋 History</button>
        <button className="nav-btn-hover" onClick={() => { setShowSettings(true); setSettingsTab('general'); }} style={S.navBtn}>⚙️ Settings</button>
      </div>

      <div style={S.bottomLeftDisclaimer}>AURA is AI and can make mistakes</div>
    </div>
  );
};

const S = {
  root: { position:"relative",display:"flex",height:"100vh",width:"100vw",overflow:"hidden",fontFamily:"'Inter',sans-serif" },
  fadeOverlay: { position:"fixed",top:0,left:0,width:"100%",height:"100%",backgroundColor:"rgba(0,0,0,0.35)",zIndex:9999,pointerEvents:"none",animation:"overlayFade 1.2s ease forwards" },
  backgroundVideo: { position:"absolute",top:0,left:0,width:"100%",height:"100%",objectFit:"cover",objectPosition:"center",zIndex:0,pointerEvents:"none" },
  topLeftBrand: { position:"fixed",top:20,left:24,zIndex:100,display:"flex",alignItems:"center",gap:10,background:"rgba(0,0,0,0.45)",backdropFilter:"blur(10px)",borderRadius:14,padding:"8px 18px",border:"1px solid rgba(74,255,158,0.3)" },
  brandMain: { fontSize:22,fontWeight:800,fontFamily:"'Orbitron',sans-serif",color:"#4aff9e",letterSpacing:2,lineHeight:1.2 },
  brandSub: { fontSize:7,fontWeight:500,color:"rgba(255,255,255,0.7)",letterSpacing:1.5,fontFamily:"'Orbitron',sans-serif" },
  bottomLeftDisclaimer: { position:"fixed",bottom:20,left:24,zIndex:100,background:"rgba(0,0,0,0.4)",backdropFilter:"blur(8px)",padding:"6px 14px",borderRadius:20,fontSize:10,color:"rgba(255,255,255,0.55)",letterSpacing:0.5,border:"1px solid rgba(255,255,255,0.1)" },
  topRight: { position:"fixed",top:18,right:24,zIndex:100,display:"flex",flexDirection:"column",alignItems:"flex-end",gap:8 },
  statusChip: { display:"flex",alignItems:"center",gap:8 },
  statusDot: { width:7,height:7,borderRadius:"50%",background:"#4aff9e",animation:"pulse 2s infinite" },
  systemChip: { display:"flex",alignItems:"center",gap:10,background:"rgba(0,0,0,0.55)",border:"1px solid rgba(74,255,158,0.3)",borderRadius:12,padding:"8px 14px",backdropFilter:"blur(12px)" },
  infoPanel: { position:"fixed",top:72,right:24,zIndex:99,width:270,background:"rgba(0,0,0,0.6)",border:"1px solid rgba(74,255,158,0.18)",borderRadius:12,backdropFilter:"blur(16px)",overflow:"hidden",animation:"slideInR .3s ease" },
  infoPanelHeader: { display:"flex",alignItems:"center",gap:10,padding:"8px 12px",background:"rgba(0,0,0,0.3)",borderBottom:"1px solid rgba(74,255,158,0.15)" },
  infoTable: { width:"100%",borderCollapse:"collapse" },
  infoTd: { padding:"5px 12px",verticalAlign:"middle" },
  historyBackdrop: { position:"fixed",top:0,left:0,right:0,bottom:0,background:"rgba(0,0,0,0.2)",zIndex:200 },
  historySidebar: { position:"fixed",top:0,left:0,width:380,height:"100vh",background:"rgba(10,10,10,0.3)",backdropFilter:"blur(25px)",borderRight:"1px solid rgba(255,255,255,0.1)",zIndex:201,animation:"slideIn .3s ease",display:"flex",flexDirection:"column" },
  historyHeader: { display:"flex",justifyContent:"space-between",alignItems:"center",padding:20,borderBottom:"1px solid rgba(255,255,255,0.1)",color:"#fff",fontWeight:600,background:"rgba(0,0,0,0.2)" },
  closeHistoryBtn: { background:"rgba(255,255,255,0.15)",border:"none",color:"#fff",fontSize:18,cursor:"pointer",padding:"4px 8px",borderRadius:4 },
  historyListSidebar: { flex:1,overflowY:"auto",padding:16,display:"flex",flexDirection:"column",gap:12 },
  historyItemSidebar: { padding:12,background:"rgba(255,255,255,0.08)",borderRadius:8,display:"flex",justifyContent:"space-between",gap:10,border:"1px solid rgba(255,255,255,0.1)",backdropFilter:"blur(5px)" },
  historyContent: { flex:1 },
  historyTypeIcon: { fontSize:11,color:"#4aff9e",marginBottom:6,fontWeight:600 },
  historyQuestion: { fontSize:12,color:"#fff",marginBottom:6,lineHeight:1.4 },
  historyAnswer: { fontSize:11,color:"rgba(255,255,255,0.7)",marginBottom:6,lineHeight:1.4 },
  historyTimestamp: { fontSize:9,color:"rgba(255,255,255,0.4)",letterSpacing:.5 },
  deleteBtnSidebar: { background:"rgba(255,255,255,0.1)",border:"none",color:"#fff",cursor:"pointer",padding:"6px 10px",borderRadius:6,fontSize:12,height:32 },
  noHistory: { textAlign:"center",color:"rgba(255,255,255,0.5)",padding:40,fontSize:13 },
  modalBackdrop: { position:"fixed",top:0,left:0,right:0,bottom:0,background:"rgba(0,0,0,0.6)",zIndex:300,backdropFilter:"blur(6px)" },
  weatherModal: { position:"fixed",top:"50%",left:"50%",transform:"translate(-50%,-50%)",width:400,background:"rgba(20,20,30,0.95)",backdropFilter:"blur(20px)",borderRadius:16,border:"1px solid rgba(74,255,158,0.3)",zIndex:301,animation:"modalFadeIn .2s ease" },
  weatherModalHeader: { display:"flex",justifyContent:"space-between",alignItems:"center",padding:"16px 20px",borderBottom:"1px solid rgba(255,255,255,0.1)",color:"#fff",fontWeight:600 },
  modalCloseBtn: { background:"rgba(255,255,255,0.1)",border:"none",color:"#fff",fontSize:16,cursor:"pointer",padding:"4px 10px",borderRadius:6 },
  weatherModalBody: { padding:20,display:"flex",flexDirection:"column",gap:15 },
  weatherSearchInput: { padding:12,background:"rgba(255,255,255,0.1)",border:"1px solid rgba(255,255,255,0.2)",borderRadius:8,color:"#fff",fontSize:14,outline:"none" },
  weatherSearchBtn: { padding:12,background:"rgba(74,255,158,0.2)",border:"1px solid rgba(74,255,158,0.3)",borderRadius:8,color:"#4aff9e",cursor:"pointer",fontSize:14,fontWeight:500 },
  weatherExample: { fontSize:11,color:"rgba(255,255,255,0.5)",textAlign:"center" },
  settingsModal: { position:"fixed",top:"50%",left:"50%",transform:"translate(-50%,-50%)",width:760,height:520,background:"rgba(12,14,20,0.97)",backdropFilter:"blur(24px)",borderRadius:20,border:"1px solid rgba(255,255,255,0.1)",zIndex:301,display:"flex",overflow:"hidden",animation:"modalFadeIn .25s ease" },
  settingsSidebar: { width:200,background:"rgba(0,0,0,0.3)",borderRight:"1px solid rgba(255,255,255,0.08)",padding:"20px 12px",display:"flex",flexDirection:"column",gap:4,position:"relative" },
  settingsSidebarTitle: { padding:"0 8px 16px",borderBottom:"1px solid rgba(255,255,255,0.08)",marginBottom:8 },
  settingsCloseBtn: { position:"absolute",top:16,right:12,background:"rgba(255,255,255,0.08)",border:"none",color:"rgba(255,255,255,0.6)",fontSize:14,cursor:"pointer",padding:"4px 8px",borderRadius:6,lineHeight:1 },
  settingsTabBtn: { display:"flex",alignItems:"center",gap:10,padding:"10px 12px",background:"none",border:"none",color:"rgba(255,255,255,0.65)",cursor:"pointer",borderRadius:10,fontSize:13,textAlign:"left",transition:"all .15s" },
  settingsTabActive: { background:"rgba(74,255,158,0.12)",color:"#fff",fontWeight:600 },
  settingsContent: { flex:1,padding:"28px 32px",overflowY:"auto",position:"relative" },
  settingsContentClose: { position:"absolute",top:16,right:16,background:"rgba(255,255,255,0.08)",border:"none",color:"rgba(255,255,255,0.5)",fontSize:14,cursor:"pointer",padding:"4px 10px",borderRadius:6 },
  settingsTitle: { color:"#fff",fontSize:20,fontWeight:700,marginBottom:6 },
  settingsDivider: { height:1,background:"rgba(255,255,255,0.08)",margin:"12px 0 20px" },
  settingRow: { display:"flex",justifyContent:"space-between",alignItems:"center",padding:"14px 0",borderBottom:"1px solid rgba(255,255,255,0.06)" },
  settingLabel: { color:"#fff",fontSize:13,fontWeight:500,marginBottom:3 },
  settingDesc: { color:"rgba(255,255,255,0.45)",fontSize:11 },
  settingSelect: { padding:"7px 12px",background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:8,color:"#fff",fontSize:12,cursor:"pointer",outline:"none" },
  dangerBtn: { padding:"8px 16px",background:"rgba(255,68,68,0.15)",border:"1px solid rgba(255,68,68,0.3)",borderRadius:8,color:"#ff6b6b",cursor:"pointer",fontSize:12,fontWeight:600 },
  toggle: { width:44,height:24,borderRadius:12,cursor:"pointer",transition:"background .2s",position:"relative",display:"flex",alignItems:"center",padding:"0 2px" },
  toggleKnob: { width:20,height:20,borderRadius:"50%",background:"#fff",transition:"transform .2s",position:"absolute" },
  weatherCardFixed: { position:"fixed",bottom:70,right:20,zIndex:100,display:"flex",alignItems:"center",gap:12,background:"rgba(0,0,0,0.55)",border:"1px solid rgba(255,255,255,0.18)",borderRadius:16,padding:"9px 18px",backdropFilter:"blur(12px)" },
  bottomNav: { position:"fixed",bottom:20,right:20,zIndex:100,display:"flex",gap:10,alignItems:"center" },
  navBtn: { padding:"9px 18px",background:"rgba(0,0,0,0.55)",border:"1px solid rgba(255,255,255,0.2)",borderRadius:22,color:"#fff",cursor:"pointer",fontSize:12,fontWeight:500,backdropFilter:"blur(10px)",transition:"all .2s",letterSpacing:.3 },
  main: { position:"relative",flex:1,overflowY:"auto",zIndex:2 },
  hero: { minHeight:"100vh",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:28,padding:"40px 50px 80px" },
  featureGrid: { display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:20,width:"100%",maxWidth:460 },
  featureLabel: { fontSize:11,color:"rgba(255,255,255,0.7)",letterSpacing:"1px",marginBottom:5 },
  featureValue: { fontSize:13,fontWeight:600,color:"#4aff9e",letterSpacing:"1px" },
  temp: { fontSize:16,fontWeight:700,color:"#fff" },
  weatherDesc: { fontSize:11,color:"rgba(255,255,255,0.85)",fontWeight:500 },
  locationText: { fontSize:11,color:"#4aff9e",marginLeft:"auto",fontWeight:600 },
  searchIconBtn: { background:"rgba(0,0,0,0.4)",border:"1px solid rgba(255,255,255,0.2)",fontSize:13,cursor:"pointer",color:"#fff",padding:"4px 8px",borderRadius:10,marginLeft:5 },
  voiceSection: { width:"100%",display:"flex",flexDirection:"column",alignItems:"center",gap:22,marginTop:10 },
  waveContainer: { display:"flex",gap:4,alignItems:"center",justifyContent:"center",height:50 },
  waveBar: { width:3,borderRadius:2,transition:"height .1s ease" },
  micButton: { width:70,height:70,borderRadius:"50%",background:"rgba(0,0,0,0.5)",border:"2px solid rgba(74,255,158,0.5)",fontSize:28,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",color:"#fff",transition:"all .3s" },
  micActive: { background:"rgba(74,255,158,0.2)",borderColor:"#4aff9e",boxShadow:"0 0 20px rgba(74,255,158,0.3)" },
  micLoading: { opacity:.6,cursor:"not-allowed" },
  micLabel: { fontSize:11,color:"#4aff9e",letterSpacing:"2px",fontWeight:500 },
  aiResponseContainer: { width:"100%",maxWidth:650,background:"rgba(0,0,0,0.45)",backdropFilter:"blur(12px)",borderRadius:20,border:"1px solid rgba(74,255,158,0.2)",overflow:"hidden",marginTop:10 },
  aiResponseHeader: { display:"flex",alignItems:"center",gap:8,padding:"12px 20px",background:"rgba(0,0,0,0.2)",borderBottom:"1px solid rgba(74,255,158,0.15)" },
  aiResponseIcon: { fontSize:16 },
  aiResponseTitle: { fontSize:11,fontWeight:600,color:"#4aff9e",letterSpacing:1 },
  aiResponseContent: { padding:20,minHeight:100 },
  aiResponseText: { fontSize:14,lineHeight:1.6,color:"#fff",fontWeight:400 },
  listeningIndicator: { padding:"10px 20px",background:"rgba(74,255,158,0.1)",borderRadius:12,textAlign:"center" },
  listeningText: { fontSize:12,color:"#4aff9e",letterSpacing:1 },
  chatbotSection: { width:"100%",maxWidth:650,margin:"10px auto 0",background:"rgba(0,0,0,0.45)",backdropFilter:"blur(10px)",borderRadius:16,border:"1px solid rgba(255,255,255,0.2)",overflow:"hidden" },
  chatHeader: { display:"flex",justifyContent:"space-between",alignItems:"center",padding:"10px 16px",borderBottom:"1px solid rgba(255,255,255,0.15)",background:"rgba(0,0,0,0.2)" },
  chatHeaderTitle: { fontSize:11,fontWeight:600,color:"#4aff9e",letterSpacing:1 },
  downloadChatBtn: { padding:"4px 10px",background:"rgba(74,255,158,0.15)",border:"1px solid rgba(74,255,158,0.3)",borderRadius:8,color:"#4aff9e",cursor:"pointer",fontSize:10 },
  chatMessages: { height:300,overflowY:"auto",padding:16,display:"flex",flexDirection:"column",gap:10 },
  chatBubble: { padding:"8px 12px",borderRadius:12,maxWidth:"80%",wordWrap:"break-word",fontSize:12,position:"relative" },
  chatBubbleHeader: { display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4 },
  chatBubbleContent: { fontSize:12,lineHeight:1.5,wordBreak:"break-word" },
  userBubble: { background:"rgba(74,255,158,0.2)",border:"1px solid rgba(74,255,158,0.3)",alignSelf:"flex-end",color:"#fff" },
  aiBubble: { background:"rgba(255,255,255,0.1)",border:"1px solid rgba(255,255,255,0.2)",alignSelf:"flex-start",color:"#fff" },
  copyBtn: { background:"none",border:"none",color:"rgba(255,255,255,0.5)",cursor:"pointer",fontSize:11,padding:"2px 5px",borderRadius:4 },
  typingCursor: { display:"inline-block",width:2,height:12,backgroundColor:"#4aff9e",marginLeft:2,animation:"blink 1s infinite" },
  emptyChat: { textAlign:"center",color:"rgba(255,255,255,0.5)",padding:30,fontSize:12 },
  typingIndicator: { display:"flex",justifyContent:"center",gap:6,padding:10,color:"#4aff9e",fontSize:14 },
  chatInputContainer: { display:"flex",padding:12,borderTop:"1px solid rgba(255,255,255,0.2)",gap:10 },
  chatInput: { flex:1,padding:8,background:"rgba(255,255,255,0.1)",border:"1px solid rgba(255,255,255,0.2)",borderRadius:8,color:"#fff",fontSize:12,outline:"none" },
  sendBtn: { padding:"8px 16px",background:"rgba(74,255,158,0.2)",border:"1px solid rgba(74,255,158,0.3)",borderRadius:8,color:"#4aff9e",cursor:"pointer",fontSize:12 },
  lenseSection: { width:"100%",maxWidth:650,margin:"10px auto 0",background:"rgba(0,0,0,0.45)",backdropFilter:"blur(10px)",borderRadius:20,border:"1px solid rgba(255,255,255,0.2)",padding:20 },
  lenseHeader: { textAlign:"center",marginBottom:15 },
  lenseIcon: { fontSize:32,display:"block",marginBottom:5 },
  lenseTitle: { fontSize:18,fontWeight:600,color:"#4aff9e",letterSpacing:1,display:"block" },
  lenseSubtitle: { fontSize:10,color:"rgba(255,255,255,0.6)",display:"block",marginTop:4 },
  actionButtons: { display:"flex",justifyContent:"center",gap:15,marginBottom:20 },
  cameraBtn: { padding:"8px 20px",background:"rgba(58,134,255,0.2)",border:"1px solid rgba(58,134,255,0.3)",borderRadius:12,color:"#3a86ff",cursor:"pointer",fontSize:13,fontWeight:500 },
  galleryBtn: { padding:"8px 20px",background:"rgba(74,255,158,0.15)",border:"1px solid rgba(74,255,158,0.3)",borderRadius:12,color:"#4aff9e",cursor:"pointer",fontSize:13,fontWeight:500,display:"inline-block" },
  fileInput: { display:"none" },
  cameraContainer: { marginBottom:15,textAlign:"center" },
  cameraPreview: { width:"100%",maxHeight:250,borderRadius:16,objectFit:"cover" },
  cameraButtons: { display:"flex",justifyContent:"center",gap:12,marginTop:12 },
  captureBtn: { padding:"6px 18px",background:"rgba(74,255,158,0.2)",border:"1px solid rgba(74,255,158,0.3)",borderRadius:10,color:"#4aff9e",cursor:"pointer",fontSize:12 },
  closeCameraBtn: { padding:"6px 18px",background:"rgba(255,68,68,0.2)",border:"1px solid rgba(255,68,68,0.3)",borderRadius:10,color:"#ff6b6b",cursor:"pointer",fontSize:12 },
  imagePreviewContainer: { position:"relative",marginBottom:15,textAlign:"center" },
  imagePreview: { maxWidth:"100%",maxHeight:200,borderRadius:12,border:"1px solid rgba(255,255,255,0.2)" },
  clearImageBtn: { position:"absolute",top:5,right:5,background:"rgba(0,0,0,0.6)",border:"none",color:"#fff",padding:"3px 8px",borderRadius:20,cursor:"pointer",fontSize:10 },
  questionArea: { display:"flex",gap:10,marginBottom:15 },
  lenseInput: { flex:1,padding:10,background:"rgba(255,255,255,0.1)",border:"1px solid rgba(255,255,255,0.2)",borderRadius:12,color:"#fff",fontSize:12,outline:"none" },
  analyzeBtn: { padding:"10px 20px",borderRadius:12,fontSize:12,fontWeight:500,border:"1px solid rgba(74,255,158,0.3)",background:"rgba(74,255,158,0.2)",color:"#4aff9e",cursor:"pointer",transition:"all .2s",minWidth:110,textAlign:"center" },

  // ── COOLDOWN BANNER ─────────────────────────────────────────────────────────
  cooldownBanner: { background:"rgba(255,180,0,0.08)",border:"1px solid rgba(255,180,0,0.3)",borderRadius:14,padding:"14px 16px",marginBottom:14,animation:"cooldownPulse 2s ease-in-out infinite" },
  cooldownTop: { display:"flex",alignItems:"center",gap:10,marginBottom:10 },
  cooldownIcon: { fontSize:18 },
  cooldownLabel: { fontSize:9,fontFamily:"'Orbitron',sans-serif",color:"rgba(255,180,0,0.8)",letterSpacing:1.5,fontWeight:700,flex:1 },
  cooldownTime: { fontSize:22,fontWeight:800,color:"#ffb400",fontFamily:"'Orbitron',sans-serif",letterSpacing:2,minWidth:70,textAlign:"right" },
  cooldownBarBg: { height:5,background:"rgba(255,255,255,0.08)",borderRadius:10,overflow:"hidden",marginBottom:10 },
  cooldownBarFill: { height:"100%",borderRadius:10,background:"linear-gradient(90deg,#ffb400,#ff6b00)",transition:"width 1s linear",backgroundSize:"200% 100%",animation:"barFlow 2s linear infinite" },
  cooldownSub: { fontSize:10,color:"rgba(255,255,255,0.45)",lineHeight:1.5 },

  visionResponseContainer: { background:"rgba(74,255,158,0.08)",borderRadius:16,border:"1px solid rgba(74,255,158,0.2)",overflow:"hidden",marginTop:15 },
  visionResponseHeader: { display:"flex",alignItems:"center",gap:8,padding:"10px 16px",background:"rgba(0,0,0,0.2)",borderBottom:"1px solid rgba(74,255,158,0.15)" },
  visionResponseTitle: { fontSize:10,fontWeight:600,color:"#4aff9e",letterSpacing:1 },
  visionResponseContent: { padding:16,minHeight:80 },
  visionResponseText: { fontSize:13,lineHeight:1.6,color:"#fff" },
  loadingDots: { display:"flex",justifyContent:"center",gap:8,fontSize:20,color:"#4aff9e" },
  noImageMessage: { textAlign:"center",padding:30,color:"rgba(255,255,255,0.5)",fontSize:12,display:"flex",flexDirection:"column",gap:8 },
  newsSection: { width:"100%",maxWidth:750,margin:"10px auto 0",background:"rgba(0,0,0,0.45)",backdropFilter:"blur(12px)",borderRadius:20,border:"1px solid rgba(255,255,255,0.15)",overflow:"hidden",transition:"all .3s" },
  newsHeader: { display:"flex",alignItems:"center",gap:10,padding:"12px 16px",background:"rgba(0,0,0,0.25)",borderBottom:"1px solid rgba(255,255,255,0.1)" },
  newsIcon: { fontSize:18 },
  newsTitle: { fontSize:12,fontWeight:600,color:"#4aff9e",letterSpacing:1 },
  newsCount: { fontSize:10,background:"rgba(74,255,158,0.15)",padding:"2px 6px",borderRadius:12,color:"#4aff9e" },
  newsRefreshBtn: { background:"rgba(255,255,255,0.1)",border:"1px solid rgba(255,255,255,0.2)",borderRadius:8,color:"#fff",padding:"4px 10px",cursor:"pointer",fontSize:12,marginLeft:"auto" },
  newsStopBtn: { background:"rgba(255,68,68,0.2)",border:"1px solid rgba(255,68,68,0.3)",borderRadius:8,color:"#ff6b6b",padding:"4px 10px",cursor:"pointer",fontSize:11 },
  categoryContainer: { display:"flex",flexWrap:"wrap",gap:6,padding:"12px 16px",borderBottom:"1px solid rgba(255,255,255,0.08)" },
  categoryBtn: { padding:"4px 12px",background:"rgba(255,255,255,0.08)",border:"1px solid rgba(255,255,255,0.15)",borderRadius:20,color:"rgba(255,255,255,0.7)",fontSize:10,cursor:"pointer",transition:"all .2s" },
  categoryActive: { background:"rgba(74,255,158,0.2)",borderColor:"#4aff9e",color:"#4aff9e" },
  newsList: { maxHeight:450,overflowY:"auto",padding:"12px 16px",display:"flex",flexDirection:"column",gap:12 },
  newsCard: { background:"rgba(255,255,255,0.06)",borderRadius:12,padding:12,border:"1px solid rgba(255,255,255,0.08)" },
  newsCardHeader: { display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8 },
  newsSource: { fontSize:9,color:"#4aff9e",fontWeight:500 },
  newsTime: { fontSize:8,color:"rgba(255,255,255,0.4)" },
  newsTitleText: { fontSize:12,fontWeight:600,color:"#fff",marginBottom:6,lineHeight:1.4 },
  newsDescription: { fontSize:10,color:"rgba(255,255,255,0.7)",lineHeight:1.5,marginBottom:10 },
  newsActions: { display:"flex",gap:8 },
  newsListenBtn: { padding:"4px 10px",background:"rgba(74,255,158,0.15)",border:"1px solid rgba(74,255,158,0.25)",borderRadius:8,color:"#4aff9e",fontSize:10,cursor:"pointer" },
  noNews: { display:"flex",flexDirection:"column",alignItems:"center",gap:10,padding:30,color:"rgba(255,255,255,0.5)",fontSize:12 },
  retryBtn: { padding:"6px 16px",background:"rgba(74,255,158,0.15)",border:"1px solid rgba(74,255,158,0.3)",borderRadius:8,color:"#4aff9e",cursor:"pointer",fontSize:11,marginTop:6 },
  newsFooter: { padding:"10px 16px",borderTop:"1px solid rgba(255,255,255,0.08)",textAlign:"center",fontSize:9,color:"rgba(255,255,255,0.35)" },
  skeletonContainer: { display:"flex",flexDirection:"column",gap:12 },
  skeletonCard: { background:"rgba(255,255,255,0.05)",borderRadius:12,padding:12 },
  skeletonHeader: { display:"flex",justifyContent:"space-between",marginBottom:8 },
  skeletonSource: { width:50,height:8,background:"rgba(255,255,255,0.1)",borderRadius:4,animation:"shimmer 1.5s ease-in-out infinite",backgroundImage:"linear-gradient(90deg,rgba(255,255,255,0) 0%,rgba(255,255,255,0.05) 50%,rgba(255,255,255,0) 100%)",backgroundSize:"200px 100%" },
  skeletonTime: { width:30,height:6,background:"rgba(255,255,255,0.08)",borderRadius:4,animation:"shimmer 1.5s ease-in-out infinite" },
  skeletonTitle: { height:12,background:"rgba(255,255,255,0.1)",borderRadius:4,marginBottom:8,width:"80%",animation:"shimmer 1.5s ease-in-out infinite" },
  skeletonLine: { height:8,background:"rgba(255,255,255,0.08)",borderRadius:4,marginBottom:6,width:"100%",animation:"shimmer 1.5s ease-in-out infinite" },
  skeletonLineShort: { height:8,background:"rgba(255,255,255,0.08)",borderRadius:4,marginBottom:6,width:"60%",animation:"shimmer 1.5s ease-in-out infinite" },
  backToMenuBtn: { marginTop:15,padding:"8px 20px",background:"rgba(0,0,0,0.5)",border:"1px solid rgba(255,255,255,0.2)",borderRadius:8,color:"#fff",cursor:"pointer",fontSize:11 },
};

export default App;