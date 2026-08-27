export type StressLevel = "healthy" | "mild" | "severe";
export type Lang = "en" | "hi";

export interface FarmPayload {
  farm: {
    farm_id: string;
    name: string;
    name_hi: string;
    latitude: number;
    longitude: number;
    crop: string;
    crop_hi: string;
    crop_emoji: string;
    area_ha: number;
    sowing_date: string;
    region: string;
  };
  metrics: {
    current_ndvi: number;
    current_ndwi: number;
    sar_vv_db: number;
    sar_vh_db: number;
    sar_ratio: number;
    ndvi_trend_pct: number;
    health_trend: "improving" | "stable" | "declining";
    stress_level: StressLevel;
    moisture_category: "Dry" | "Adequate" | "Wet";
  };
  prediction: {
    predicted_crop: string;
    confidence: number;
    secondary_crop: string;
    secondary_confidence: number;
    features: { name: string; name_hi: string; importance: number }[];
  };
  advisory: {
    summary: Record<Lang, string>;
    irrigation_advice: Record<Lang, string>;
    action_items: { title: Record<Lang, string>; detail: Record<Lang, string>; urgency: StressLevel }[];
  };
}

export interface SeriesPoint {
  date: string;
  day: number;
  ndvi: number;
  sar_vv: number;
}

/** Deterministic pseudo-random so SSR and client agree. */
function rand(seed: number) {
  const x = Math.sin(seed * 127.1) * 43758.5453;
  return x - Math.floor(x);
}

export function buildSeries(payload: FarmPayload, days: number): SeriesPoint[] {
  const peak = payload.metrics.current_ndvi;
  const base = payload.farm.farm_id.length;
  const out: SeriesPoint[] = [];
  for (let i = days; i >= 0; i -= Math.max(1, Math.round(days / 45))) {
    const t = (days - i) / days;
    const curve = 0.18 + (peak - 0.18) * Math.sin(Math.PI * Math.min(1, t * 0.92));
    const noise = (rand(base + i) - 0.5) * 0.035;
    const d = new Date();
    d.setDate(d.getDate() - i);
    out.push({
      date: d.toISOString().slice(0, 10),
      day: -i,
      ndvi: Math.max(0.05, Math.min(0.95, Number((curve + noise).toFixed(3)))),
      sar_vv: Number(
        (payload.metrics.sar_vv_db - 3.2 + curve * 3.4 + (rand(base + i * 3) - 0.5) * 1.9).toFixed(2),
      ),
    });
  }
  return out;
}

export const FARMS: FarmPayload[] = [
  {
    farm: {
      farm_id: "farm_sehore_001",
      name: "Sehore Pilot Farm 1 — MP",
      name_hi: "सीहोर पायलट फार्म 1 — म.प्र.",
      latitude: 23.2045,
      longitude: 77.0825,
      crop: "Wheat",
      crop_hi: "गेहूँ",
      crop_emoji: "🌾",
      area_ha: 4.5,
      sowing_date: "2025-11-18",
      region: "Madhya Pradesh",
    },
    metrics: {
      current_ndvi: 0.72,
      current_ndwi: 0.35,
      sar_vv_db: -10.8,
      sar_vh_db: -16.4,
      sar_ratio: 0.65,
      ndvi_trend_pct: 8.4,
      health_trend: "improving",
      stress_level: "healthy",
      moisture_category: "Adequate",
    },
    prediction: {
      predicted_crop: "Wheat",
      confidence: 0.942,
      secondary_crop: "Mustard",
      secondary_confidence: 0.048,
      features: [
        { name: "NDVI temporal profile", name_hi: "एनडीवीआई समय-प्रोफ़ाइल", importance: 0.34 },
        { name: "SAR VH backscatter", name_hi: "एसएआर वीएच बैकस्कैटर", importance: 0.24 },
        { name: "VV/VH ratio", name_hi: "वीवी/वीएच अनुपात", importance: 0.17 },
        { name: "NDWI water content", name_hi: "एनडीडब्ल्यूआई जल मात्रा", importance: 0.14 },
        { name: "Red-edge slope", name_hi: "रेड-एज ढलान", importance: 0.11 },
      ],
    },
    advisory: {
      summary: {
        en: "Vegetation index is at optimal vegetative peak for day 92 after sowing. SAR backscatter indicates adequate root-zone moisture with no irrigation deficit detected.",
        hi: "बुवाई के 92वें दिन वनस्पति सूचकांक अपने आदर्श स्तर पर है। एसएआर बैकस्कैटर से जड़-क्षेत्र में पर्याप्त नमी दिख रही है, सिंचाई की कमी नहीं है।",
      },
      irrigation_advice: {
        en: "No immediate irrigation needed for the next 4–5 days. Re-evaluate when VV backscatter drops below -14 dB.",
        hi: "अगले 4–5 दिनों तक सिंचाई की आवश्यकता नहीं है। जब वीवी बैकस्कैटर -14 dB से नीचे जाए तब पुनः जाँचें।",
      },
      action_items: [
        {
          title: { en: "Hold irrigation cycle", hi: "सिंचाई रोकें" },
          detail: {
            en: "Root-zone moisture is adequate. Skip the scheduled flood irrigation to avoid lodging risk.",
            hi: "जड़-क्षेत्र में नमी पर्याप्त है। गिरने के जोखिम से बचने हेतु निर्धारित सिंचाई टालें।",
          },
          urgency: "healthy",
        },
        {
          title: { en: "Aphid vigilance at tillering", hi: "कल्ले अवस्था में माहू निगरानी" },
          detail: {
            en: "Scout 10 random tillers per acre twice this week; treat if >5 aphids per tiller.",
            hi: "इस सप्ताह दो बार प्रति एकड़ 10 कल्लों की जाँच करें; प्रति कल्ला 5 से अधिक माहू पर उपचार करें।",
          },
          urgency: "mild",
        },
        {
          title: { en: "Nitrogen top-dressing window", hi: "नाइट्रोजन टॉप-ड्रेसिंग" },
          detail: {
            en: "Apply 25 kg/ha urea within 6 days while NDVI slope stays positive.",
            hi: "एनडीवीआई बढ़त बनी रहने तक 6 दिनों में 25 कि.ग्रा./हे. यूरिया डालें।",
          },
          urgency: "healthy",
        },
      ],
    },
  },
  {
    farm: {
      farm_id: "farm_varanasi_014",
      name: "Varanasi Wheat Plot — UP",
      name_hi: "वाराणसी गेहूँ प्लॉट — उ.प्र.",
      latitude: 25.3176,
      longitude: 82.9739,
      crop: "Mustard",
      crop_hi: "सरसों",
      crop_emoji: "🌱",
      area_ha: 2.8,
      sowing_date: "2025-11-02",
      region: "Uttar Pradesh",
    },
    metrics: {
      current_ndvi: 0.54,
      current_ndwi: 0.21,
      sar_vv_db: -13.9,
      sar_vh_db: -19.2,
      sar_ratio: 0.51,
      ndvi_trend_pct: -4.7,
      health_trend: "declining",
      stress_level: "mild",
      moisture_category: "Dry",
    },
    prediction: {
      predicted_crop: "Mustard",
      confidence: 0.871,
      secondary_crop: "Wheat",
      secondary_confidence: 0.092,
      features: [
        { name: "NDVI temporal profile", name_hi: "एनडीवीआई समय-प्रोफ़ाइल", importance: 0.31 },
        { name: "VV/VH ratio", name_hi: "वीवी/वीएच अनुपात", importance: 0.26 },
        { name: "SAR VH backscatter", name_hi: "एसएआर वीएच बैकस्कैटर", importance: 0.19 },
        { name: "NDWI water content", name_hi: "एनडीडब्ल्यूआई जल मात्रा", importance: 0.13 },
        { name: "Red-edge slope", name_hi: "रेड-एज ढलान", importance: 0.11 },
      ],
    },
    advisory: {
      summary: {
        en: "Canopy greenness has slipped 4.7% over 14 days and radar backscatter is trending dry. Early moisture stress is emerging in the north-west quadrant of the plot.",
        hi: "14 दिनों में हरियाली 4.7% घटी है और रडार बैकस्कैटर सूखेपन की ओर है। प्लॉट के उत्तर-पश्चिम भाग में नमी तनाव शुरू हो रहा है।",
      },
      irrigation_advice: {
        en: "Schedule a light irrigation within the next 48 hours, prioritising the north-west quadrant.",
        hi: "अगले 48 घंटों में हल्की सिंचाई करें, विशेषकर उत्तर-पश्चिम भाग में।",
      },
      action_items: [
        {
          title: { en: "Irrigate within 48 hours", hi: "48 घंटों में सिंचाई करें" },
          detail: {
            en: "VV backscatter at -13.9 dB signals depleting root-zone water. Apply 35–40 mm.",
            hi: "-13.9 dB वीवी बैकस्कैटर जड़-क्षेत्र में पानी की कमी दर्शाता है। 35–40 मि.मी. पानी दें।",
          },
          urgency: "severe",
        },
        {
          title: { en: "Verify canopy dip on ground", hi: "ज़मीन पर हरियाली गिरावट जाँचें" },
          detail: {
            en: "Confirm whether the NDVI dip is stress or aphid damage before spraying.",
            hi: "छिड़काव से पहले जाँचें कि गिरावट तनाव से है या माहू से।",
          },
          urgency: "mild",
        },
      ],
    },
  },
  {
    farm: {
      farm_id: "farm_patna_007",
      name: "Patna Rice Zone — Bihar",
      name_hi: "पटना धान क्षेत्र — बिहार",
      latitude: 25.5941,
      longitude: 85.1376,
      crop: "Rice",
      crop_hi: "धान",
      crop_emoji: "🍚",
      area_ha: 6.2,
      sowing_date: "2025-07-04",
      region: "Bihar",
    },
    metrics: {
      current_ndvi: 0.38,
      current_ndwi: 0.48,
      sar_vv_db: -8.1,
      sar_vh_db: -14.1,
      sar_ratio: 0.78,
      ndvi_trend_pct: -12.6,
      health_trend: "declining",
      stress_level: "severe",
      moisture_category: "Wet",
    },
    prediction: {
      predicted_crop: "Rice",
      confidence: 0.918,
      secondary_crop: "Sugarcane",
      secondary_confidence: 0.055,
      features: [
        { name: "SAR VH backscatter", name_hi: "एसएआर वीएच बैकस्कैटर", importance: 0.36 },
        { name: "NDWI water content", name_hi: "एनडीडब्ल्यूआई जल मात्रा", importance: 0.25 },
        { name: "NDVI temporal profile", name_hi: "एनडीवीआई समय-प्रोफ़ाइल", importance: 0.18 },
        { name: "VV/VH ratio", name_hi: "वीवी/वीएच अनुपात", importance: 0.13 },
        { name: "Red-edge slope", name_hi: "रेड-एज ढलान", importance: 0.08 },
      ],
    },
    advisory: {
      summary: {
        en: "A 0.12 NDVI collapse in 10 days combined with unusually high VV backscatter points to waterlogging after the last monsoon spell. Immediate drainage intervention is advised.",
        hi: "10 दिनों में 0.12 एनडीवीआई गिरावट और असामान्य रूप से उच्च वीवी बैकस्कैटर जलभराव दर्शाते हैं। तुरंत जल निकासी करें।",
      },
      irrigation_advice: {
        en: "Stop all irrigation. Open field drains and target 5 cm standing water within 72 hours.",
        hi: "सिंचाई पूरी तरह रोकें। नालियाँ खोलें और 72 घंटों में जलस्तर 5 से.मी. तक लाएँ।",
      },
      action_items: [
        {
          title: { en: "Drain excess standing water", hi: "अतिरिक्त पानी निकालें" },
          detail: {
            en: "Backscatter above -9 dB across 60% of the AOI indicates flooded canopy conditions.",
            hi: "60% क्षेत्र में -9 dB से ऊपर बैकस्कैटर जलभराव दर्शाता है।",
          },
          urgency: "severe",
        },
        {
          title: { en: "Inspect for blast & sheath blight", hi: "ब्लास्ट व शीथ ब्लाइट जाँचें" },
          detail: {
            en: "Prolonged saturation raises fungal pressure; scout lower leaf sheaths.",
            hi: "लंबे जलभराव से फफूँद का खतरा बढ़ता है; निचली पत्तियाँ जाँचें।",
          },
          urgency: "severe",
        },
        {
          title: { en: "Delay nitrogen application", hi: "नाइट्रोजन टालें" },
          detail: {
            en: "Hold urea until the field drains to avoid leaching losses.",
            hi: "जल निकासी तक यूरिया न डालें, अन्यथा पोषक तत्व बह जाएंगे।",
          },
          urgency: "mild",
        },
      ],
    },
  },
];

export const T = {
  en: {
    liveFeed: "Live Satellite Feed (Sentinel-1 & Sentinel-2)",
    liveMode: "Live Earth Engine Feed",
    demoMode: "Demo Mode",
    addField: "+ Add Custom Field via GPS",
    locate: "Locate My Farm",
    aoi: "AOI Buffer",
    ndviTitle: "Canopy Greenness (NDVI)",
    sarTitle: "Soil & Crop Moisture (SAR VV/VH)",
    cropTitle: "Predicted Crop Classification",
    stressTitle: "Stress & Anomaly Level",
    dayTrend: "14-day trend",
    confidence: "Confidence",
    analytics: "Dual-Sensor Analytics",
    analyticsSub: "Optical NDVI vs. all-weather SAR radar backscatter",
    features: "Classification Feature Importance",
    advisory: "AI Agronomic Advisory",
    advisorySub: "Grounded in Sentinel-1/2 remote sensing physics",
    summary: "Executive Summary",
    actions: "Immediate Actions",
    irrigation: "Irrigation Guidance",
    ask: "Ask AgriN",
    askSub: "Your satellite-grounded farm assistant",
    send: "Send",
    report: "Download Farm Health Report",
    refresh: "Refresh Live Satellite Imagery",
    share: "Share Advisory via WhatsApp",
    healthy: "Healthy",
    mild: "Mild Stress",
    severe: "Severe Anomaly",
    allWeather: "All-weather · Cloud penetrating",
    optimal: "vs. optimal vegetative peak 0.85",
    d30: "Last 30 Days",
    d60: "Last 60 Days",
    d90: "Full Crop Cycle",
    provenance:
      "Generated from Sentinel-2 L2A optical reflectance and Sentinel-1 GRD radar backscatter. No agronomic claim is made beyond what the observed indices support.",
    placeholder: "Ask about irrigation, NDVI drops, fertiliser…",
  },
  hi: {
    liveFeed: "लाइव सैटेलाइट फ़ीड (सेंटिनल-1 और सेंटिनल-2)",
    liveMode: "लाइव अर्थ इंजन फ़ीड",
    demoMode: "डेमो मोड",
    addField: "+ जीपीएस से नया खेत जोड़ें",
    locate: "मेरा खेत खोजें",
    aoi: "क्षेत्र त्रिज्या",
    ndviTitle: "फसल हरियाली (एनडीवीआई)",
    sarTitle: "मिट्टी व फसल नमी (एसएआर वीवी/वीएच)",
    cropTitle: "अनुमानित फसल वर्गीकरण",
    stressTitle: "तनाव व असामान्यता स्तर",
    dayTrend: "14-दिन रुझान",
    confidence: "विश्वसनीयता",
    analytics: "द्वि-सेंसर विश्लेषण",
    analyticsSub: "ऑप्टिकल एनडीवीआई बनाम एसएआर रडार बैकस्कैटर",
    features: "वर्गीकरण विशेषता महत्व",
    advisory: "एआई कृषि सलाह",
    advisorySub: "सेंटिनल-1/2 रिमोट सेंसिंग भौतिकी पर आधारित",
    summary: "सार",
    actions: "तत्काल कार्य",
    irrigation: "सिंचाई मार्गदर्शन",
    ask: "AgriN से पूछें",
    askSub: "आपका सैटेलाइट आधारित कृषि सहायक",
    send: "भेजें",
    report: "फ़ार्म हेल्थ रिपोर्ट डाउनलोड करें",
    refresh: "लाइव सैटेलाइट इमेजरी रिफ्रेश करें",
    share: "व्हाट्सएप पर सलाह भेजें",
    healthy: "स्वस्थ",
    mild: "हल्का तनाव",
    severe: "गंभीर असामान्यता",
    allWeather: "हर मौसम · बादल भेदी",
    optimal: "आदर्श शिखर 0.85 की तुलना में",
    d30: "पिछले 30 दिन",
    d60: "पिछले 60 दिन",
    d90: "पूरा फसल चक्र",
    provenance:
      "सेंटिनल-2 ऑप्टिकल परावर्तन और सेंटिनल-1 रडार बैकस्कैटर से निर्मित। केवल उपग्रह सूचकांकों द्वारा समर्थित सलाह ही दी गई है।",
    placeholder: "सिंचाई, एनडीवीआई गिरावट, उर्वरक के बारे में पूछें…",
  },
} satisfies Record<Lang, Record<string, string>>;

export const SUGGESTIONS: Record<Lang, string[]> = {
  en: [
    "Should I irrigate this week based on SAR moisture?",
    "Why did my NDVI drop recently?",
    "What fertiliser schedule fits the current stage?",
    "Is cloud cover affecting today's reading?",
  ],
  hi: [
    "क्या इस सप्ताह सिंचाई करनी चाहिए?",
    "मेरा एनडीवीआई क्यों गिरा?",
    "अभी कौन-सा उर्वरक कार्यक्रम सही है?",
    "क्या बादल आज की रीडिंग पर असर डाल रहे हैं?",
  ],
};

export function answerFor(question: string, p: FarmPayload, lang: Lang): string {
  const q = question.toLowerCase();
  const m = p.metrics;
  if (lang === "hi") {
    if (q.includes("सिंचाई") || q.includes("irrigat"))
      return `${p.farm.name_hi} का वीवी बैकस्कैटर ${m.sar_vv_db} dB है (${m.moisture_category})। ${p.advisory.irrigation_advice.hi}`;
    if (q.includes("एनडीवीआई") || q.includes("ndvi"))
      return `एनडीवीआई ${m.current_ndvi} है और 14 दिनों में ${m.ndvi_trend_pct}% बदला है। ${p.advisory.summary.hi}`;
    if (q.includes("उर्वरक") || q.includes("fertil"))
      return `वर्तमान अवस्था में 25 कि.ग्रा./हे. यूरिया की टॉप-ड्रेसिंग उपयुक्त है, बशर्ते नमी पर्याप्त हो (अभी ${m.moisture_category})।`;
    if (q.includes("बादल") || q.includes("cloud"))
      return `ऑप्टिकल रीडिंग बादलों से प्रभावित हो सकती है, परंतु सेंटिनल-1 रडार बादलों के पार देखता है — इसलिए ${m.sar_vv_db} dB भरोसेमंद है।`;
    return `${p.advisory.summary.hi} ${p.advisory.irrigation_advice.hi}`;
  }
  if (q.includes("irrigat") || q.includes("water"))
    return `For ${p.farm.name}, Sentinel-1 VV backscatter reads ${m.sar_vv_db} dB with a VV/VH ratio of ${m.sar_ratio} — root-zone moisture is ${m.moisture_category.toLowerCase()}. ${p.advisory.irrigation_advice.en}`;
  if (q.includes("ndvi") || q.includes("drop") || q.includes("green"))
    return `Current NDVI is ${m.current_ndvi} (${m.ndvi_trend_pct > 0 ? "+" : ""}${m.ndvi_trend_pct}% over 14 days). ${p.advisory.summary.en}`;
  if (q.includes("fertil") || q.includes("nitrogen") || q.includes("urea"))
    return `At the current phenological stage a 25 kg/ha urea top-dressing is appropriate, but only while moisture stays ${m.moisture_category === "Dry" ? "restored — irrigate first" : "adequate"}. Split the dose if NDVI slope flattens.`;
  if (q.includes("cloud") || q.includes("rain") || q.includes("monsoon"))
    return `Optical NDVI can be masked by cloud, but Sentinel-1 C-band radar penetrates cloud cover, so the ${m.sar_vv_db} dB backscatter reading remains valid today.`;
  return `${p.advisory.summary.en} ${p.advisory.irrigation_advice.en}`;
}
