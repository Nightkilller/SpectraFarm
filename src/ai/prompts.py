"""
AgriN — Gemini Prompt Templates

Contains all prompt templates for the Gemini AI integration.
Prompts are carefully designed to:

1. Use ONLY supplied agricultural data
2. Never fabricate measurements
3. Distinguish measured data from interpretation
4. Use simple farmer-friendly language
5. Recommend expert verification for uncertain cases
6. Support multilingual responses
"""

SYSTEM_PROMPT = """You are AgriN, an AI-powered agricultural advisory assistant.

## Your Role
You help farmers understand their crop conditions by explaining satellite-based
agricultural observations in simple, clear language.

## Critical Rules — NEVER VIOLATE THESE

1. **Use ONLY the supplied agricultural data.** Never invent NDVI values,
   rainfall amounts, crop types, soil properties, disease names, or any
   other scientific measurement.

2. **Never fabricate missing data.** If information is not provided, say
   "This information is not available" rather than guessing.

3. **Clearly distinguish between measured data and interpretation.**
   Say "The satellite data shows..." for measured values.
   Say "This may suggest..." or "This could indicate..." for interpretations.

4. **Never claim certainty when data is uncertain.** Use hedging language:
   "may", "could", "possibly", "the satellite indicators suggest".

5. **Never give specific pesticide, fertilizer, or chemical recommendations.**
   Instead say "Consult your local agricultural extension officer for
   specific treatment recommendations."

6. **Recommend local expert verification** for any disease, pest, or
   specific treatment decisions.

7. **Use simple, farmer-friendly language.** Avoid jargon. When you must
   use technical terms (like NDVI), explain them simply.

8. **Respond in the requested language.** If the user asks in Hindi,
   respond in Hindi. Match the language naturally.

9. **Explain technical indicators simply.** Example:
   NDVI → "vegetation health index — a number that shows how green
   and healthy your crop looks from satellite images"

10. **Never present AI recommendations as guaranteed results.** Always
    include a note that satellite-based observations should be verified
    in the field.

## Response Style
- Be concise but helpful
- Use bullet points for clarity
- Include emojis sparingly for visual structure (🌾 📊 💧 ✅ ⚠️)
- Structure responses with clear sections
- End with a limitation/verification note
"""


def build_advisory_prompt(analysis_data: dict, language: str = "en") -> str:
    """
    Build a prompt for generating a farm advisory from structured analysis data.
    """
    lang_instruction = ""
    if language == "hi":
        lang_instruction = "\n\n**IMPORTANT: Respond entirely in Hindi (हिन्दी). Use Devanagari script.**\n"
    elif language != "en":
        lang_instruction = f"\n\n**IMPORTANT: Respond in {language}.**\n"

    prompt = f"""Based on the following satellite-derived agricultural observations,
provide a clear, farmer-friendly advisory.{lang_instruction}

## Agricultural Data (from satellite analysis — do NOT modify these values)

- **Farm:** {analysis_data.get('farm_name', 'Unknown')}
- **Location:** {analysis_data.get('location', 'Unknown')}
- **Area:** {analysis_data.get('area_ha', 'Unknown')} hectares
- **Detected Crop:** {analysis_data.get('crop', 'Unknown')}
- **Crop Confidence:** {analysis_data.get('crop_confidence', 'N/A')}
- **Current NDVI:** {analysis_data.get('ndvi_current', 'N/A')}
- **Previous NDVI:** {analysis_data.get('ndvi_previous', 'N/A')}
- **NDVI Trend:** {analysis_data.get('ndvi_trend', 'N/A')}
- **Stress Level:** {analysis_data.get('stress_level', 'N/A')}
- **Stress Indicator:** {analysis_data.get('stress_indicator', 'N/A')}
- **Observation Date:** {analysis_data.get('observation_date', 'N/A')}
- **Data Source:** {analysis_data.get('data_source', 'N/A')}

## Your Task

Provide an advisory with these sections:
1. **Crop Status** — Brief summary of current crop condition
2. **Observation** — What the satellite data shows
3. **Possible Interpretation** — What this might mean (use hedging language)
4. **Suggested Action** — What the farmer should check or consider
5. **Confidence & Limitation** — Honest statement about data limitations

Keep it concise (under 250 words). Use simple language a farmer can understand.
"""
    return prompt


def build_qa_prompt(
    question: str,
    analysis_data: dict,
    language: str = "en",
) -> str:
    """
    Build a prompt for the Ask AgriN Q&A feature.
    """
    lang_instruction = ""
    if language == "hi":
        lang_instruction = "\n**Respond in Hindi (हिन्दी) using Devanagari script.**\n"
    elif language != "en":
        lang_instruction = f"\n**Respond in {language}.**\n"

    prompt = f"""A farmer is asking about their crop. Answer their question using
ONLY the agricultural data provided below. If the answer requires information
not available in the data, say so honestly.{lang_instruction}

## Current Farm Context (satellite-derived data)

- **Farm:** {analysis_data.get('farm_name', 'Unknown')}
- **Crop:** {analysis_data.get('crop', 'Unknown')} (confidence: {analysis_data.get('crop_confidence', 'N/A')})
- **Current NDVI:** {analysis_data.get('ndvi_current', 'N/A')}
- **Previous NDVI:** {analysis_data.get('ndvi_previous', 'N/A')}
- **NDVI Trend:** {analysis_data.get('ndvi_trend', 'N/A')}
- **Stress Level:** {analysis_data.get('stress_level', 'N/A')}
- **Observation Date:** {analysis_data.get('observation_date', 'N/A')}

## Farmer's Question

"{question}"

## Instructions

- Answer clearly and concisely
- Use simple language
- If the question is about something not covered by the data, explain what
  information IS available and suggest checking with a local expert
- Never invent data points
- Keep the answer under 200 words
"""
    return prompt


def farm_analysis_to_prompt_data(analysis) -> dict:
    """
    Convert a FarmAnalysis Pydantic object to a flat dict for prompt templates.
    """
    data = {
        "farm_name": analysis.farm.name,
        "location": f"{analysis.farm.latitude}°N, {analysis.farm.longitude}°E",
        "area_ha": analysis.farm.area_ha,
        "data_source": analysis.data_source.value.upper(),
    }

    if analysis.crop_prediction:
        data["crop"] = analysis.crop_prediction.predicted_crop.value.capitalize()
        data["crop_confidence"] = f"{analysis.crop_prediction.confidence:.0%}"
    else:
        data["crop"] = analysis.farm.crop.value.capitalize() if analysis.farm.crop else "Unknown"
        data["crop_confidence"] = "N/A"

    if analysis.stress_assessment:
        data["stress_level"] = analysis.stress_assessment.stress_level.value.capitalize()
        data["stress_indicator"] = f"{analysis.stress_assessment.indicator_value:.2f}"
    else:
        data["stress_level"] = "Unknown"
        data["stress_indicator"] = "N/A"

    data["ndvi_current"] = f"{analysis.ndvi_current:.4f}" if analysis.ndvi_current is not None else "N/A"
    data["ndvi_previous"] = f"{analysis.ndvi_previous:.4f}" if analysis.ndvi_previous is not None else "N/A"
    data["ndvi_trend"] = analysis.ndvi_trend.value.capitalize() if analysis.ndvi_trend else "N/A"
    data["observation_date"] = str(analysis.observation_date) if analysis.observation_date else "N/A"

    return data
