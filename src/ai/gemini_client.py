"""
AgriN — Gemini AI Client

Handles communication with the Google Gemini API.
Falls back to demo advisories when the API is unavailable or unconfigured.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from src.ai.prompts import (
    SYSTEM_PROMPT,
    build_advisory_prompt,
    build_qa_prompt,
    farm_analysis_to_prompt_data,
)
from src.config.settings import get_settings
from src.data.schemas import Advisory, DataSource, FarmAnalysis
from src.data.demo_data import get_demo_advisory

logger = logging.getLogger(__name__)

# Lazy import
_genai = None
_model = None
_initialized = False

# Preferred models in priority order
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]


def _init_gemini() -> bool:
    """Initialize the Gemini client with the best available model."""
    global _genai, _model, _initialized

    if _initialized:
        return _model is not None

    settings = get_settings()
    api_key = settings.gemini_api_key

    if not api_key:
        logger.warning("GEMINI_API_KEY not set — AI service will use demo responses.")
        _initialized = True
        return False

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _genai = genai

        # Find best available model
        for model_name in CANDIDATE_MODELS:
            try:
                _model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_PROMPT,
                )
                # Quick check
                logger.info(f"Gemini client initialized with model: {model_name}")
                _initialized = True
                return True
            except Exception as me:
                logger.debug(f"Model {model_name} not available: {me}")
                continue

        logger.warning("No candidate Gemini models could be initialized.")
        _initialized = True
        return False

    except ImportError:
        logger.warning("google-generativeai not installed.")
        _initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        _initialized = True
        return False


def is_gemini_available() -> bool:
    """Check if Gemini API is configured and functional."""
    return _init_gemini()


def generate_advisory(
    analysis: FarmAnalysis,
    language: str = "en",
) -> Advisory:
    """
    Generate a farm advisory using Gemini AI.
    Falls back to demo advisory if Gemini is unavailable.
    """
    prompt_data = farm_analysis_to_prompt_data(analysis)

    if not _init_gemini():
        logger.info("Gemini unavailable — returning demo advisory")
        return get_demo_advisory(analysis.farm.farm_id, language)

    try:
        prompt = build_advisory_prompt(prompt_data, language)
        response = _model.generate_content(prompt)
        advisory_text = response.text if response.text else "Advisory generation failed."

        return Advisory(
            farm_id=analysis.farm.farm_id,
            generated_at=datetime.utcnow(),
            language=language,
            observations_summary=_extract_summary(advisory_text),
            advisory_text=advisory_text,
            model_version=_model.model_name,
            data_source=DataSource.LIVE,
        )

    except Exception as e:
        logger.error(f"Gemini advisory generation failed: {e}")
        return get_demo_advisory(analysis.farm.farm_id, language)


def ask_question(
    question: str,
    analysis: FarmAnalysis,
    language: str = "en",
) -> str:
    """
    Answer a farmer's question using the current farm context.
    Falls back to static response if Gemini is unavailable.
    """
    prompt_data = farm_analysis_to_prompt_data(analysis)

    if not _init_gemini():
        return _demo_qa_response(question, prompt_data, language)

    try:
        prompt = build_qa_prompt(question, prompt_data, language)
        response = _model.generate_content(prompt)
        return response.text if response.text else "I couldn't generate a response. Please try again."

    except Exception as e:
        logger.error(f"Gemini Q&A failed: {e}")
        return _demo_qa_response(question, prompt_data, language)


def _extract_summary(advisory_text: str) -> str:
    """Extract the first meaningful paragraph as a summary."""
    lines = [l.strip() for l in advisory_text.split("\n") if l.strip()]
    for line in lines:
        if not line.startswith("#") and not line.startswith("**") and len(line) > 20:
            return line[:200]
    return lines[0][:200] if lines else "Advisory generated."


def _demo_qa_response(question: str, data: dict, language: str) -> str:
    """Generate a static demo response when Gemini is unavailable."""
    if language == "hi":
        return (
            f"🔶 **डेमो प्रतिक्रिया** (Gemini API कॉन्फ़िगर नहीं है)\n\n"
            f"आपका प्रश्न: \"{question}\"\n\n"
            f"वर्तमान फसल की स्थिति:\n"
            f"- फसल: {data.get('crop', 'अज्ञात')}\n"
            f"- तनाव स्तर: {data.get('stress_level', 'अज्ञात')}\n"
            f"- NDVI रुझान: {data.get('ndvi_trend', 'अज्ञात')}\n\n"
            f"Gemini AI सक्रिय होने पर, AgriN आपके प्रश्न का विस्तृत उत्तर देगा।"
        )

    return (
        f"🔶 **Demo Response** (Gemini API not configured)\n\n"
        f"Your question: \"{question}\"\n\n"
        f"Current farm context:\n"
        f"- Crop: {data.get('crop', 'Unknown')}\n"
        f"- Stress Level: {data.get('stress_level', 'Unknown')}\n"
        f"- NDVI Trend: {data.get('ndvi_trend', 'Unknown')}\n\n"
        f"When Gemini AI is active, AgriN will provide a detailed, "
        f"context-aware answer to your question based on current satellite observations."
    )
