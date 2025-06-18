from fastapi import FastAPI, HTTPException
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uuid
import logging
import os
from TTS.api import TTS
import httpx
from transformers import pipeline

# LangChain imports
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI(title="AI Voice Sales Agent for Interactive Care")

# Conversation store
conversations = {}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Models ----------
class StartCallRequest(BaseModel):
    phone_number: str
    customer_name: str


class RespondRequest(BaseModel):
    message: str


class CallResponse(BaseModel):
    call_id: str
    message: str
    first_message: Optional[str] = None


class ConversationResponse(BaseModel):
    call_id: str
    history: List[Dict[str, str]]


# ---------- TTS (Coqui) ----------
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)


def text_to_speech(text: str) -> str:
    """Convert text to audio using Coqui TTS."""
    if not text.strip():
        text = "I'm sorry, I couldn't generate a response. Please try again."
    audio_dir = "Audio"
    os.makedirs(audio_dir, exist_ok=True)
    output_file = os.path.join(audio_dir, f"output_{uuid.uuid4()}.wav")
    try:
        tts.tts_to_file(text=text.strip(), file_path=output_file)
        return output_file
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail="TTS processing failed")


# ---------- STT (HuggingFace Whisper) ----------
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


def speech_to_text(audio_input: str) -> str:
    """Convert audio or text input to text. For simulation, accepts text input."""
    try:
        if not audio_input:
            raise HTTPException(status_code=400, detail="No input provided")
        # For simulation, treat input as text if it’s not a valid file path
        if not os.path.exists(audio_input):
            logger.info(f"Using text input for simulation: {audio_input}")
            return audio_input
        # For real audio processing
        result = stt_pipeline(audio_input)
        return result["text"]
    except Exception as e:
        logger.error(f"STT error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT processing failed: {str(e)}")


# ---------- Sentiment Analysis ----------
sentiment_analyzer = pipeline("sentiment-analysis")


def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of customer response."""
    try:
        result = sentiment_analyzer(text)[0]
        return result["label"]
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return "NEUTRAL"


# ---------- LangChain Custom Ollama LLM ----------
class OllamaLLM(LLM):
    model: str = "qwen2:0.5b"
    base_url: str = "http://localhost:11434/api/generate"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
    ) -> str:
        try:
            response = httpx.post(
                self.base_url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            return f"Error calling Ollama: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "ollama"


ollama_llm = OllamaLLM()

# LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["history", "customer_message", "stage", "sentiment"],
    template="""
    History: {history}
    Current Stage: {stage}
    Customer Sentiment: {sentiment}
    Customer: {customer_message}

    You are an AI Voice Sales Agent for Interactive Care Bangladesh selling the AI Mastery Bootcamp ($299, 12 weeks, covers LLMs, Computer Vision, MLOps).
    Follow these stages:
    1. Introduction: Greet warmly and introduce the course (e.g., "Hi, I'm from AI Mastery Academy, where we offer a 12-week AI Bootcamp...").
    2. Qualification: Ask 2–3 questions to understand their needs (e.g., "What AI skills are you looking to develop?").
    3. Pitch: Highlight course benefits relevant to their responses (e.g., "Our bootcamp includes hands-on projects...").
    4. Objection Handling: Address concerns like price ("We have a special offer at $299"), time, or relevance.
    5. Closing: Propose a follow-up call or enrollment (e.g., "Can we schedule a call to discuss enrollment?").

    Based on the current stage and sentiment, respond in 3–5 sentences. Be friendly, professional, and adapt to the customer's sentiment.
    """
)

langchain_chain = LLMChain(llm=ollama_llm, prompt=prompt_template)


# ---------- API Endpoints ----------
@app.post("/start-call", response_model=CallResponse)
async def start_call(request: StartCallRequest):
    """Start a new call and initialize conversation."""
    call_id = str(uuid.uuid4())
    first_message = f"Hello {request.customer_name}, I'm from AI Mastery Academy. We're offering a 12-week AI Mastery Bootcamp to help you master AI skills. What’s your experience with AI so far?"
    conversations[call_id] = [{"role": "agent", "message": first_message, "stage": "introduction"}]
    logger.info(f"Started call with ID: {call_id}")
    return CallResponse(call_id=call_id, message="Call started", first_message=first_message)


@app.post("/respond/{call_id}", response_model=CallResponse)
async def respond(call_id: str, request: RespondRequest):
    """Handle customer response and generate agent reply."""
    if call_id not in conversations:
        raise HTTPException(status_code=404, detail="Call not found")

    try:
        customer_message = speech_to_text(request.message)
    except HTTPException as e:
        logger.error(f"Failed to process customer message: {str(e)}")
        raise e

    conversations[call_id].append({"role": "customer", "message": customer_message})

    # Determine current stage
    last_message = conversations[call_id][-2] if len(conversations[call_id]) > 1 else conversations[call_id][0]
    stage = last_message.get("stage", "introduction")
    if stage == "introduction":
        stage = "qualification"
    elif stage == "qualification" and len([m for m in conversations[call_id] if m["role"] == "customer"]) >= 2:
        stage = "pitch"
    elif "objection" in customer_message.lower() or "expensive" in customer_message.lower() or "time" in customer_message.lower():
        stage = "objection_handling"
    elif stage == "pitch" or stage == "objection_handling":
        stage = "closing"

    # Analyze sentiment
    sentiment = analyze_sentiment(customer_message)

    history = "\n".join([f"{msg['role']}: {msg['message']}" for msg in conversations[call_id]])

    try:
        response = langchain_chain.run({
            "history": history,
            "customer_message": customer_message,
            "stage": stage,
            "sentiment": sentiment
        })
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        response = "I'm sorry, I couldn't generate a response. Please try again."

    if len(response) > 800:
        response = response[:800] + "..."

    try:
        audio_path = text_to_speech(response)
    except HTTPException as e:
        logger.error(f"TTS failed: {str(e)}")
        raise e

    conversations[call_id].append({"role": "agent", "message": response, "audio": audio_path, "stage": stage})

    should_end_call = "schedule a follow-up" in response.lower() or "goodbye" in customer_message.lower()
    logger.info(f"Responded for call {call_id}, stage: {stage}, sentiment: {sentiment}")
    return CallResponse(call_id=call_id, message=response, first_message=None)


@app.get("/conversation/{call_id}", response_model=ConversationResponse)
async def get_conversation(call_id: str):
    """Retrieve conversation history for a given call."""
    if call_id not in conversations:
        raise HTTPException(status_code=404, detail="Call not found")
    return ConversationResponse(call_id=call_id, history=conversations[call_id])
