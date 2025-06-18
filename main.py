from fastapi import FastAPI, HTTPException
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uuid
import logging
import os
from TTS.api import TTS
import httpx

# LangChain imports
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


app = FastAPI(title="AI Voice Sales Agent for Interactive care")

# Conversation store
conversations = {}

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
    if not text.strip():
        text = "I'm sorry, I couldn't generate a response. Please try again."
    audio_dir = "Audio"
    os.makedirs(audio_dir, exist_ok=True)
    output_file = os.path.join(audio_dir, f"output_{uuid.uuid4()}.wav")
    tts.tts_to_file(text=text.strip(), file_path=output_file)
    return output_file

def speech_to_text(audio_input: str) -> str:
    return audio_input

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
            return f"Error calling Ollama: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "ollama"

ollama_llm = OllamaLLM()

# LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["history", "customer_message"],
    template="""
    History:
    {history}
    
    Customer: {customer_message}
    You are selling the AI Mastery Bootcamp ($299, 12 weeks, covers AI skills like LLMs, Computer Vision, MLOps) for Interactive Care Bangladesh. 
    Respond in 3–5 sentences: pitch the course, ask about their goals, or handle objections.
    """
    )

langchain_chain = LLMChain(llm=ollama_llm, prompt=prompt_template)

# ---------- API Endpoints ----------

@app.post("/start-call", response_model=CallResponse)
async def start_call(request: StartCallRequest):
    call_id = str(uuid.uuid4())
    first_message = f"Hello {request.customer_name}, I'm from AI Mastery Academy. Interested in AI skills?"
    conversations[call_id] = [{"role": "agent", "message": first_message}]
    return CallResponse(call_id=call_id, message="Call started", first_message=first_message)

@app.post("/respond/{call_id}", response_model=CallResponse)
async def respond(call_id: str, request: RespondRequest):
    if call_id not in conversations:
        raise HTTPException(status_code=404, detail="Call not found")

    customer_message = speech_to_text(request.message)
    conversations[call_id].append({"role": "customer", "message": customer_message})
    history = "\n".join([f"{msg['role']}: {msg['message']}" for msg in conversations[call_id]])

    try:
        response = langchain_chain.run({"history": history, "customer_message": customer_message})
    except Exception as e:
        response = "I'm sorry, I couldn't generate a response. Please try again."

    if len(response) > 800:
        response = response[:800] + "..."

    audio_path = text_to_speech(response)
    conversations[call_id].append({"role": "agent", "message": response, "audio": audio_path})
    should_end_call = "schedule a follow-up" in response.lower() or "goodbye" in customer_message.lower()
    return CallResponse(call_id=call_id, message=response, first_message=None)

@app.get("/conversation/{call_id}", response_model=ConversationResponse)
async def get_conversation(call_id: str):
    if call_id not in conversations:
        raise HTTPException(status_code=404, detail="Call not found")
    return ConversationResponse(call_id=call_id, history=conversations[call_id])


