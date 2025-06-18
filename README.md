# AI Voice Sales Agent for Interactive Care

## 🌟 Overview

This project simulates an AI-powered voice sales agent that makes automated calls to pitch the "AI Mastery Bootcamp". It supports real-time conversation handling using a large language model (LLM), generates human-like responses, and converts them into audio responses using a TTS system. Calls are fully simulated without telephony integration.

## 🤝 Features

* Start a simulated call with a customer
* AI-driven dynamic conversation flow
* Objection handling and qualification logic
* Text-to-Speech (TTS) support via Coqui TTS
* Mocked Speech-to-Text (STT)
* Conversation history tracking
* CPU-only Ollama model (qwen2:0.5b) integration

## 🚀 Tech Stack

* **FastAPI**: Backend framework
* **LangChain**: LLM orchestration
* **Ollama (qwen2:0.5b)**: Lightweight open-source LLM
* **Coqui TTS**: Open-source text-to-speech synthesis
* **httpx**: HTTP client for calling Ollama API

## 📊 Architecture Diagram

```
+--------------------------+
|  FastAPI Server          |
+--------------------------+
       |         |           
       |         |
       v         v
+------------+ +------------------+
| LangChain  | | Coqui TTS (TTS)  |
| + LLMChain | +------------------+
+------------+
       |
       v
  Ollama Server (qwen2:0.5b)

+--------------------------+
|   Simulated Conversation |
|  (State stored in memory)|
+--------------------------+
```

## 🔧 Installation

### 1. Clone the repo

```bash
git clone https://github.com/anik007-code/Voice_Agent 

```

### 2. Create virtual environment & activate

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama with the required model

```bash
ollama run qwen2:0.5b
```

### 5. Run the FastAPI server

```bash
uvicorn main:app --reload
```

## 🔐 .env Example

No external secrets needed for now. You can still create a `.env` if you later add API keys (e.g., ElevenLabs, OpenAI).

## 💡 API Documentation

You can access the interactive Swagger UI:

```
http://127.0.0.1:8000/docs
```

### POST `/start-call`

Start a new conversation.

```json
{
  "phone_number": "01712345678",
  "customer_name": "Anik"
}
```

Response:

```json
{
  "call_id": "uuid",
  "message": "Call started",
  "first_message": "Hello Anik, I'm from AI Mastery Academy. Interested in AI skills?"
}
```

### POST `/respond/{call_id}`

Send customer response and get agent reply.

```json
{
  "message": "Tell me more about the course."
}
```

### GET `/conversation/{call_id}`

Fetch conversation history.

---

## 🎥 Demo

* Audio saved in `Audio/` folder as `.wav`
* Use Swagger UI to simulate calls
* **There is a file in GitHub named audio.py, if you runthe file then you will listen the generated audio conversation.** 

---

## 🚀 Credits

* Developed by:  Md Arifur Rahman Anik
* For: Junior AI Engineer Assessment Task
* GitHub: [https://github.com/anik007-code/Voice\_Agent ](https://github.com/anik007-code/Voice_Agent )
