from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
import shutil
import warnings
from gtts import gTTS
import tempfile
import base64

# moviepy emits harmless SyntaxWarning messages about invalid escape sequences
warnings.filterwarnings(
    "ignore",
    message=r".*invalid escape sequence.*",
    category=SyntaxWarning,
)
import moviepy.editor as moviepy


app = FastAPI()

# Allow all origins for development (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NVIDIA_API_KEY = "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"
 
# Load a conversational AI model from Hugging Face
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=NVIDIA_API_KEY)


@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    r = sr.Recognizer()
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Save uploaded audio file to server
    webm_path = os.path.join(uploads_dir, audio.filename)
    with open(webm_path, "wb") as out_file:
        shutil.copyfileobj(audio.file, out_file)
    print("File saved at:", webm_path)

    # Convert webm to wav
    if webm_path.endswith(".webm"):
        wav_path = webm_path + ".wav"
    else:
        wav_path = webm_path.rsplit(".", 1)[0] + ".wav"
    try:
        print("Converting webm to wav...")
        clip = moviepy.AudioFileClip(webm_path)
        clip.write_audiofile(wav_path)
        print("WAV file created at:", wav_path)
    except Exception as e:
        return JSONResponse(content={"response": f"Failed to convert webm to wav: {e}"}, status_code=500)

    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)

    try:
        # Recognize speech (Voice-to-Text)
        text = r.recognize_google(audio_data)
        print("You said:", text)

        # Create the prompt for the LLM
        prompt = f"""
You are speaking as Thumpala Lakshmi Kumari with four years of experience, a candidate interviewing for an AI Agent Team role at 100x.
Respond to each question as if you are in a real interview.

Guidelines for your answers:
- Keep answers short and natural (2–4 sentences).
- Speak in first person ("I…").
- Show confidence, curiosity, and enthusiasm.
- Highlight your experience in Python, AI, LLMs, LangChain, RAG, agentic workflows, and problem-solving.
- Be authentic — mix professional strengths with personal qualities.
- If the question is unexpected, still answer positively and relate back to your skills or growth mindset.
This is the question: {text}
"""
        # Get response from the LLM
        response = llm.invoke(prompt)
        print("LLM response:", response)

        # Prepare the text response and format it for HTML
        raw = response.content if response.content else "No response."
        formatted = raw.replace("**", "<b>").replace("**", "</b>")  # Simple formatting for bold
        lines = formatted.split('\n')
        html_lines = []

        # Build the HTML response
        for line in lines:
            if line.strip():
                html_lines.append(f"<p>{line.strip()}</p>")

        html_response = ''.join(html_lines)

        # Convert the LLM's response text to speech (Audio)
        tts = gTTS(text=raw, lang='en')
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            audio_file_path = temp_audio.name

        # Read the audio file and encode it in base64
        with open(audio_file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

        # Return both HTML response and audio in base64 format
        return JSONResponse(content={
            "response": html_response,
            "audio_base64": audio_base64
        })

    except sr.UnknownValueError:
        return JSONResponse(content={"response": "Could not understand audio"}, status_code=400)
    except sr.RequestError as e:
        return JSONResponse(content={"response": f"Could not request results from speech recognition service; {e}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
