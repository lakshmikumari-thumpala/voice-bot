from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
import shutil
import warnings

# moviepy emits harmless SyntaxWarning messages about invalid escape sequences
# (these come from third-party package code on import). Silence those specific
# warnings so logs are cleaner. We add this filter before importing moviepy.
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
llm = ChatNVIDIA(model="llama-3.2-nemoretriever-300m-embed-v2", api_key=NVIDIA_API_KEY)


@app.post("/chat")
async def chat(audio: UploadFile = File(...)):
    r = sr.Recognizer()
    # Ensure uploads directory exists
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Save the uploaded file under uploads folder
    webm_path = os.path.join(uploads_dir, audio.filename)
    with open(webm_path, "wb") as out_file:
        shutil.copyfileobj(audio.file, out_file)
    print("thisiis the path", webm_path)
    # Convert webm to wav using moviepy
    if webm_path.endswith(".webm"):
        wav_path = webm_path + ".wav"
    else:
        wav_path = webm_path.rsplit(".", 1)[0] + ".wav"
    try:
        print("Converting webm to wav...")
        clip = moviepy.AudioFileClip(webm_path)
        print("Audio duration:",)
        clip.write_audiofile(wav_path)
        print("WAV file created at:", wav_path)
    except Exception as e:
        return JSONResponse(content={"response": f"Failed to convert webm to wav: {e}"}, status_code=500)

    # Now use wav_path with speech_recognition
    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)

    # 1. Voice to Text
    try:
        text = r.recognize_google(audio_data)  # Or another STT engine
        print("You said:", text)
        # 2. Text Processing with LLM
        prompt  = f"""
You are speaking as Thumpala Lakshmi Kumari with four years of experience, a candidate interviewing for an AI Agent Team role at 100x.
Respond to each question as if you are in a real interview.

Guidelines for your answers:
- Keep answers short and natural (2–4 sentences).
- Speak in first person ("I…").
- Show confidence, curiosity, and enthusiasm.
- Highlight your experience in Python, AI, LLMs, LangChain, RAG, agentic workflows, and problem-solving.
- Be authentic — mix professional strengths with personal qualities.
- If the question is unexpected, still answer positively and relate back to your skills or growth mindset.
This is the questions {text}
"""
        response = llm.invoke(prompt)
        print("LLM response:", response)
        # Return as HTML for frontend rendering
        # Format the LLM response for better display
        import re
        raw = response.content if response.content else "No response."
        # Convert markdown bold (**text**) to <b>text</b>
        formatted = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", raw)
        lines = formatted.split('\n')
        html_lines = []
        in_ol = False
        in_ul = False
        for line in lines:
            num_list_match = re.match(r"^\s*\*?\*?(\d+)\.\*?\*?\s*(.*)", line)
            bullet_match = re.match(r"^\s*[-\*]\s*(.*)", line)
            if num_list_match:
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                if not in_ol:
                    html_lines.append('<ol>')
                    in_ol = True
                html_lines.append(f"<li>{num_list_match.group(2).strip()}</li>")
            elif bullet_match:
                if in_ol:
                    html_lines.append('</ol>')
                    in_ol = False
                if not in_ul:
                    html_lines.append('<ul>')
                    in_ul = True
                html_lines.append(f"<li>{bullet_match.group(1).strip()}</li>")
            else:
                if in_ol:
                    html_lines.append('</ol>')
                    in_ol = False
                if in_ul:
                    html_lines.append('</ul>')
                    in_ul = False
                if line.strip():
                    html_lines.append(f"<p>{line.strip()}</p>")
        if in_ol:
            html_lines.append('</ol>')
        if in_ul:
            html_lines.append('</ul>')
        html_response = '\n'.join(html_lines)
        scrollable_html = f"""
<div id="scrollable-response" style="
  height: 500px;
  overflow-y: auto;
  padding: 20px;
  box-sizing: border-box;
  border: 1px solid #ccc;
  background-color: #fff;
  scroll-behavior: smooth;
">
{html_response}
</div>

<script>
  // Ensure scrolling happens *after* content loads
  window.onload = function() {{
      setTimeout(function() {{
          var el = document.getElementById('scrollable-response');
          if (el) {{
              el.scrollTop = el.scrollHeight;
          }}
      }}, 300); // small delay ensures rendering completes
  }};
</script>
"""


    except sr.UnknownValueError:
        return JSONResponse(content={"response": "Could not understand audio"}, status_code=400)
    except sr.RequestError as e:
        return JSONResponse(content={"response": f"Could not request results from speech recognition service; {e}"}, status_code=500)

    return {"response": scrollable_html}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
