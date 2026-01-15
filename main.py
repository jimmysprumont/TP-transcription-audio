from fastapi import FastAPI
import gradio as gr
from transformers import pipeline
import tempfile
import os

# ===== Modèle Whisper =====
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny"
)

# Créer dossier temporaire
TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# ===== Fonction principale =====
def transcrire_audio(file):
    """
    file : fichier audio/vidéo
    """
    result = pipe(file.name, return_timestamps=True)
    chunks = result.get("chunks", [])
    text = " ".join(chunk.get("text", "") for chunk in chunks)

    tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(tmp_txt.name, "w", encoding="utf-8") as f:
        f.write(text)

    return text, tmp_txt.name

# ===== Gradio UI =====
with gr.Blocks(title="Whisper Cloud Transcription") as demo:
    gr.Markdown("## 🎙️ Transcription Audio / Vidéo (Cloud HF)")
    media_input = gr.File(label="Dépose un fichier audio ou vidéo", file_types=["audio","video"])
    btn = gr.Button("Lancer")
    output_text = gr.Textbox(label="Résultat", lines=10)
    output_file = gr.File(label="Télécharger le fichier texte")
    btn.click(transcrire_audio, inputs=media_input, outputs=[output_text, output_file])

# ===== Créer FastAPI et monter Gradio =====
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")  # Gradio disponible à la racine
