from fastapi import FastAPI, UploadFile, File
from transformers import pipeline

app = FastAPI()

# Charger le pipeline une seule fois (performance)
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny"
)


@app.post("/transcrire")
async def transcribe_audio(file: UploadFile = File(...)):
    # Enregistrer temporairement le fichier
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())


    # Transcrire
    result = pipe(temp_path, return_timestamps=True)

    return {"transcription": result.get("chunks")}


