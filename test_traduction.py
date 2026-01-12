#!/usr/bin/env python3
"""
Test ASR Dockerisé : input.mp4 → Whisper → real_speech.txt → WER normalisé
"""

import sys
import os
from pathlib import Path
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer

def main(audio_file="input.mp4", gt_file="real_speech.txt"):
    print(f"🧠 Test ASR: {audio_file} → {gt_file}")
    
    # Vérif fichiers
    if not Path(audio_file).exists():
        print(f"❌ {audio_file} manquant!")
        sys.exit(1)
    if not Path(gt_file).exists():
        print(f"❌ {gt_file} manquant!")
        sys.exit(1)
    
    # Pipeline
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    transcription = pipe(audio_file, return_timestamps=True)["chunks"]
    model_text = ' '.join([c["text"] for c in transcription])
    
    # Ground truth
    with open(gt_file, "r") as f:
        ground_truth = f.read().strip()
    
    # Normalisation
    normalizer = BasicTextNormalizer()
    model_norm = normalizer(model_text)
    gt_norm = normalizer(ground_truth)
    
    # Métriques
    wer_brut = jiwer.wer(ground_truth, model_text)
    cer_brut = jiwer.cer(ground_truth, model_text)
    wer_norm = jiwer.wer(gt_norm, model_norm)
    cer_norm = jiwer.cer(gt_norm, model_norm)
    
    # Output
    print("\n📝 MODÈLE:", model_text)
    print("📝 GT:", ground_truth)
    print(f"\n🎯 BRUT     → WER: {wer_brut:.1%} | CER: {cer_brut:.1%}")
    print(f"🎯 NORMALISÉ → WER: {wer_norm:.1%} | CER: {cer_norm:.1%}")
    
    # Verdict
    if wer_norm < 0.15:
        print("✅ EXCELLENT! ✓")
        sys.exit(0)
    elif wer_norm < 0.25:
        print("✅ BON! ✓")
        sys.exit(0)
    else:
        print("❌ À AMÉLIORER!")
        sys.exit(1)

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    gt = sys.argv[2] if len(sys.argv) > 2 else "real_speech.txt"
    main(audio, gt)
