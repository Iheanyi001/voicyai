import torch
from pathlib import Path
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from scipy.io.wavfile import write
import whisper
import numpy as np

# Manually provide the corrected transcript for best results
transcript = (
    "Hello everyone and welcome to our deep dive into HTML. The very backbone of every website you've ever visited. "
    "In this session, we will break down the essential elements, structures, and tags that tell your browser what to show and how to organize it. "
    "By the end, I believe you will understand the skeleton of a web page and be ready to start coding your own."
)
print("Transcript:", transcript)

# Load pretrained checkpoints
encoder.load_model(Path("encoder/saved_models/default/encoder.pt"))
synthesizer = Synthesizer(Path("synthesizer/saved_models/default/synthesizer.pt"))
vocoder.load_model(Path("vocoder/saved_models/default/vocoder.pt"))

# 1. Encode the target speaker

wav_target = encoder.preprocess_wav("samples/target_voice.wav")
embed = encoder.embed_utterance(wav_target)

specs = synthesizer.synthesize_spectrograms([transcript], [embed])
print("Spectrogram shape:", specs[0].shape)
print("Spectrogram max/min:", specs[0].max(), specs[0].min())

wav = vocoder.infer_waveform(specs[0])
print("Waveform min/max:", wav.min(), wav.max())

# Normalize to int16 range before saving
wav = wav / np.abs(wav).max() * 32767
write("converted.wav", synthesizer.sample_rate, wav.astype(np.int16))
print("Saved converted.wav")
