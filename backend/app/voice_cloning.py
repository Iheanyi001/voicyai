from pathlib import Path
import numpy as np
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
import vocoder.inference as vocoder
from scipy.io.wavfile import write
import whisper
import os
import logging
from nltk.tokenize import sent_tokenize
import wave
import contextlib
from utils.logmmse import profile_noise, denoise
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
from datetime import datetime
import shutil
import nltk
from scipy import signal
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()

def get_cached_model(model_name):
    """Get or create a cached TTS model instance."""
    with _model_lock:
        if model_name not in _model_cache:
            logger.info(f"Loading model {model_name} into cache")
            from TTS.api import TTS
            _model_cache[model_name] = TTS(model_name=model_name)
        return _model_cache[model_name]

# --- OVERRIDE TORCH.LOAD TO DISABLE WEIGHTS_ONLY SECURITY ---
import torch
# Monkey-patch torch.load to force weights_only=False for trusted TTS checkpoints
_orig_torch_load = torch.load

def _patched_torch_load(f, *args, weights_only=True, **kwargs):
    return _orig_torch_load(f, *args, weights_only=False, **kwargs)

torch.load = _patched_torch_load

# Set NLTK_DATA to a writable directory
os.environ['NLTK_DATA'] = str(Path.home() / 'nltk_data')
try:
    nltk.download('punkt', quiet=True, force=True)
    nltk.download('punkt_tab', quiet=True, force=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

@lru_cache(maxsize=32)
def get_transcript(audio_path, provided_transcript=None):
    """Get transcript from provided text or generate using Whisper."""
    if provided_transcript:
        logger.info("Using provided transcript")
        return provided_transcript.strip()
    
    try:
        logger.info("Generating transcript using Whisper")
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        transcript = result["text"].strip()
        if not transcript:
            raise RuntimeError("Whisper generated empty transcript")
        return transcript
    except Exception as e:
        logger.error(f"Error generating transcript: {e}")
        raise RuntimeError(f"Failed to generate transcript: {str(e)}")

def split_text_into_chunks(text, max_chunk_size=250):
    """Split text into chunks using regex-based sentence detection for better performance."""
    # Define sentence boundary regex pattern
    # This matches common sentence endings followed by spaces and capital letters
    sentence_pattern = r'[.!?]+[\s]+(?=[A-Z])|[.!?]+[\s]*$'
    
    # First, clean up any excessive whitespace
    text = ' '.join(text.split())
    
    # Initialize variables
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Split into rough sentences first
    # This creates a list of sentence-like segments
    rough_sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
    
    # Add back the sentence endings that were removed by the split
    sentences = []
    text_remaining = text
    for sentence in rough_sentences:
        # Find the sentence in the remaining text
        start_idx = text_remaining.find(sentence)
        if start_idx != -1:
            # Look for the end of this sentence
            next_start = start_idx + len(sentence)
            end_match = re.search(sentence_pattern, text_remaining[next_start:])
            if end_match:
                # Include the ending punctuation and space
                sentence = text_remaining[start_idx:next_start + end_match.end()]
            text_remaining = text_remaining[next_start:]
        sentences.append(sentence)
    
    # Process each sentence
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If a single sentence is longer than max_chunk_size,
        # split it by commas or other natural breaks
        if sentence_size > max_chunk_size:
            # Split by commas, semicolons, or dashes
            subparts = re.split(r'[,;–—]\s*', sentence)
            for part in subparts:
                part = part.strip()
                if not part:
                    continue
                    
                part_size = len(part)
                if current_size + part_size > max_chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(part)
                current_size += part_size
        else:
            # Normal sentence processing
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_chunk(args):
    """Process a single text chunk with TTS."""
    chunk, target_voice_path, temp_output, model_name = args
    try:
        tts = get_cached_model(model_name)
        
        # Add explicit end markers to ensure the model stops
        processed_chunk = chunk.strip()
        if not processed_chunk.endswith(('.', '!', '?')):
            processed_chunk += '.'
        processed_chunk += ' '  # Add space after punctuation
        
        try:
            # First try with the advanced parameters
            tts.tts_to_file(
                text=processed_chunk,
                speaker_wav=str(target_voice_path),
                language="en",
                file_path=str(temp_output),
                speed=0.9,
                temperature=0.65
                # Removed potentially unsupported parameters
            )
        except Exception as e:
            logger.warning(f"Error with advanced parameters: {e}. Falling back to basic parameters.")
            # Fall back to basic parameters if the advanced ones cause errors
            tts.tts_to_file(
                text=processed_chunk,
                speaker_wav=str(target_voice_path),
                language="en",
                file_path=str(temp_output)
            )
        return temp_output
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        raise

def combine_wav_files(input_files, output_file):
    """Combine multiple WAV files into one with improved audio quality."""
    try:
        # Get parameters from first file
        with contextlib.closing(wave.open(str(input_files[0]), 'rb')) as first_wav:
            params = first_wav.getparams()
        
        # Combine all files
        with wave.open(str(output_file), 'wb') as output_wav:
            output_wav.setparams(params)
            
            # Process each file with improved audio quality
            for wav_file in input_files:
                with contextlib.closing(wave.open(str(wav_file), 'rb')) as w:
                    # Read frames
                    frames = w.readframes(w.getnframes())
                    
                    # Convert to numpy array for processing
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    
                    # Apply a gentle high-pass filter to reduce low-frequency noise
                    nyquist = 0.5 * params[2]  # Half the sample rate
                    cutoff = 70 / nyquist  # High-pass at 70Hz
                    b, a = signal.butter(3, cutoff, btype='high')
                    audio_data = signal.filtfilt(b, a, audio_data).astype(np.int16)
                    
                    # Normalize audio to use full dynamic range
                    audio_data = audio_data / np.abs(audio_data).max() * 32766
                    
                    # Apply gentle compression for more consistent volume
                    threshold = 0.6
                    ratio = 0.7
                    knee = 0.1
                    
                    # Convert to float for processing
                    audio_float = audio_data.astype(np.float32) / 32767.0
                    
                    # Calculate signal level
                    signal_level = np.abs(audio_float)
                    
                    # Apply soft knee compression
                    mask_above = signal_level > (threshold - knee)
                    mask_below = signal_level <= (threshold - knee)
                    
                    # Below threshold: no compression
                    compressed = audio_float.copy()
                    
                    # Above threshold: apply compression
                    knee_range = mask_above & (signal_level < (threshold + knee))
                    above_range = signal_level >= (threshold + knee)
                    
                    # Soft knee compression
                    if np.any(knee_range):
                        knee_factor = ((signal_level[knee_range] - (threshold - knee)) / (2 * knee)) ** 2
                        gain_reduction = (1 - ratio) * knee_factor
                        compressed[knee_range] = audio_float[knee_range] * (1.0 - gain_reduction)
                    
                    # Full compression
                    if np.any(above_range):
                        gain_reduction = (signal_level[above_range] - threshold) * (1 - ratio)
                        compressed[above_range] = (audio_float[above_range] - gain_reduction * np.sign(audio_float[above_range]))
                    
                    # Convert back to int16
                    audio_data = (compressed * 32767).astype(np.int16)
                    
                    # Add a very subtle reverb effect for more natural sound
                    reverb_time = 0.1
                    sample_rate = params[2]
                    reverb_samples = int(reverb_time * sample_rate)
                    
                    if reverb_samples > 0:
                        reverb = np.zeros_like(audio_data, dtype=np.float32)
                        decay = np.exp(-6.0 * np.arange(reverb_samples) / reverb_samples)
                        
                        # Apply reverb with very low mix level
                        for i in range(len(audio_data)):
                            if i < reverb_samples:
                                continue
                            # Add a very subtle reverb (5% mix)
                            reverb[i] = 0.05 * np.sum(audio_data[i-reverb_samples:i].astype(np.float32) * decay)
                        
                        # Mix reverb with original
                        audio_data = (audio_data.astype(np.float32) + reverb).astype(np.int16)
                    
                    # Write processed frames
                    output_wav.writeframes(audio_data.tobytes())
                    
    except Exception as e:
        logger.error(f"Error combining WAV files: {str(e)}")
        raise

def preprocess_text(text):
    """Preprocess text to ensure it's compatible with XTTS model."""
    # Remove any special characters that might cause issues
    text = text.replace('"', '').replace('"', '').replace('"', '')
    text = text.replace(''', '').replace(''', '')
    text = text.replace('–', '-').replace('—', '-')
    
    # Ensure proper sentence endings
    text = text.strip()
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    # Add natural pauses after punctuation
    text = text.replace('.', '. ')
    text = text.replace('!', '! ')
    text = text.replace('?', '? ')
    text = text.replace(',', ', ')
    
    # Remove multiple spaces and normalize whitespace
    text = ' '.join(text.split())
    
    # Add slight pauses between sentences for more natural speech
    text = text.replace('. ', '.  ')
    text = text.replace('! ', '!  ')
    text = text.replace('? ', '?  ')
    
    # Add explicit end marker
    text = text.strip() + ' '  # Ensure space after final punctuation
    
    return text

def process_voice_cloning_xtts(audio_path, target_voice_path, user_type, transcript):
    try:
        start_time = datetime.now()
        logger.info("Starting voice cloning process...")
        
        # Get project root directory
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "output"
        temp_dir = project_root / "temp"
        
        # Create directories if they don't exist
        output_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.wav"
        output_path = output_dir / output_filename
        
        # Get or create TTS model with caching
        logger.info("Loading TTS model...")
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts = get_cached_model(model_name)
        
        # Handle target voice path
        if user_type == "free" or not target_voice_path:
            # Use default voice for free users or when no target voice is provided
            default_voice = project_root / "samples/target_voice.wav"
            if not default_voice.exists():
                raise RuntimeError("Default voice file not found")
            target_voice_path = str(default_voice)
        
        # Analyze target voice to optimize TTS parameters
        logger.info("Analyzing target voice characteristics...")
        import librosa
        
        # Load target voice audio with original sample rate
        y_target, sr_target = librosa.load(target_voice_path, sr=None)
        
        # Enhanced speech rate detection
        onset_env = librosa.onset.onset_strength(y=y_target, sr=sr_target, 
                                               hop_length=256)  # Increased precision
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_target,
                                         hop_length=256)
        
        # Calculate speech rate based on tempo and syllable detection
        syllables = librosa.feature.rms(y=y_target, hop_length=256)
        syllable_peaks = librosa.util.peak_pick(x=syllables[0], pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.3, wait=5)
        speech_duration = librosa.get_duration(y=y_target, sr=sr_target)
        speech_rate = len(syllable_peaks) / speech_duration
        
        # More natural speed calculation
        speed_param = 1.0  # Start with neutral speed
        if speech_rate > 4.0:  # Fast speech
            speed_param = 0.95
        elif speech_rate < 2.5:  # Slow speech
            speed_param = 1.05
        
        # Analyze pitch characteristics
        pitches, magnitudes = librosa.piptrack(y=y_target, sr=sr_target)
        pitch_values = pitches[magnitudes > magnitudes.mean()]
        
        if len(pitch_values) > 0:
            mean_pitch = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_variability = pitch_std / mean_pitch if mean_pitch > 0 else 0.15
        else:
            pitch_variability = 0.15  # Default if pitch detection fails
        
        # Calculate temperature based on pitch variability
        # More conservative temperature range
        temp_param = 0.6  # Start with stable temperature
        if pitch_variability > 0.2:  # Highly variable pitch
            temp_param = min(0.7, 0.6 + pitch_variability * 0.3)
        elif pitch_variability < 0.1:  # Very stable pitch
            temp_param = max(0.5, 0.6 - (0.1 - pitch_variability) * 0.3)
        
        logger.info(f"Voice analysis results - Speech rate: {speech_rate:.2f}, "
                   f"Speed: {speed_param:.2f}, Temperature: {temp_param:.2f}")
        
        # Process text with natural chunking
        processed_transcript = preprocess_text(transcript)
        chunks = split_text_into_chunks(processed_transcript, max_chunk_size=150)  # Smaller chunks
        
        # Process chunks with optimized parameters
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_output = temp_dir / f"chunk_{i}.wav"
            
            try:
                # Generate speech with optimized parameters
                tts.tts_to_file(
                    text=chunk,
                    speaker_wav=target_voice_path,
                    language="en",
                    file_path=str(chunk_output),
                    speed=speed_param,
                    temperature=temp_param
                )
                
                if chunk_output.exists():
                    # Enhance audio quality
                    enhance_audio_quality(chunk_output)
                    chunk_files.append(chunk_output)
                    
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {e}")
                continue
        
        if not chunk_files:
            raise RuntimeError("No chunks were successfully processed")
        
        # Combine chunks with smooth transitions
        logger.info("Combining audio chunks with smooth transitions...")
        combine_chunks_smoothly(chunk_files, output_path)
        
        # Clean up temporary files
        for chunk_file in chunk_files:
            try:
                chunk_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {chunk_file}: {e}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error in voice cloning: {str(e)}")
        raise

def enhance_audio_quality(audio_file):
    """Apply targeted enhancements to improve audio clarity while maintaining naturalness."""
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(str(audio_file), sr=None)
        
        # 1. More gentle noise reduction
        noise_sample = y[:int(sr*0.1)] if len(y) > sr*0.1 else y[:int(len(y)*0.1)]
        noise_profile = profile_noise(noise_sample, sr)
        y_clean = denoise(y, noise_profile, noise_reduce_percent=0.70)  # Reduced from default
        
        # 2. Subtle high-pass filter (raised from 80Hz to 60Hz for more warmth)
        from scipy import signal
        cutoff = 60 / (0.5 * sr)  # 60Hz high-pass
        b, a = signal.butter(2, cutoff, btype='high')  # Reduced order from 3 to 2
        y_filtered = signal.filtfilt(b, a, y_clean)
        
        # 3. Very gentle normalization
        y_norm = librosa.util.normalize(y_filtered) * 0.92  # Reduced from 0.95
        
        # 4. More natural formant preservation
        # Gentler EQ boost in the 1-3kHz range (reduced from 1-4kHz)
        b_eq, a_eq = signal.butter(2, [1000/(sr/2), 3000/(sr/2)], btype='bandpass')
        y_eq = signal.filtfilt(b_eq, a_eq, y_norm)
        
        # Mix with original (80% original, 20% enhanced - more original)
        y_enhanced = 0.8 * y_norm + 0.2 * y_eq
        
        # 5. More subtle reverb
        reverb_length = int(sr * 0.03)  # Reduced from 0.05 to 0.03 (30ms)
        reverb = np.zeros_like(y_enhanced)
        decay = np.exp(-8.0 * np.arange(reverb_length) / reverb_length)  # Faster decay
        
        # Apply minimal reverb
        for i in range(len(y_enhanced)):
            if i < reverb_length:
                continue
            # Reduced reverb mix to 3%
            reverb[i] = 0.03 * np.sum(y_enhanced[i-reverb_length:i] * decay)
        
        # Final output with reverb
        y_final = y_enhanced + reverb
        
        # Gentle final normalization
        y_final = librosa.util.normalize(y_final) * 0.92
        
        # Save enhanced audio
        sf.write(str(audio_file), y_final, sr)
        
    except Exception as e:
        logger.warning(f"Audio enhancement error: {str(e)}")
        # Continue with original file if enhancement fails

def combine_chunks_smoothly(input_files, output_file):
    """Combine audio chunks with smooth crossfades between segments."""
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        # Read all audio files
        audio_segments = []
        sample_rate = None
        
        for file_path in input_files:
            y, sr = librosa.load(str(file_path), sr=sample_rate)
            
            # Use first file's sample rate for all files
            if sample_rate is None:
                sample_rate = sr
            
            audio_segments.append(y)
        
        if not audio_segments:
            raise RuntimeError("No audio segments to combine")
        
        # Define crossfade length (100ms)
        crossfade_length = int(sample_rate * 0.1)
        
        # Combine with crossfades
        combined = audio_segments[0]
        
        for i in range(1, len(audio_segments)):
            # Get current segment
            segment = audio_segments[i]
            
            # If segment or combined is too short, just append
            if len(combined) < crossfade_length or len(segment) < crossfade_length:
                combined = np.concatenate([combined, segment])
                continue
            
            # Create crossfade weights
            fade_out = np.linspace(1, 0, crossfade_length)
            fade_in = np.linspace(0, 1, crossfade_length)
            
            # Apply crossfade
            combined_end = combined[-crossfade_length:]
            segment_start = segment[:crossfade_length]
            
            # Mix the overlapping parts
            crossfade = (combined_end * fade_out) + (segment_start * fade_in)
            
            # Combine everything
            combined = np.concatenate([combined[:-crossfade_length], crossfade, segment[crossfade_length:]])
        
        # Final normalization
        combined = librosa.util.normalize(combined) * 0.95
        
        # Save combined audio
        sf.write(str(output_file), combined, sample_rate)
        
    except Exception as e:
        logger.error(f"Error combining audio chunks: {str(e)}")
        
        # Fallback to basic combination if smooth technique fails
        try:
            import wave
            import contextlib
            
            with wave.open(str(output_file), 'wb') as output_wav:
                # Use first file for parameters
                with contextlib.closing(wave.open(str(input_files[0]), 'rb')) as first_wav:
                    params = first_wav.getparams()
                    output_wav.setparams(params)
                
                # Concatenate all files directly
                for wav_file in input_files:
                    with contextlib.closing(wave.open(str(wav_file), 'rb')) as w:
                        output_wav.writeframes(w.readframes(w.getnframes()))
                        
        except Exception as fallback_error:
            logger.error(f"Fallback audio combination also failed: {str(fallback_error)}")
            raise

def final_enhance_audio(output_file):
    """Apply final enhancement to the complete audio file."""
    try:
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(str(output_file), sr=None)
        
        # Normalize levels
        y_norm = librosa.util.normalize(y) * 0.95
        
        # Dynamic range compression for consistent volume
        # Simple peak compression
        threshold = 0.6
        ratio = 3.0  # Higher ratio for more aggressive compression
        attack = int(sr * 0.005)  # 5ms attack
        release = int(sr * 0.05)  # 50ms release
        
        y_compressed = np.zeros_like(y_norm)
        env = np.zeros_like(y_norm)
        
        # Simple envelope follower with attack/release
        for i in range(len(y_norm)):
            # Calculate current level (absolute value)
            level = abs(y_norm[i])
            
            # Apply attack/release
            if level > env[i-1] if i > 0 else 0:
                # Attack phase - fast rise
                if i > 0:
                    env[i] = env[i-1] + (level - env[i-1]) / attack
                else:
                    env[i] = level
            else:
                # Release phase - slow fall
                if i > 0:
                    env[i] = env[i-1] + (level - env[i-1]) / release
                else:
                    env[i] = level
        
        # Apply compression
        for i in range(len(y_norm)):
            if env[i] > threshold:
                # Calculate gain reduction
                gain_reduction = 1.0 - ((env[i] - threshold) * (1.0 - 1.0/ratio) / env[i])
                y_compressed[i] = y_norm[i] * gain_reduction
            else:
                y_compressed[i] = y_norm[i]
        
        # Final maximizing (make it louder while preventing clipping)
        y_final = librosa.util.normalize(y_compressed) * 0.95
        
        # Save enhanced final audio
        sf.write(str(output_file), y_final, sr)
        
    except Exception as e:
        logger.warning(f"Final audio enhancement error: {str(e)}")
        # Continue with original file if enhancement fails

# Keep the original process_voice_cloning function for backward compatibility
def process_voice_cloning(audio_path, target_voice_path, user_type):
    """Legacy voice cloning function using the original model."""
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        ENCODER_MODEL_PATH = PROJECT_ROOT / "encoder/saved_models/default/encoder.pt"
        SYNTHESIZER_MODEL_PATH = PROJECT_ROOT / "synthesizer/saved_models/default/synthesizer.pt"
        VOCODER_MODEL_PATH = PROJECT_ROOT / "vocoder/saved_models/default/vocoder.pt"
        DEFAULT_TARGET_VOICE = PROJECT_ROOT / "samples/target_voice.wav"
        OUTPUT_PATH = PROJECT_ROOT / "backend/uploads/converted.wav"

        logger.info("Loading models")
        encoder.load_model(ENCODER_MODEL_PATH)
        synthesizer = Synthesizer(SYNTHESIZER_MODEL_PATH)
        vocoder.load_model(VOCODER_MODEL_PATH)

        # Handle target voice
        if user_type == "free" or not target_voice_path:
            target_voice_path = DEFAULT_TARGET_VOICE
        else:
            target_voice_path = PROJECT_ROOT / target_voice_path \
                if not str(target_voice_path).startswith(str(PROJECT_ROOT)) \
                else Path(target_voice_path)

        logger.info("Processing target voice")
        wav_target = encoder.preprocess_wav(str(target_voice_path))
        embed = encoder.embed_utterance(wav_target)

        # Get transcript
        transcript = get_transcript(str(PROJECT_ROOT / audio_path))
        logger.info(f"Using transcript: {transcript[:100]}...")

        # Process audio
        if transcript:
            logger.info("Processing with transcript")
            sentences = sent_tokenize(transcript)
            specs = []
            for sentence in sentences:
                if sentence.strip():
                    s = synthesizer.synthesize_spectrograms([sentence.strip()], [embed])[0]
                    specs.append(s)
            pause_frames = np.zeros((specs[0].shape[0], 20), dtype=np.float32)
            full_spec = specs[0]
            for s in specs[1:]:
                full_spec = np.concatenate((full_spec, pause_frames, s), axis=1)
            specs = [full_spec]
        else:
            logger.info("Processing without transcript")
            wav_input = encoder.preprocess_wav(str(PROJECT_ROOT / audio_path))
            mel = Synthesizer.make_spectrogram(wav_input)
            specs = [mel]

        logger.info("Generating waveform")
        wav = vocoder.infer_waveform(specs[0])
        sample_rate = synthesizer.sample_rate
        
        logger.info("Applying noise reduction")
        noise_len = int(sample_rate * 0.5)
        noise_profile = profile_noise(wav[:noise_len], sample_rate)
        wav = denoise(wav, noise_profile)
        wav = wav / np.abs(wav).max() * 32767
        
        logger.info("Saving output")
        write(str(OUTPUT_PATH), sample_rate, wav.astype(np.int16))
        return OUTPUT_PATH
        
    except Exception as e:
        logger.error(f"Error in voice cloning: {e}")
        raise RuntimeError(f"Voice cloning failed: {str(e)}")

def train_custom_voice_model(audio_files, voice_name, user_email):
    """
    Train a custom voice model from user audio files.
    
    Args:
        audio_files: List of Path objects pointing to audio files
        voice_name: Name for the voice model
        user_email: User's email for model organization
        
    Returns:
        Path to the trained model
    """
    try:
        logger.info(f"Starting training for voice model '{voice_name}'")
        start_time = datetime.now()
        
        # Get project root directory
        project_root = Path(__file__).resolve().parents[2]
        
        # Create models directory for this user
        models_dir = project_root / "models" / user_email
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a unique model path
        model_path = models_dir / f"{voice_name}.pth"
        
        # Create temporary processing directory
        temp_dir = project_root / "temp" / f"train_{voice_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Get TTS model for encoder
        logger.info("Loading TTS model...")
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts = get_cached_model(model_name)
        
        logger.info(f"Processing {len(audio_files)} audio files for training")
        
        # Process each audio file
        processed_files = []
        for i, audio_file in enumerate(audio_files):
            try:
                # Prepare output path for processed audio
                output_file = temp_dir / f"processed_{i}_{audio_file.name}"
                
                # Process the audio file (resample, normalize, etc.)
                logger.info(f"Processing audio file {i+1}/{len(audio_files)}: {audio_file.name}")
                
                # 1. Convert audio to WAV format if necessary
                if audio_file.suffix.lower() != '.wav':
                    import subprocess
                    temp_wav = temp_dir / f"temp_{i}.wav"
                    subprocess.run([
                        'ffmpeg', '-y', '-i', str(audio_file), 
                        '-acodec', 'pcm_s16le', '-ar', '22050', 
                        str(temp_wav)
                    ], check=True, capture_output=True)
                    audio_file = temp_wav
                
                # 2. Process the audio for better quality voice model
                import librosa
                import soundfile as sf
                
                # Load audio
                y, sr = librosa.load(str(audio_file), sr=16000)
                
                # Trim silence more aggressively for training data
                y, _ = librosa.effects.trim(y, top_db=30)
                
                # Normalize audio
                y = librosa.util.normalize(y)
                
                # Remove background noise
                # High-pass filter to remove low rumble
                cutoff = 80 / (0.5 * sr)  # 80Hz high-pass filter
                b, a = signal.butter(3, cutoff, btype='high')
                y = signal.filtfilt(b, a, y)
                
                # Noise reduction
                if len(y) > sr:  # If we have at least 1 second of audio
                    noise_sample = y[:int(sr*0.2)]  # Use first 200ms as noise profile
                    noise_profile = profile_noise(noise_sample, sr)
                    y = denoise(y, noise_profile)
                
                # Apply compression for more consistent volume
                y_abs = np.abs(y)
                threshold = 0.5
                ratio = 0.7
                y_compressed = np.zeros_like(y)
                mask = y_abs <= threshold
                y_compressed[mask] = y[mask]
                y_compressed[~mask] = np.sign(y[~mask]) * (
                    threshold + (y_abs[~mask] - threshold) * ratio
                )
                
                # Save processed audio
                sf.write(str(output_file), y_compressed, sr, format='WAV')
                
                processed_files.append(output_file)
                logger.info(f"Processed audio file {i+1}: {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing audio file {audio_file.name}: {str(e)}")
                # Continue with other files
        
        if not processed_files:
            raise RuntimeError("No audio files were successfully processed")
        
        # Concatenate all processed audio files to create a more robust voice profile
        concatenated_audio = []
        for file in processed_files:
            y, sr = librosa.load(str(file), sr=16000)
            
            # Add a short silence between utterances
            silence = np.zeros(int(sr * 0.5))  # 0.5 second silence
            concatenated_audio.append(y)
            concatenated_audio.append(silence)
        
        # Combine all audio
        final_audio = np.concatenate(concatenated_audio)
        
        # Final normalization
        final_audio = librosa.util.normalize(final_audio)
        
        # Create the voice profile
        profile_path = models_dir / f"{voice_name}.wav"
        sf.write(str(profile_path), final_audio, sr, format='WAV')
        
        logger.info(f"Voice profile created at {profile_path}")
        
        # Clean up temporary files
        logger.info("Cleaning up temporary files")
        for file in processed_files:
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {file}: {str(e)}")
        
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to delete temporary directory {temp_dir}: {str(e)}")
        
        # Calculate and log total processing time
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Voice model trained successfully in {total_time:.2f} seconds")
        
        return profile_path
        
    except Exception as e:
        logger.error(f"Error in voice model training: {str(e)}")
        raise

def use_custom_voice_model(model_name, user_email, text):
    """
    Use a custom voice model to generate speech.
    
    Args:
        model_name: Name of the voice model
        user_email: User's email
        text: Text to convert to speech
        
    Returns:
        Path to the generated audio file
    """
    try:
        # Get project root directory
        project_root = Path(__file__).resolve().parents[2]
        
        # Get model path
        models_dir = project_root / "models" / user_email
        voice_profile = models_dir / f"{model_name}.wav"
        
        if not voice_profile.exists():
            raise RuntimeError(f"Voice model '{model_name}' not found")
        
        # Generate output path
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"custom_voice_{timestamp}.wav"
        
        # Process the text with the custom voice
        logger.info(f"Generating speech with custom voice model '{model_name}'")
        
        return process_voice_cloning_xtts(
            None,  # No source audio
            str(voice_profile),
            "paid",  # Custom voices are for paid users
            text
        )
        
    except Exception as e:
        logger.error(f"Error using custom voice model: {str(e)}")
        raise
