import librosa

def allowed_audio_duration(audio_path, user_type):
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    if user_type == "free":
        return duration <= 120  # 2 minutes
    # Paid users: allow up to 10 minutes (customize as needed)
    return duration <= 600