from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from app.voice_cloning import process_voice_cloning_xtts, train_custom_voice_model
from app.auth import get_user_from_token
import os
from pathlib import Path
import sqlite3
from passlib.context import CryptContext
from jose import jwt
from fastapi import status
import shutil
from datetime import datetime
import whisper
from pydantic import BaseModel
from typing import Optional
import re

SECRET_KEY = "your-secret-key"  # Should match the one in auth.py
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.db")

router = APIRouter()

class TextToSpeechRequest(BaseModel):
    text: str
    targetVoice: Optional[str] = None
    userType: str = "free"

class VoiceConversionRequest(BaseModel):
    sourceAudio: str
    targetVoice: Optional[str] = None
    userType: str = "free"
    transcript: str

class VoiceTrainingRequest(BaseModel):
    audioFiles: list
    voiceName: str
    userType: str = "paid"

@router.post("/api/register")
def register(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    print(f"Register called with: name={name}, email={email}")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            user_type TEXT NOT NULL DEFAULT 'free'
        )""")
        print("Table ensured.")
        hashed_password = pwd_context.hash(password)
        print("Password hashed.")
        c.execute("INSERT INTO users (name, email, password, user_type) VALUES (?, ?, ?, ?)",
                  (name, email, hashed_password, "free"))
        conn.commit()
        print("User inserted.")
    except sqlite3.IntegrityError:
        print("Email already registered.")
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered.")
    except Exception as e:
        print(f"Registration error: {e}")
        conn.close()
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")
    conn.close()
    print("Registration successful.")
    return {"message": "Registration successful. Please log in."}

@router.post("/api/login")
def login(email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, email, password, user_type FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    if not user or not pwd_context.verify(password, user[3]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")
    token_data = {
        "email": user[2],  # Changed from "sub" to "email"
        "user_type": user[4],
        "name": user[1]
    }
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "user_type": user[4], "name": user[1]}

@router.post("/api/upload")
async def upload_file(
    audio: UploadFile = File(...),
    user=Depends(get_user_from_token)
):
    try:
        # Create user-specific upload directory using email
        user_email = user.get("email") or user.get("sub")
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found in token")
            
        uploads_dir = Path(__file__).resolve().parents[2] / "uploads" / user_email
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{audio.filename}"
        file_path = uploads_dir / filename
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Store file info in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            filename TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )""")
        c.execute("INSERT INTO uploads (user_email, filename) VALUES (?, ?)",
                 (user_email, filename))
        conn.commit()
        conn.close()
        
        return {"message": "File uploaded successfully", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    user=Depends(get_user_from_token)
):
    try:
        # Create temporary file for processing
        temp_dir = Path(__file__).resolve().parents[2] / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"transcribe_{audio.filename}"
        
        # Save the uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # Load Whisper model and transcribe
        model = whisper.load_model("base")
        result = model.transcribe(str(temp_path))
        
        # Clean up temporary file
        temp_path.unlink()
        
        return {"transcription": result["text"]}
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Ensure the file is closed
        await audio.close()

@router.post("/api/convert")
async def convert_voice(
    request: VoiceConversionRequest,
    user=Depends(get_user_from_token)
):
    try:
        user_type = user["user_type"]
        user_email = user.get("email") or user.get("sub")
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found in token")
            
        project_root = Path(__file__).resolve().parents[2]
        uploads_dir = project_root / "uploads" / user_email
        models_dir = project_root / "models" / user_email
        
        uploads_dir.mkdir(exist_ok=True)
        
        # Get source audio path
        source_audio_path = uploads_dir / request.sourceAudio
        if not source_audio_path.exists():
            raise HTTPException(status_code=400, detail="Source audio file not found")
        
        # Handle target voice path
        target_voice_path = None
        if user_type == "paid" and request.targetVoice:
            # First check uploads directory
            upload_voice_path = uploads_dir / request.targetVoice
            
            # Then check models directory
            model_voice_path = models_dir / f"{request.targetVoice}.wav"
            
            # Use whichever exists
            if upload_voice_path.exists():
                target_voice_path = str(upload_voice_path)
            elif model_voice_path.exists():
                target_voice_path = str(model_voice_path)
            else:
                raise HTTPException(status_code=400, detail=f"Target voice file not found: {request.targetVoice}")
        
        # Get relative paths
        rel_source_path = source_audio_path.relative_to(project_root)
        
        # Process voice conversion
        result_path = process_voice_cloning_xtts(
            str(rel_source_path),
            target_voice_path,
            user_type,
            request.transcript
        )
        
        return FileResponse(str(result_path), media_type="audio/wav", filename="converted.wav")
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Voice conversion error: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=f"Voice conversion failed: {str(e)}")

@router.post("/api/text-to-speech")
async def text_to_speech(
    request: TextToSpeechRequest,
    user=Depends(get_user_from_token)
):
    try:
        user_type = user["user_type"]
        user_email = user.get("email") or user.get("sub")
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found in token")
            
        project_root = Path(__file__).resolve().parents[2]
        uploads_dir = project_root / "uploads" / user_email
        models_dir = project_root / "models" / user_email
        
        uploads_dir.mkdir(exist_ok=True)
        
        # Handle target voice path
        target_voice_path = None
        if user_type == "paid" and request.targetVoice:
            # First check uploads directory
            upload_voice_path = uploads_dir / request.targetVoice
            
            # Then check models directory
            model_voice_path = models_dir / f"{request.targetVoice}.wav"
            
            # Use whichever exists
            if upload_voice_path.exists():
                target_voice_path = str(upload_voice_path)
            elif model_voice_path.exists():
                target_voice_path = str(model_voice_path)
            else:
                raise HTTPException(status_code=400, detail=f"Target voice file not found: {request.targetVoice}")
        
        # Process text-to-speech
        result_path = process_voice_cloning_xtts(
            None,  # No source audio needed
            target_voice_path,
            user_type,
            request.text  # Use text directly as transcript
        )
        
        return FileResponse(str(result_path), media_type="audio/wav", filename="converted.wav")
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Text-to-speech error: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech conversion failed: {str(e)}")

@router.post("/api/user/upgrade")
def upgrade_user(user=Depends(get_user_from_token)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    user_email = user.get("email") or user.get("sub")
    if not user_email:
        conn.close()
        raise HTTPException(status_code=400, detail="User email not found in token")
    c.execute("UPDATE users SET user_type = 'paid' WHERE email = ?", (user_email,))
    conn.commit()
    conn.close()
    return {"message": "User upgraded to premium."}

@router.get("/api/user/uploads")
def list_uploads(user=Depends(get_user_from_token)):
    uploads_dir = Path(__file__).resolve().parents[2] / "uploads"
    # For demo: list all files, but in production, associate uploads with user in DB
    user_email = user.get("sub") or user.get("username")
    files = []
    for f in uploads_dir.glob("*"):
        if f.is_file():
            files.append({
                "filename": f.name,
                "timestamp": f.stat().st_mtime,
                "status": "done"
            })
    return {"uploads": files}

@router.get("/api/verify-token")
async def verify_token(user=Depends(get_user_from_token)):
    return {"message": "Token is valid", "user": user}

@router.get("/api/uploads/{filename}")
async def get_uploaded_file(
    filename: str,
    user=Depends(get_user_from_token)
):
    try:
        user_email = user.get("email") or user.get("sub")
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found in token")
            
        uploads_dir = Path(__file__).resolve().parents[2] / "uploads" / user_email
        file_path = uploads_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            str(file_path),
            media_type="audio/wav",  # You might want to determine this based on file extension
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {str(e)}")

@router.post("/api/train-voice")
async def train_voice(
    request: VoiceTrainingRequest,
    user=Depends(get_user_from_token)
):
    try:
        # Check if user is paid
        user_type = user["user_type"]
        if user_type != "paid":
            raise HTTPException(status_code=403, detail="Voice training is only available for premium users")
            
        user_email = user.get("email") or user.get("sub")
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found in token")
            
        uploads_dir = Path(__file__).resolve().parents[2] / "uploads" / user_email
        if not uploads_dir.exists():
            raise HTTPException(status_code=404, detail="User uploads directory not found")
        
        # Create models directory for this user
        models_dir = Path(__file__).resolve().parents[2] / "models" / user_email
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Validate voice name (only allow alphanumeric and underscores)
        voice_name = request.voiceName
        if not re.match(r'^[a-zA-Z0-9_]+$', voice_name):
            raise HTTPException(status_code=400, detail="Voice name can only contain letters, numbers, and underscores")
        
        # Get audio file paths
        audio_files = []
        for filename in request.audioFiles:
            file_path = uploads_dir / filename
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
            audio_files.append(file_path)
        
        if len(audio_files) < 1:
            raise HTTPException(status_code=400, detail="At least one audio file is required")
        
        # Train the model
        model_path = train_custom_voice_model(audio_files, voice_name, user_email)
        
        return {
            "message": f"Voice model '{voice_name}' trained successfully",
            "model_name": voice_name,
            "status": "complete"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice training failed: {str(e)}")

@router.get("/api/voice-models")
async def list_voice_models(user=Depends(get_user_from_token)):
    try:
        user_email = user.get("email") or user.get("sub")
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found in token")
            
        models_dir = Path(__file__).resolve().parents[2] / "models" / user_email
        if not models_dir.exists():
            return {"models": []}
        
        models = []
        # Look for both .pth and .wav files as voice models
        for model_path in list(models_dir.glob("*.pth")) + list(models_dir.glob("*.wav")):
            model_name = model_path.stem
            models.append({
                "name": model_name,
                "created": model_path.stat().st_mtime
            })
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voice models: {str(e)}")