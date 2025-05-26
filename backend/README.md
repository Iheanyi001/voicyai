# Voice Cloning Service Backend

This is a FastAPI backend for a voice cloning service. It supports:
- User authentication (free/paid)
- Audio upload with duration checks (2 min for free, 10 min for paid)
- Optional target voice upload (paid only)
- Voice cloning endpoint that returns a generated .wav file

## Project Structure
- `main.py`: FastAPI app entry point
- `app/`: API modules (routes, auth, utils, voice_cloning)
- `uploads/`: Stores uploaded audio files

## Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the server:
   ```sh
   uvicorn main:app --reload
   ```
3. The API will be available at http://127.0.0.1:8000

## API Endpoint
- `POST /api/clone-voice/`
  - Form fields:
    - `user_type`: "free" or "paid"
    - `audio`: Your voice file (.wav)
    - `target_voice`: (Paid only) Target voice file (.wav)
  - Returns: `converted.wav` (audio/wav)

## Notes
- Authentication is a placeholder. Add JWT/session logic for production.
- The voice cloning logic is a placeholder. Integrate your ML pipeline in `app/voice_cloning.py`.

---

For frontend integration, see the upcoming `frontend/` folder.
