from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str
    voice_sample: str
    voice_speed: float
