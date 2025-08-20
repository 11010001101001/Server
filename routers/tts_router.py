from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.tts_request import TTSRequest
from voice_synthesizers.tts_voice_synthesizer import TTSVoiceSynthesizer

router = APIRouter()
voice_synthesizer = TTSVoiceSynthesizer()


@router.post('/tts')
def tts_endpoint(req: TTSRequest):
    print(req)
    buf = voice_synthesizer.synthesize(req)
    return StreamingResponse(buf, media_type='audio/wav')
