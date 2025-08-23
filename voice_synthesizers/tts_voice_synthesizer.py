import io
import os

import soundfile as sf
import torch

cwd = os.getcwd()
os.environ["TTS_HOME"] = f'{cwd}/voice_models'

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])
torch.backends.cudnn.benchmark = True

from TTS.api import TTS
from schemas.tts_request import TTSRequest


class TTSVoiceSynthesizer:
    def __init__(self):
        self.tts = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=True
        )
        self.tts.to('cuda')

    def synthesize(self, request: TTSRequest):
        audio = self.tts.tts(
            text=request.text,
            speaker_wav=f'voice_samples/{request.voice_sample}',
            language="ru",
            speed=request.voice_speed,
            split_sentences=False
        )
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format='WAV')
        buf.seek(0)
        return buf
