from google.cloud import speech_v1p1beta1 as speech
import io

def transcribe(source):
    """Transcribe the given audio file from a local or bucket path"""
    with io.open(source, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
    sample_rate_hertz=48000,
    language_code="en-US",
    audio_channel_count=1
    )
    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio)
    best_alternative = speech.SpeechRecognitionAlternative()
    for result in response.results:
        best_alternative = result.alternatives[0]
    transcript = best_alternative.transcript
    return transcript
