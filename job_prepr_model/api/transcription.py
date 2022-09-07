from google.cloud import speech_v1 as speech
import io
import os


def transcribe(source):
    """Transcribe the given audio file from a local or bucket path"""
    if os.environ.get("TRANSCRIPTION_SOURCE") == "local":
        client = speech.SpeechClient()

        with io.open(source, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        audio_channel_count=2
        )
    else:
        config = dict(language_code="en-UK", enable_automatic_punctuation=True)
        audio = dict(uri=source)

    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        best_alternative = result.alternatives[0]
    transcript = best_alternative.transcript
    print(transcript)

transcribe("/Users/andrei/Downloads/you-are-acting-so-weird.wav")
