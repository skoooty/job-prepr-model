from google.cloud import speech_v1 as speech
import io

def transcribe_from_bucket(uri):
    '''
    Pass this function your audio file and the config and you will get a transcription
    '''
    config = dict(language_code="en-UK", enable_automatic_punctuation=True)
    audio = dict(uri=uri)
    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        best_alternative = result.alternatives[0]
    transcript = best_alternative.transcript
    print(transcript)



def transcribe_from_local(path):
    """Transcribe the given audio file from a local path"""
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        audio_channel_count=2
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
    return result.alternatives[0].transcript

transcribe_from_local("/Users/andrei/Downloads/you-are-acting-so-weird.wav")
