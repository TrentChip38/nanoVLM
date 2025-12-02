import pyttsx3

def speak(text):
    """
    Converts the given text to speech and plays it through the computer's speakers.
    """
    engine = pyttsx3.init() # Initialize the TTS engine
    engine.say(text)      # Queue the text to be spoken
    engine.runAndWait()   # Wait for the speech to complete

# Example usage:
speak("Hello, I am your synthesized voice assistant.")
speak("This Python project uses a VLM model to analize camera input.")
speak("I will describe what I see every minute.")