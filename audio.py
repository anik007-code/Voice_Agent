import wave
import pyaudio

# Path to your .wav file
filename = "Audio/output_94a8272a-87d9-4ef4-b589-149b4c8a36ca.wav"

# Open the .wav file
wf = wave.open(filename, 'rb')

# Create a PyAudio object
p = pyaudio.PyAudio()

# Open a stream with the correct audio parameters
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# Read data in chunks
chunk = 1024
data = wf.readframes(chunk)

# Play the audio by writing data to the stream
while data:
    stream.write(data)
    data = wf.readframes(chunk)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
wf.close()
