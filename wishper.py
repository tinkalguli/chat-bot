import whisper
from whisper.utils import get_writer

model = whisper.load_model("base")
audio = "call_recordings/ipbx.policyadvisorbrok.callrecording.users.1001.1703354643273-1701412302808-x1001u1-+16049927920_E.mp3"
result = model.transcribe(audio, fp16=False)
output_directory = "./"

# Save as an TEXT file
text_writer = get_writer("txt", output_directory)
text_writer(result, audio)