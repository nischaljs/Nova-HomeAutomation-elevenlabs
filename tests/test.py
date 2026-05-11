import os
from groq import Groq
from dotenv import load_dotenv
# from playsound import playsound
# -----------------------------
# INIT
# -----------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ"))
async def generate_speech(text):
    print(f"Generating speech for: {text}")
    speech_file_path = "response.wav" 
    model = "canopylabs/orpheus-v1-english"
    voice = "hannah"
    
    response_format = "wav"

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format
    )
    response.write_to_file(speech_file_path)
    print("reponse saved")
    return speech_file_path

    # to stream to esp 
    # from websocket import stream_audio
    # await stream_audio()
    



# # if __name__ == "__main__":
# #     import asyncio
# #     test_text = "Hello, this is a test of the speech generation system!"
# #     asyncio.run(generate_speech(test_text))



# import winsound
# import os
# import time
# from speech import play_audio
# import asyncio
# FILE = "response.wav"

# def wait_for_file(path):
#     print("Waiting for file...")

#     while not os.path.exists(path) or os.path.getsize(path) < 1000:
#         time.sleep(0.1)

#     print("File ready!")

# # def play_audio(path):
# #     print("Playing audio...")
# #     winsound.PlaySound(path, winsound.SND_FILENAME)

# if __name__ == "__main__":
#     wait_for_file(FILE)
#     asyncio.run(play_audio(FILE))
#     # play_audio(FILE)