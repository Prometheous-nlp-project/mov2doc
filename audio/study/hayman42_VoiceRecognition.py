import speech_recognition as sr
import os
from time import time


def download(url, fname):
    os.system(f"yt-dlp -P ./data -o {fname}.%(ext)s -x --audio-format wav {url}")


def to_text(fname, adjust_for_noise=True):
    r = sr.Recognizer()
    data = sr.AudioFile(fname)

    with data as source:
        if adjust_for_noise:
            r.adjust_for_ambient_noise(source)

        audio = r.record(source, duration=60)

    print("Recognize", fname)
    s = time()
    text = r.recognize_google(audio)
    # text = r.recognize_sphinx(audio)
    print("Execution time:", time() - s)
    print()

    return text


if __name__ == "__main__":
    # "data" directory is required
    fname = "english"  # https://youtube.com/shorts/s74SSoJuobg?feature=share
    # download("https://youtube.com/shorts/s74SSoJuobg?feature=share", fname)
    print(to_text(f"./data/{fname}.wav"))
    print()
    print(to_text(f"./data/{fname}.wav", False))
