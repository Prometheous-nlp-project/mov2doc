import uuid
import os
from time import time
from threading import Thread

import numpy as np
import speech_recognition as sr
from scipy.io import wavfile


def download(url):
    """
    download wav file from `url` as `fname`
    """
    fname = str(uuid.uuid4())
    os.system(
        f"yt-dlp -P ./data -o '{fname}.%(ext)s' -x --audio-format wav {url}")
    if os.path.getsize("./data/"+fname+".wav") > 500*1024*1024:
        raise Exception("video too long.")
    freq, data = wavfile.read("./data/"+fname+".wav")
    """
    scipy.signal 이나 librosa를 통해 downsampling을 할 수 있지만, 시간이 오래 걸려 권장되지 않음
    """
    data = data[:, 0]
    os.remove(f"./data/{fname}.wav")
    return data, freq, fname


def marking_not_to_cut(data, bound, patience, restore):
    not_to_cut = np.ones_like(data, dtype=np.bool8)
    prv = (np.abs(data) > bound).cumsum()
    prv_to_cut = np.where(prv[:-patience] == prv[patience:])
    if prv_to_cut[0].shape[0] == 0:
        return not_to_cut
    groups = np.split(prv_to_cut[0], np.where(
        np.diff(prv_to_cut[0]) != 1)[0] + 1)
    to_cut = np.concatenate(
        [np.arange(x[0] + restore, x[-1] + patience - 1) for x in groups])
    not_to_cut[to_cut] = False
    return not_to_cut


def find_bound(data, patience, restore, target_rate=.20):
    lt, rt = np.abs(data).min(), np.abs(data).max()
    total = data.shape[0]
    while lt + 1 < rt:
        md = (lt+rt)//2
        marked = marking_not_to_cut(data, md, patience, restore)
        if target_rate < (1 - marked.sum() / total):
            rt = md
        else:
            lt = md
    return lt


def silence_trunc_bs(data, freq):
    '''
    이분탐색을 사용하여 threshold를 구하여 해당 값보다 낮을 때 truncation 하는 경우
    18초의 running time
    '''
    bound = find_bound(data, freq//40, freq//400, target_rate=.2)
    marked = marking_not_to_cut(data, bound, freq//10, freq//100)
    idx = np.where(marked == True)[0]
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    each_dur = [len(x) for x in groups]
    tmp = 0
    pos = []
    for i, v in enumerate(each_dur):
        if tmp + v > freq * 60:
            tmp = v
            pos.append(i - 1)
        else:
            tmp += v
    sum_dur = np.array(each_dur).cumsum()
    if len(pos):
        pos = np.array(pos)
        timestamp = sum_dur[pos]
    else:
        timestamp = []
    return marked, timestamp


def to_text(fname, language="en-US", adjust_for_noise=True, offset=0, duration=60):
    r = sr.Recognizer()
    data = sr.AudioFile(fname)

    with data as source:
        if adjust_for_noise:
            r.adjust_for_ambient_noise(source)
        audio = r.record(source, offset=offset, duration=duration)

    text = r.recognize_google(audio, language=language)

    return text


def get_txt(i, data, freq, fname, timestamp, reduce_noise, result):
    if 0 < i and i < len(timestamp):
        target = data[timestamp[i-1]:timestamp[i]]
    elif i == 0:
        if len(timestamp) == 0:
            target = data
        else:
            target = data[:timestamp[i]]
    else:
        target = data[timestamp[i-1]:]
    wavfile.write(f"./data/{fname}_{i}.wav", freq, target)
    txt = to_text(f"./data/{fname}_{i}.wav", adjust_for_noise=reduce_noise)
    result[i] = txt


def stt(data, freq, fname, timestamp, reduce_noise):
    n_threads = len(timestamp)+1
    result = [None]*n_threads
    threads = []

    s = time()
    for i in range(n_threads):
        th = Thread(target=get_txt, args=(i, data, freq,
                    fname, timestamp, reduce_noise, result))
        threads.append(th)
        th.start()

    for i, th in enumerate(threads):
        th.join()
        os.remove(f"./data/{fname}_{i}.wav")

    return ' '.join(result)


def stt_from_url(url, reduce_noise=False):
    total_time = 0
    print(f"start downloading {url}...")
    s = time()
    data, freq, fname = download(url)
    e = time()
    print(f"time elapsed: {round(e - s, 4)}s")
    total_time += e-s

    print("start silence truncation...")
    s = time()
    marked, timestamp = silence_trunc_bs(data, freq)
    data = data[marked]
    e = time()
    print(f"time elapsed: {round(e - s, 4)}s")
    total_time += e-s

    print("start speech-to-text by google web api...")
    s = time()
    text = stt(data, freq, fname, timestamp, reduce_noise)
    e = time()
    print(f"time elapsed: {round(e - s, 4)}s")
    total_time += e-s

    print(f"total time elapsed: {round(total_time, 4)}s")
    return text
