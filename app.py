from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sounddevice as sd
import soundfile as sf
from vosk import Model, KaldiRecognizer, SetLogLevel
import re
import wave
import os
import requests
from pydub import AudioSegment
from pydub.playback import play
import os
from tkinter import messagebox


data = pd.read_csv('dataset/train_46451.csv')
documents =  data['title'] + ' ' + data['question'] + ' ' + data['context'] + ' ' + data['answer'] 

# ساخت مدل TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)


def chatbot_get_response(question):
    # تبدیل متن کاربر به بردار TF-IDF
    user_input = question
    user_tfidf = tfidf_vectorizer.transform([user_input])

    # محاسبه شباهت کیسه‌کلماتی با cosine similarity
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)

    # یافتن بهترین پاسخ بر اساس شباهت
    best_match_index = similarities.argmax()
    max_similarity = similarities.argmax()
    best_answer = data['answer'][best_match_index]
    best_title = data['title'][best_match_index]
    best_context = data['context'][best_match_index]
    best_question = data['question'][best_match_index]

    return best_answer, best_title, best_context, best_question, similarities[0][similarities.argmax()]


def speech_to_text():
    samplerate = 44100  # نرخ نمونه‌برداری
    duration = 8  # مدت زمان ضبط به ثانیه
    filename = 'output.wav'  # نام فایل خروجی

    print("The AI ​​model is listening...")
    myrecording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # انتظار برای اتمام ضبط
    print("End of listening.")
    sf.write(filename, myrecording, samplerate)
    wf = wave.open("output.wav", "rb")
    model = Model("vosk-model-small-fa-0.5/")
    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data): pass 
        else: pass

    result = rec.FinalResult()
    pattern = r'"text" : "([^"]+)"'
    match = re.search(pattern, result)
    if match:
        extracted_text = match.group(1)
        return extracted_text    


def extract_first_sentences(text):
        sentences = text.split(".")
        first_four_sentences = sentences[:2]
        result_text = ". ".join(first_four_sentences).strip()
        if text.endswith("."): result_text += "."
        return result_text


def tts_persian(text):
    extract_first_sentences(text)
    url = f"https://tts.datacula.com/api/tts?text={text}&model_name=amir"
    audio_url = "output.wav"
    # تنظیم هدرهای درخواست
    headers = {
        'accept': 'application/json'
    }

    print("The AI ​​model is thinking...")
    # ارسال درخواست GET به API
    response = requests.get(url, headers=headers)

    # بررسی وضعیت پاسخ
    if response.status_code == 200:
        # ذخیره فایل wav در صورت موفقیت‌آمیز بودن درخواست
        with open(audio_url, "wb") as file:
            file.write(response.content)
    else:
        print("خطا در درخواست API:", response.status_code, response.text)
    return audio_url


def say_welcome():
    os.system('cls')
    audio = AudioSegment.from_wav('voices/welcome.wav')
    play(audio)


def say_next_question():
    os.system('cls')
    audio = AudioSegment.from_wav('voices/next-question.wav')
    play(audio)


say_welcome()
while True:
    input_text = speech_to_text()
    os.system('cls')
    best_answer, best_title, best_context, best_question, max_similarity = chatbot_get_response(question=input_text)
    if max_similarity<0.1: best_context = "چطور میتوانم کمکتان کنم؟ سوال خود را از من بپرسید"
    audio_url = tts_persian(text=best_context)
    audio = AudioSegment.from_wav(audio_url)
    play(audio)
    messagebox.showinfo("هوش مصنوعی BonyadAI", f"سوال پرسیده شده توسط شما: \n{input_text}\n\nمتن کامل پاسخ: \n{best_context}")
    say_next_question()