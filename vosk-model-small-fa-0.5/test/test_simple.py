#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import wave

wf = wave.open("test.wav", "rb")
model = Model("..")
rec = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())

print(rec.FinalResult())
