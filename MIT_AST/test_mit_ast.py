# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:57:38 2024

See LICENSE file in the root of the repository. 

Copyright (c) Aki Härmä, DACS/FSE, Maastricht University, 2023
"""

# Use a pipeline as a high-level helper
from transformers import pipeline
import torchaudio as ta
import torch
# Load model directly
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

class MIT_AST_model():
    def __init__(self):
        self.pipe = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
        self.extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def classify(self, audio_file):
        sig, fs = ta.load(audio_file)
        sig16 = ta.transforms.Resample(orig_freq=fs,new_freq=16000)(sig[0,:])
        inputs = self.extractor(sig16, sampling_rate=16000, return_tensors="pt")        
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_ids = torch.argmax(logits, dim=-1).item()
        predicted_label = self.model.config.id2label[predicted_class_ids]
        return predicted_label

# classifier = MIT_AST_model()
# afile = "../gardenTransformer/data/samples/bluetit_1.wav"
# bfile ='miaow_16k.wav'
# cfile = 'speech_whistling2.wav'
# door_file = 'er_file_2023_09_02_10_00_17.wav'
# jack_file = 'er_file_2023_09_02_16_45_18.wav'
# owl_file = 'er_file_2023_09_04_7_13_08.wav'
# tit_file = 'big_test_folder/few_garden_files/er_file_2023_09_04_7_49_38.wav'

# res = classifier.classify(tit_file)

# print(res)

