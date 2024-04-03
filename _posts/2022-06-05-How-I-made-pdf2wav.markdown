---
layout: post
title: "How I made pdf2wav"
date: 2022-06-05 22:56:00
categories: [Deep Learning]
tags: [pdf2wav, DeepLearning, AI, TTS, Text-To-Speech, spectrogram, vocoder, FastPitch, MelGAN, tacotron2, speechsynthesis]
comments: true
#image:
#  feature: hook_model.jpg
#  credit: Markus Spiske
#  creditlink: https://www.pexels.com/photo/display-coding-programming-development-1921326/
---

By constructing a pipeline of pdf reader stream to FastPitch spectrogram generator to MelGAN vocoder. pdf2wav can convert any pdf*(, doc, txt, etc.)* file to wav*(, flac, mp3, etc.)* audio file. This can be used as an audiobook generator.

##### Samples:

/assets/audio/pdf2wav.wav

{% include open-embed.html %}