# Urdu checklist audio

The checklist recordings use [AhsanTalal's Matcha Urdu female model](https://huggingface.co/AhsanTalal/urdu-matcha-tts-female)
with the [Matcha-TTS HiFi-GAN vocoder](https://github.com/shivammehta25/Matcha-TTS).
The Urdu model is distributed under the Apache 2.0 license.

Each MP3 filename uses the first 16 hexadecimal digits of the SHA-256 hash of
the checklist text before spoken number formatting. The app can synthesize
audio on demand if the source text changes.
