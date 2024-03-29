# kaggle-bengali-handwritten

<p align="center">
  <img src="poster.png"
  alt="Markdown Monster icon"
      width=800
      height=500/>
</p>

## Description (Copied from kaggle)

Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant business and educational interest in developing AI that can optically recognize images of the language handwritten. This challenge hopes to improve on approaches to Bengali recognition.

Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).

Bangladesh-based non-profit Bengali.AI is focused on helping to solve this problem. They build and release crowdsourced, metadata-rich datasets and open source them through research competitions. Through this work, Bengali.AI hopes to democratize and accelerate research in Bengali language technologies and to promote machine learning education.

For this competition, you’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

By participating in the competition, you’ll hopefully accelerate Bengali handwritten optical character recognition research and help enable the digitalization of educational resources. Moreover, the methods introduced in the competition will also empower cousin languages in the Indian subcontinent.

[Click to see the full challenge info](https://www.kaggle.com/c/bengaliai-cv19/overview)

## Experiments pipeline

- Hardware stack

  - RAM : 16 GB
  - Accelerator type : Nvidia GPU Geforce GTX 1060 Max-Q
  - VRAM : 6.1 GB
  - num workers : 4 (CPU count)

- Software stack
  - Language : Python (version 3.8.6)
  - DL library : Pytorch (version 1.7.1) + Pytorch Lightning (1.2.0)

## Approach

## Usage

- Training
- Inference

## Acknowledgements

- The code is based on learning from the shared notebooks on kaggle
- Some of the snippets code copied from anywhere will be linked to their source (original implementation for credits)
