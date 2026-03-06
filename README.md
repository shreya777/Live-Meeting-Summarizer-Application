# Live-Meeting-Summarizer-Application
Project Development Phases
### Week 1 – Project Setup & Speech-to-Text Evaluation**
Designed the overall system pipeline: Audio → Text → Speaker Identification → Summary.\
Studied and compared Speech-to-Text (STT) models such as Whisper and Vosk.\
Collected meeting audio samples from the AMI Meeting Corpus and custom recordings.\
Evaluated transcription accuracy using Word Error Rate (WER).\
**Output**\
System architecture design
Sample meeting audio dataset
STT model comparison report
**Tech Stack**
Python, Vosk, OpenAI Whisper, JiWER (for WER calculation)

### Week 2 – Real-Time Speech-to-Text
Implemented live audio capture from the microphone.\
Built a real-time speech-to-text transcription system.
Displayed live transcription output in the terminal.
Optimized speed and transcription accuracy.
**Output**\
Working real-time STT application
Transcription logs
Accuracy report (WER < 15%)
**Tech Stack**\
Python, PyAudio / SoundDevice, Threading, Selected STT model (Whisper / Vosk)

### Week 3 – Speaker Diarization
Implemented speaker identification in meeting audio.\
Detected and labeled multiple speakers in conversations.\
Integrated diarization output with transcription results to generate speaker-wise transcripts.\
**Output**\
Speaker-wise meeting transcript
Diarization module
Accuracy report (DER < 20%)
**Tech Stack**\
pyannote.audio, torchaudio, AMI Meeting Corpus dataset

##Current Progress:
### Week 4 - AI-based summarization
