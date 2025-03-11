import speech_recognition as sr
import os
import time
import wave
import contextlib
from transformers import pipeline
import torch
import re
from deepmultilingualpunctuation import PunctuationModel

def add_punctuation(text):
    print("adding punctuation marks...")
    start_time = time.time()
    
    #try except (if model does not work)
    # deepmultilingualpunctuation model
    model = PunctuationModel(model="kredor/punctuate-all")
    
    punctuated_text = model.restore_punctuation(text)
    
    sentences = re.split(r'([.!?] )', punctuated_text)
    capitalized_text = ""
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i]
            if sentence and len(sentence) > 0:
                sentence = sentence[0].upper() + sentence[1:]
                capitalized_text += sentence
            if i + 1 < len(sentences):
                capitalized_text += sentences[i+1]
    
    capitalized_text = re.sub(r'\s+', ' ', capitalized_text).strip()
    
    elapsed_time = time.time() - start_time
    print(f"punctuation added (time: {elapsed_time:.2f} seconds)")
    
    return capitalized_text

def analyze_emotion(audio_file_path):
    print("emotion analysis starting...")
    start_time = time.time()
    
    try:
        classifier = pipeline("audio-classification", 
                             model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                             device=0 if torch.cuda.is_available() else -1)
        
        emotion_result = classifier(audio_file_path)
        
        top_emotion = emotion_result[0]["label"]
        
        emotion_result = top_emotion
        
        elapsed_time = time.time() - start_time
        print(f"emotion analysis completed: {emotion_result} (time: {elapsed_time:.2f} seconds)")
        
        return emotion_result
    except Exception as e:
        print(f"emotion analysis error: {e}")
        return "unknown"

def transcribe_audio_file(audio_file_path):
    print(f"starting process: {audio_file_path}")
    start_time = time.time()
    
    if not os.path.exists(audio_file_path):
        return f"error: {audio_file_path} not found."
    
    file_size_mb = os.path.getsize(audio_file_path) / (1024*1024)
    print(f"file found. size: {file_size_mb:.2f} MB")
    
    file_ext = os.path.splitext(audio_file_path)[1].lower()
    
    if file_ext == '.wav':
        try:
            with contextlib.closing(wave.open(audio_file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                print(f"wav file duration: {duration:.2f} seconds")
        except Exception as e:
            print(f"error: could not get wav file information: {e}")
    
    recognizer = sr.Recognizer()
    
    try:
        print("loading audio file...")
        with sr.AudioFile(audio_file_path) as source:
            print("applying noise reduction...")

            recognizer.adjust_for_ambient_noise(source) # decrease noise
            
            print("recording audio...")
            audio_data = recognizer.record(source)
            
            print("sending audio to google speech recognition api...")
            text = recognizer.recognize_google(audio_data, language="en-US")
            
            elapsed_time = time.time() - start_time
            print(f"process completed. time taken: {elapsed_time:.2f} seconds")
            
            return text
    except sr.UnknownValueError:
        print("error: google speech recognition could not understand audio")
        return "error: google speech recognition could not understand audio"
    except sr.RequestError as e:
        print(f"error: no result from google speech recognition service; {e}")
        return f"error: no result from google speech recognition service; {e}"
    except Exception as e:
        print(f"error: {e}")
        return f"error occurred: {e}"
    finally:
        elapsed_time = time.time() - start_time
        print(f"total time: {elapsed_time:.2f} seconds")

def transcribe_large_audio(audio_file_path, chunk_duration_ms=60000):
    from pydub import AudioSegment
    import tempfile
    
    print(f"starting large file processing: {audio_file_path}")
    
    file_size_mb = os.path.getsize(audio_file_path) / (1024*1024)
    
    if file_size_mb < 10:
        return transcribe_audio_file(audio_file_path)
    
    print(f"large file detected ({file_size_mb:.2f} MB). splitting into chunks...")
    
    try:
        audio = AudioSegment.from_wav(audio_file_path)
        
        duration_ms = len(audio)
        print(f"total duration: {duration_ms/1000:.2f} seconds")
        
        chunks = [audio[i:i+chunk_duration_ms] for i in range(0, duration_ms, chunk_duration_ms)]
        print(f"file split into {len(chunks)} chunks")
        
        transcriptions = []
        
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            print(f"processing chunk {i+1}/{len(chunks)}...")
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            chunk.export(temp_file.name, format="wav")
            
            chunk_text = transcribe_audio_file(temp_file.name)
            
            os.unlink(temp_file.name) # delete temp file
            
            if not chunk_text.startswith("error:"):
                transcriptions.append(chunk_text)
            
            chunk_elapsed_time = time.time() - chunk_start_time
            print(f"chunk {i+1} completed. time taken: {chunk_elapsed_time:.2f} seconds")
        
        full_text = " ".join(transcriptions)
        return full_text
    
    except Exception as e:
        print(f"large file processing error: {e}")
        return f"error: large file processing error: {e}"
    
def save_to_file(text, output_file="result.txt"):
    try:
        line_length = 80 
        words = text.split()
        formatted_text = ""
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 > line_length:
                formatted_text += current_line + "\n"
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            formatted_text += current_line
        
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(formatted_text)
        print(f"result saved to '{output_file}'")
        return True
    except Exception as e:
        print(f"error: could not save to file: {e}")
        return False

def main():
    audio_file_path = input("please enter the path of the audio file: ")
    
    print(f"processing audio file: {audio_file_path}")
    
    file_size_mb = os.path.getsize(audio_file_path) / (1024*1024)
    
    if file_size_mb > 10 and audio_file_path.lower().endswith('.wav'):
        text_result = transcribe_large_audio(audio_file_path)
    else:
        text_result = transcribe_audio_file(audio_file_path)
    
    punctuated_text = add_punctuation(text_result)
    
    emotion = analyze_emotion(audio_file_path)
    
    final_result = f"{punctuated_text} ({emotion})"
    
    print("\nconverted text with punctuation and emotion:")
    print(final_result)

    save_to_file(final_result, "result.txt")

if __name__ == "__main__":
    main() 