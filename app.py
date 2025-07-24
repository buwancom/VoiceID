from flask import Flask, request, render_template, jsonify
import os
import librosa
import numpy as np
import soundfile as sf
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    webm_path = os.path.join(UPLOAD_FOLDER, 'input.webm')
    wav_path = os.path.join(UPLOAD_FOLDER, 'input.wav')
    file.save(webm_path)

    # Convert to WAV
    convert_to_wav(webm_path, wav_path)

    # Process MFCCs from 2-sec chunks
    mfcc_chunks = extract_mfcc_chunks(wav_path)

    return jsonify({'mfcc_chunks': mfcc_chunks})


def convert_to_wav(input_path, output_path):
    subprocess.run(['ffmpeg', '-y', '-i', input_path, output_path],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def extract_mfcc_chunks(filepath):
    y, sr = librosa.load(filepath, sr=None)
    chunk_duration = 2  # seconds
    chunk_size = sr * chunk_duration
    total_chunks = len(y) // chunk_size
    mfcc_list = []

    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = y[start:end]
        if len(chunk) == 0:
            continue
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_list.append(mfcc_mean.tolist())

    return mfcc_list

if __name__ == '__main__':
    app.run(debug=True)
