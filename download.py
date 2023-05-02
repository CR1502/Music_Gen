import os
import csv
from pytube import YouTube
import torchaudio

# Define the path to save the audio files
save_path = './musiccaps_dataset'

# Create the directory if it does not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define the path to the csv file containing the video IDs and start/end times
csv_path = './musiccaps-public.csv'

# Open the csv file and read the video IDs and start/end times
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        video_id = row['ytid']
        start_time = row['start_s']
        end_time = row['end_s']

        # Define the YouTube video URL
        video_url = f'https://www.youtube.com/watch?v={video_id}'

        # Download the YouTube video using pytube
        yt = YouTube(video_url)
        yt.streams.filter(only_audio=True).first().download(output_path=save_path)

        # Load the downloaded audio file using torchaudio
        audio_path = os.path.join(save_path, f'{video_id}.mp4')
        waveform, sample_rate = torchaudio.load(audio_path)

        # Trim the audio to the desired start and end times
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        waveform = waveform[:, start_frame:end_frame]

        # Save the trimmed audio file as a WAV file
        output_path = os.path.join(save_path, f'{video_id}.wav')
        torchaudio.save(output_path, waveform, sample_rate=sample_rate)

        # Delete the original downloaded audio file
        os.remove(audio_path)
