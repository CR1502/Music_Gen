import csv
import subprocess
import os

# Path to the MusicCaps-public.csv file
csv_file = "musiccaps-public.csv"

# Path to output directory to store audio files
output_dir = "music_data"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the MusicCaps-public.csv file and extract audio files
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Skip the header row
        if reader.line_num == 1:
            continue

        # Extract the fields from the row
        ytid = "ytid"
        start_time = "start_s"
        end_time = "end_s"

        # Construct the URL to the YouTube video
        video_url = f"https://www.youtube.com/watch?v={ytid}"

        # Construct the output filename for the audio file
        output_filename = f"{ytid}_{start_time}-{end_time}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # Download the audio file using youtube-dl and ffmpeg
        command = f"youtube-dl -x --audio-format wav -o - {video_url} | ffmpeg -i - -ss {start_time} -to {end_time} -vn -ar 44100 -ac 2 -ab 192k -f wav {output_path}"
        subprocess.call(command, shell=True)
