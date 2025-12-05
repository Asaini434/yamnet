import os
import shutil
import pandas as pd
from pydub import AudioSegment

tracks_csv = "Data/tracks.csv"
audio_dir = "Data/fma_small"
output_root = "datasets/fma_small"

tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1]) # load metadata
genre_col = ('track', 'genre_top')
known_genres = ['Hip-Hop', 'Rock', 'Pop']  # Have GTZAN equivalents
unknown_genres = ['Folk', 'Experimental', 'Electronic']  # Don't have GTZAN equivalents
target_genres = known_genres + unknown_genres
all_genre_tracks = tracks[genre_col].dropna()
filtered_tracks = all_genre_tracks[all_genre_tracks.isin(target_genres)]
print(f"Total tracks in target genres: {len(filtered_tracks)}")
for genre in target_genres:
    count = (filtered_tracks == genre).sum()
    print(f"  {genre}: {count}")

samples_per_genre = 20
sampled_tracks = []
for genre in target_genres:
    genre_tracks = filtered_tracks[filtered_tracks == genre]
    existing_tracks = [] # filter to only tracks that exist in fma_small
    for track_id in genre_tracks.index:
        track_id_str = f"{track_id:06d}"
        folder = track_id_str[:3]
        src_file = os.path.join(audio_dir, folder, f"{track_id_str}.mp3")
        if os.path.exists(src_file):
            existing_tracks.append(track_id)
    existing_tracks = genre_tracks.loc[existing_tracks]
    if len(existing_tracks) >= samples_per_genre:
        sampled = existing_tracks.sample(n=samples_per_genre, random_state=69)
    else:
        sampled = existing_tracks
    sampled_tracks.append(sampled)
    print(f"{genre}: {len(sampled)} available")
selected_tracks = pd.concat(sampled_tracks)
print(f"\nSelected {len(selected_tracks)} tracks total")
os.makedirs(os.path.join(output_root, "test"), exist_ok=True)
for genre in target_genres:
    genre_clean = genre.lower().replace(' ', '_').replace('-', '_')
    os.makedirs(os.path.join(output_root, "test", genre_clean), exist_ok=True)

print("Converting MP3 to WAV")
converted_count = 0
for track_id in selected_tracks.index:
    genre = selected_tracks[track_id]
    track_id_str = f"{track_id:06d}"
    folder = track_id_str[:3]
    src_file = os.path.join(audio_dir, folder, f"{track_id_str}.mp3")
    if os.path.exists(src_file):
        genre_clean = genre.lower().replace(' ', '_').replace('-', '_')
        dst_file = os.path.join(output_root, "test", genre_clean, f"{track_id_str}.wav")
        audio = AudioSegment.from_mp3(src_file)
        audio.export(dst_file, format="wav")
        converted_count += 1
        print(f"Converted {track_id_str}, {genre}")
print(f"Done, converted {converted_count}/{len(selected_tracks)}")