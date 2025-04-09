import requests
import os
import pandas as pd
import sklearn as skl
from sklearn.cluster import KMeans
from sklearn import preprocessing

print("Identify [g]enres or create [p]laylist from genre file?")
cmd = input()[0]

if (cmd == 'g'):

  print("Choose number of genres to identify or use [d]efault (500):")
  K = input()
  if (K[0] == 'd'):
    K = 500
  else:
    try:
      K = int(K)
    except:
      print("Not a number.")
      exit()

  # Read data
  df = pd.read_csv('./archive/spotify_songs.csv').drop_duplicates(subset=['track_id'])
  songs = df[['track_name', 'track_artist']]
  ids = df[['track_id']]
  data = df[['track_popularity', 
          'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
  data = preprocessing.normalize(data)

  # Cluster songs into genres
  model = KMeans(n_clusters=K)
  predictions = model.fit_predict(data)

  # Create directories if necessary
  if not os.path.exists('genres'):
    os.makedirs('genres')
  if not os.path.exists('ids'):
    os.makedirs('ids')

  # Write genre information to files
  genres = [0] * K
  for i in range(K):
    genres[i] = songs[predictions == i].dropna()
    file = open('./genres/genre' + str(i) + '.txt', 'w')
    for index, row in genres[i].iterrows():
      file.write(str((row['track_artist'] + ' - ' + row['track_name']).encode(errors='replace'))[2:-1] + '\n')
    genreids = ids[predictions == i].dropna()
    id_file = open('./ids/id' + str(i) + '.txt', 'w')
    id_file.write('[')
    for index, row in genreids[0:100].iterrows():
      id_file.write('"spotify:track:' + str(row['track_id']) + '",')
    id_file.write(']')
    id_file.close()

  print("Generated " + str(K) + " genres. Song names printed to genres folder. Run this script again to generate Spotify playlist from genre.")

elif (cmd == 'p'):

  print("Which genre number would you like to generate a playlist from?")
  try:
    num = int(input())
  except:
    print("Not a number.")
    exit()

  # Read track ids for playlist
  try:
    file = open('./ids/id' + str(num) + '.txt', 'r')
  except:
    print("Could not find that genre, ensure enough genres have been generated first.")
    exit()
  uris = file.read()[:-2] + ']'
  json = '{"uris": ' + uris + '}'

  # Get access token for Spotify API
  resp = requests.post(url='https://accounts.spotify.com/api/token', 
    data={'grant_type': 'refresh_token', 'refresh_token': 
    'AQBgoJNJ5o85mx3pPjt5-Ttbyf4QrCh1WiSvctlY-fchNjsdZpagqHtOAAGoHJxfw3jquBKIByTo16N7BfZFmG8F1anoybqSnQHUJ_14OZubIhv4CEfbwLvIlHE6Q3t0zG0'},
    headers={'Content': 'application/x-www-form-urlencoded', 'Authorization': 
    'Basic MTQ5NTFiYWM3MjYxNDljZGI2ZTA5MjUzZTgxZDVjNjE6YTQ2MmU0OTMxZmQyNDgxMTg0NTZiOTkxZTY4OTBjNzE='})
  token = 'Bearer ' + resp.json()['access_token']

  # Create new empty playlist
  print("Enter playlist name:")
  name = input()
  resp = requests.post(url='https://api.spotify.com/v1/users/3172ndz4p4ozhsz6cho67s4jl2ga/playlists', 
    json={'name': name, 'public': True},
    headers={'Content-Type': 'application/json', 'Authorization': token})
  id = resp.json()['id']

  # Populate playlist
  resp = requests.post(url='https://api.spotify.com/v1/playlists/' + id + '/tracks', 
    data=json,
    headers={'Content-Type': 'application/json', 'Authorization': token})

  print("Playlist created, go to https://open.spotify.com/playlist/" + id)
  
else:
  print("Command not recognized.")