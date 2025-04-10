import requests
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import warnings

def recommend(df):
  # Load 30000 dataset
  songs = pd.DataFrame(df[['track_name', 'track_artist']])
  ids = pd.DataFrame(df[['track_id']])
  data = pd.DataFrame(df[['track_popularity', 
          'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])

  pos_ids = []
  while True:
    try:
      print("Enter the number of a generated genre you like: ")
      pos_ids.append(int(input()))
    except:
      print("Not a number.")
    print("Add another genre? (y/n)")
    response = input().lower()
    while response[0] != 'n' and response[0] != 'y':
      print("Please respond [y]es or [n]o.")
      response = input().lower()
    if response[0] == 'n':
      break

  neg_ids = []
  while True:
    try:
      print("Enter the number of a generated genre you dislike: ")
      neg_ids.append(int(input()))
    except:
      print("Not a number.")
    print("Add another genre? (y/n)")
    response = input().lower()
    while response[0] != 'n' and response[0] != 'y':
      print("Please respond [y]es or [n]o.")
      response = input().lower()
    if response[0] == 'n':
      break

  print("Analyzing songs...")
  pos_indices = []
  for id in pos_ids:
    file = open('ids/id' + str(id) + '.txt')
    text = file.read()
    strs = text.split(',')
    pos_indices.append(ids[ids['track_id'] == strs[0][16:-1]].index.values[0])
    for i in range(1, len(strs) - 1):
      pos_indices.append(ids[ids['track_id'] == strs[i][15:-1]].index.values[0])

  pos_data = data.loc[pos_indices]

  neg_indices = []
  for id in neg_ids:
    file = open('ids/id' + str(id) + '.txt')
    text = file.read()
    strs = text.split(',')
    neg_indices.append(ids[ids['track_id'] == strs[0][16:-1]].index.values[0])
    for i in range(1, len(strs) - 1):
      neg_indices.append(ids[ids['track_id'] == strs[i][15:-1]].index.values[0])

  neg_data = data.loc[neg_indices]

  labels = [1] * len(pos_data)
  labels.extend([0] * len(neg_data))
  user_data = pd.concat([pos_data, neg_data])
  user_data = preprocessing.normalize(user_data)

  # Remove duplicates
  indices = pos_indices
  indices.extend(neg_indices)
  warnings.simplefilter(action='ignore', category='SettingWithCopyWarning')
  songs.drop(index=indices, inplace=True)
  ids.drop(index=indices, inplace=True)
  data.drop(index=indices, inplace=True)
  data = preprocessing.normalize(data)

  # Perform KNN on 30000 dataset using playlist songs as nearest neighbors
  model = KNeighborsClassifier(n_neighbors=int(len(user_data) / 20), weights='distance')
  model.fit(user_data, labels)
  confidence = model.predict_proba(data)[:,1]
  confidence = np.array(confidence)
  order = np.array(np.argsort(confidence))

  # Get access token for Spotify API
  resp = requests.post(url='https://accounts.spotify.com/api/token', 
    data={'grant_type': 'refresh_token', 'refresh_token': 
    'AQBgoJNJ5o85mx3pPjt5-Ttbyf4QrCh1WiSvctlY-fchNjsdZpagqHtOAAGoHJxfw3jquBKIByTo16N7BfZFmG8F1anoybqSnQHUJ_14OZubIhv4CEfbwLvIlHE6Q3t0zG0'},
    headers={'Content': 'application/x-www-form-urlencoded', 'Authorization': 
    'Basic MTQ5NTFiYWM3MjYxNDljZGI2ZTA5MjUzZTgxZDVjNjE6YTQ2MmU0OTMxZmQyNDgxMTg0NTZiOTkxZTY4OTBjNzE='})
  token = 'Bearer ' + resp.json()['access_token']

  # Assemble id list for spotify API
  sorted_ids = np.array(ids)[order]
  track_ids = sorted_ids[:100]
  uris = "["
  for i in range(99):
    uris += '"spotify:track:' + str(track_ids[i])[2:-2] + '",'
  uris += '"spotify:track:' + str(track_ids[99])[2:-2] + '"]'
  json = '{"uris": ' + uris + '}'

  # Create new empty playlist
  print("Analysis complete, creating recommendation playlist.")
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