import requests
import os
import pandas as pd
import sklearn as skl
from sklearn.cluster import KMeans
from sklearn import preprocessing

def gen_kmeans(df, K):
  # Read data
  data = df[['track_popularity', 
          'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
  data = preprocessing.normalize(data)

  # Cluster songs into genres
  model = KMeans(n_clusters=K)
  predictions = model.fit_predict(data)

  return predictions

def playlist(num, name):
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
  #print("Enter playlist name:")
  #name = input()
  resp = requests.post(url='https://api.spotify.com/v1/users/3172ndz4p4ozhsz6cho67s4jl2ga/playlists', 
    json={'name': name, 'public': True},
    headers={'Content-Type': 'application/json', 'Authorization': token})
  id = resp.json()['id']

  # Populate playlist
  resp = requests.post(url='https://api.spotify.com/v1/playlists/' + id + '/tracks', 
    data=json,
    headers={'Content-Type': 'application/json', 'Authorization': token})

  print("Playlist created, go to https://open.spotify.com/playlist/" + id)