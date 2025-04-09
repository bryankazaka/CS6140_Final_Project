import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sklearn as skl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import shutil
import time
import random
import requests
import json
from collections import Counter
from enhanced_clustering import EnhancedClustering
from kmeans import gen_kmeans
from kmeans import playlist
from knn import recommend
import LLM
import Seq2Seq

### Functions

def gen_forest(df):
  model = EnhancedClustering(method='ensemble')
  X = df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']].values
  y = df['playlist_genre'].values
  df_train = pd.DataFrame(X, columns=[
      'danceability','energy','loudness','speechiness',
      'acousticness','instrumentalness','valence','tempo'
  ])
  
  model.fit(df_train)
  labels = model.predict(df_train)
  return labels

def name_llm(genres):
  titles = [0] * len(genres)
  for i in range(len(genres)):
    prompt = LLM.create_zero_shot_prompt(genres[i])
    resp = LLM.generate_title_with_mistral(prompt)
    split = resp.split('\"')
    try:
      titles[i] = split[1].replace(' ', '_').replace(':', ';')
    except:
      print("Failed to parse title. Mistral response: ")
      print(resp)
    time.sleep(1)
  return titles

def name_seq(df, genres):
  track_tokens = df['track_name'].apply(Seq2Seq.tokenize)
  title_tokens = df['playlist_subgenre'].apply(Seq2Seq.tokenize)
  input_vocab = Seq2Seq.build_vocab(track_tokens)
  target_vocab = Seq2Seq.build_vocab(title_tokens)
  inputs = torch.tensor([Seq2Seq.encode(t, input_vocab, Seq2Seq.MAX_INPUT_LEN) for t in track_tokens])
  targets = torch.tensor([Seq2Seq.encode(t, target_vocab, Seq2Seq.MAX_OUTPUT_LEN) for t in title_tokens])
  X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.1, random_state=42)

  # Instantiate model
  embed_size = 64
  hidden_size = 128
  attention = Seq2Seq.Attention(hidden_size)
  encoder = Seq2Seq.Encoder(len(input_vocab), embed_size, hidden_size)
  decoder = Seq2Seq.AttnDecoder(len(target_vocab), embed_size, hidden_size, attention)
  model = Seq2Seq.Seq2Seq(encoder, decoder, len(target_vocab), Seq2Seq.MAX_OUTPUT_LEN)

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss(ignore_index=0)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  NUM_EPOCHS = 10
  BATCH_SIZE = 32
  epoch_losses = []  # To store loss for each epoch

  for epoch in range(NUM_EPOCHS):
      model.train()
      epoch_loss = 0

      for i in range(0, len(X_train), BATCH_SIZE):
          src = X_train[i:i+BATCH_SIZE]
          trg = y_train[i:i+BATCH_SIZE]

          optimizer.zero_grad()
          output = model(src, trg)

          output = output[:, 1:].reshape(-1, model.target_vocab_size)
          trg = trg[:, 1:].reshape(-1)

          loss = criterion(output, trg)
          loss.backward()
          optimizer.step()

          epoch_loss += loss.item()

      avg_loss = epoch_loss / (len(X_train) // BATCH_SIZE)
      epoch_losses.append(avg_loss)
      print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
      print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
      
  titles = [0] * len(genres)
  for i in range(len(genres)):
    namestr = ''
    for name in genres[i]['track_name']:
      namestr += name + ' '
    titles[i] = Seq2Seq.predict_title(namestr, model, input_vocab, target_vocab, Seq2Seq.MAX_INPUT_LEN, Seq2Seq.MAX_OUTPUT_LEN).replace(' ', '_').replace(':', ';')
  
  return titles
    

### Script begins

while True:

  print("[G]enerate genres or create a Spotify [p]laylist?")
  print("Press any other key to exit.")
  try:
    cmd = input().lower()[0]
  except:
    break

  if cmd == 'g':

    # Load data
    df = pd.read_csv('spotify_songs.csv').drop_duplicates(subset=['track_id'])
    df['track_name'] = df['track_name'].str.strip().str.title()
    df['track_artist'] = df['track_artist'].str.strip().str.title()
    df['playlist_genre'] = df['playlist_genre'].str.strip().str.lower()
    df['playlist_subgenre'] = df['playlist_subgenre'].str.strip().str.lower()
    labels = []
    names = []
    cmd = ''

    # Generate genres
    while cmd != 'r' and cmd != 'k':
      print("Generate genres with [R]andom Forest or [K]-means?")
      cmd = input().lower()[0]
    if cmd == 'r':
      labels = gen_forest(df)
    else:
      k = -1
      while k < 0:
        print("How many genres should be generated?")
        try:
          k = int(input())
        except:
          print("Not a number.")
      labels = gen_kmeans(df, k)

    # Split data into genres
    k = np.max(labels)
    genres = [0] * k
    for i in range(k):
      genres[i] = df[labels == i].dropna()

    # Generate genre names
    while cmd != 'l' and cmd != 's':
      print("Generate names with [L]LM or [S]eq-to-seq?")
      cmd = input().lower()[0]
    if cmd == 'l':
      names = name_llm(genres)
    else:
      names = name_seq(df, genres)

    # Create directories
    if os.path.exists('genres'):
      shutil.rmtree('genres')
    os.makedirs('genres')
    if os.path.exists('ids'):
      shutil.rmtree('ids')
    os.makedirs('ids')

    # Write to file
    for i in range(k):
      file = open('./genres/genre' + str(i) + '_' + names[i] + '.txt', 'w')
      for index, row in genres[i].iterrows():
        file.write(str((row['track_artist'] + ' - ' + row['track_name']).encode(errors='replace'))[2:-1] + '\n')
      genreids = genres[i][['track_id']]
      id_file = open('./ids/id' + str(i) + '.txt', 'w')
      id_file.write('[')
      for index, row in genreids[0:100].iterrows():
        id_file.write('"spotify:track:' + str(row['track_id']) + '",')
      id_file.write(']')
      id_file.close()
    
    print('Tracklists saved to genres folder. Use the [p]laylists command to convert into Spotify playlists.')

  elif cmd == 'p':
    while cmd != 'e' and cmd != 'r':
      print('Create a playlist from an [e]xisting genre or generate [r]ecommendations?')
      cmd = input().lower()[0]
    if cmd == 'e':
      id = -1
      while id < 0:
        print("Enter ID of genre to convert to playlist:")
        try:
          id = int(input())
        except:
          print("Not a number.")
      playlist(id)
    else:
      df = pd.read_csv('spotify_songs.csv').drop_duplicates(subset=['track_id'])
      recommend(df)

  else:
    break
