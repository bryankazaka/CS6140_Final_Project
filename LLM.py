import pandas as pd
import numpy as np
import requests
import json
import random

Mistral_API_Key="473aIIXwkwsZFT9RWpzkyqoFVrR8AcNw"

def create_zero_shot_prompt(playlist_df):
    """
    Create a prompt for the LLM to generate a playlist title without examples
    
    Args:
        playlist_df: DataFrame containing songs for the playlist
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Get a sample of songs to include in the prompt (avoid token limits)
    sample_size = min(5, len(playlist_df))
    sample_songs = playlist_df.sample(sample_size) if len(playlist_df) > sample_size else playlist_df
    
    # Create the prompt
    prompt = f"""Task: Generate a creative, catchy playlist title

Songs in the playlist:
"""
    
    # Add song information
    for i, (_, song) in enumerate(sample_songs.iterrows()):
        prompt += f"{i+1}. \"{song['track_name']}\" by {song['track_artist']}\n"
    
    # Add additional context about the songs
    prompt += "\nAdditional details:\n"
    
    # Calculate average popularity if available
    if 'track_popularity' in playlist_df.columns:
        avg_popularity = playlist_df['track_popularity'].mean()
        prompt += f"- Average track popularity: {avg_popularity:.1f}/100\n"
    
    # Get release date range if available
    if 'track_album_release_date' in playlist_df.columns:
        dates = playlist_df['track_album_release_date'].dropna()
        if not dates.empty:
            min_date = min(dates)
            max_date = max(dates)
            prompt += f"- Release date range: {min_date} to {max_date}\n"
    
    # Note if there are remixes
    has_remixes = any(playlist_df['track_name'].str.contains('Remix', case=False, na=False))
    if has_remixes:
        prompt += "- Contains remix tracks\n"
    
    prompt += "\nGenerate a single, creative playlist title that captures the mood and theme of these songs. The title should be catchy, marketable, and appealing to listeners:"
    
    return prompt

def generate_title_with_mistral(prompt, api_key=Mistral_API_Key, temperature=0.7):
    """
    Generate a playlist title using Mistral Large LLM via API
    
    Args:
        prompt: Text prompt to send to the model
        api_key: API key for the service (defaults to notebook variable Mistral_API_Key)
        temperature: Sampling temperature (higher = more creative)
        
    Returns:
        str: Generated playlist title
    """
    if not api_key:
        raise ValueError("No API key provided.")
    
    url = "https://api.mistral.ai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "mistral-large-latest",  # Use the latest Mistral Large model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 100  # Enough tokens for a title
    }
    
    try:
        print(f"Sending prompt to Mistral API: {prompt[:100]}...")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        
        response_data = response.json()
        
        # Extract the generated title from the response
        title = response_data["choices"][0]["message"]["content"].strip()
        
        # Clean up the title - remove quotes if present and any "Title:" prefix
        title = title.strip('"\'')
        if ":" in title and title.split(":", 1)[0].lower() in ["title", "playlist title", "playlist"]:
            title = title.split(":", 1)[1].strip()
            
        return title
    
    except Exception as e:
        print(f"Error calling Mistral API: {str(e)}")
        return f"Error generating title: {str(e)}"

def create_few_shot_prompt(playlist_df, num_examples=3):
    """
    Create a prompt for the LLM to generate a playlist title with examples (few-shot learning)
    
    Args:
        playlist_df: DataFrame containing songs for the playlist
        num_examples: Number of few-shot examples to include
        
    Returns:
        str: Formatted prompt with examples for the LLM
    """
    # Get a sample of songs to include in the prompt (avoid token limits)
    sample_size = min(5, len(playlist_df))
    sample_songs = playlist_df.sample(sample_size) if len(playlist_df) > sample_size else playlist_df
    
    # Few-shot examples
    examples = [
        {
            "songs": [
                "Don't Start Now by Dua Lipa",
                "Blinding Lights by The Weeknd",
                "Physical by Dua Lipa",
                "Roses (Imanbek Remix) by SAINt JHN",
                "Midnight Sky by Miley Cyrus"
            ],
            "title": "Neon Disco Revival"
        },
        {
            "songs": [
                "bad guy by Billie Eilish",
                "SICKO MODE by Travis Scott",
                "Old Town Road by Lil Nas X",
                "Shallow by Lady Gaga",
                "Truth Hurts by Lizzo"
            ],
            "title": "Chart Toppers: New Classics"
        },
        {
            "songs": [
                "Dynamite by BTS",
                "Watermelon Sugar by Harry Styles",
                "Rain On Me by Lady Gaga",
                "positions by Ariana Grande",
                "Say So by Doja Cat"
            ],
            "title": "Summer Pop Explosion"
        },
        {
            "songs": [
                "Memories (Dillon Francis Remix) by Maroon 5",
                "In Your Eyes (Remix) by The Weeknd",
                "Dream On Me (Remix) by Ella Henderson",
                "Let Me Down Slowly (Remix) by Alec Benjamin",
                "Someone You Loved (Future Humans Remix) by Lewis Capaldi"
            ],
            "title": "Remix Renaissance"
        },
        {
            "songs": [
                "All The Time (Don Diablo Remix) by Zara Larsson",
                "Call You Mine (Keanu Silva Remix) by The Chainsmokers",
                "Close To Me (Red Triangle Remix) by Ellie Goulding",
                "I Don't Care (With Justin Bieber) (Loud Luxury Remix) by Ed Sheeran",
                "Higher Love (Kygo Remix) by Whitney Houston"
            ],
            "title": "EDM Remix Odyssey"
        }
    ]
    
    # Select a subset of examples
    selected_examples = random.sample(examples, min(num_examples, len(examples)))
    
    # Start building the prompt
    prompt = "Task: Generate a creative, catchy playlist title based on the songs in the playlist.\n\n"
    prompt += "Here are some examples of playlists and good titles for them:\n\n"
    
    # Add the few-shot examples
    for i, example in enumerate(selected_examples):
        prompt += f"Example {i+1}:\n"
        prompt += "Songs:\n"
        for j, song in enumerate(example["songs"]):
            prompt += f"{j+1}. {song}\n"
        prompt += f"Title: {example['title']}\n\n"
    
    # Now add the actual songs we want a title for
    prompt += "Now, generate a creative title for this playlist:\n"
    prompt += "Songs:\n"
    
    # Add song information
    for i, (_, song) in enumerate(sample_songs.iterrows()):
        prompt += f"{i+1}. \"{song['track_name']}\" by {song['track_artist']}\n"
    
    # Add additional context about the songs if available
    if 'track_popularity' in playlist_df.columns:
        avg_popularity = playlist_df['track_popularity'].mean()
        prompt += f"\nAverage track popularity: {avg_popularity:.1f}/100\n"
    
    # Check if there are remixes
    has_remixes = any(playlist_df['track_name'].str.contains('Remix', case=False, na=False))
    if has_remixes:
        prompt += "Note: This playlist contains remix tracks.\n"
    
    prompt += "\nTitle:"
    
    return prompt

def generate_title_with_mistral_few_shot(prompt, api_key=Mistral_API_Key, temperature=0.7):
    """
    Generate a playlist title using Mistral Large LLM via API with few-shot examples
    
    Args:
        prompt: Text prompt to send to the model (including few-shot examples)
        api_key: API key for the service
        temperature: Sampling temperature (higher = more creative)
        
    Returns:
        str: Generated playlist title
    """
    if not api_key:
        raise ValueError("No API key provided.")
    
    url = "https://api.mistral.ai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 50  # Enough for a title
    }
    
    try:
        print(f"Sending few-shot prompt to Mistral API...")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        response_data = response.json()
        
        # Extract the generated title from the response
        title = response_data["choices"][0]["message"]["content"].strip()
        
        # Clean up the title - remove quotes if present
        title = title.strip('"\'')
        
        return title
    
    except Exception as e:
        print(f"Error calling Mistral API: {str(e)}")
        return f"Error generating title: {str(e)}"