import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import time
import json
import requests
import xml.etree.ElementTree as ET
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-001", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Enhanced cache for video transcripts and summaries
video_cache = {}

# def get_transcript(video_id):
#     """Fetch and cache video transcript with timestamps."""
#     # Check if transcript is already cached
#     if video_id in video_cache and "Transcript" in video_cache[video_id]:
#         print(f"Using cached transcript for video ID: {video_id}")
#         return video_cache[video_id]["Transcript"]
    
#     try:
#         transcript = YouTubeTranscriptApi()
#         caption = transcript.fetch(video_id)
#         # print(caption)
#         formatted_lines = []
#         for snippet in caption.snippets:
#             total_seconds = int(snippet.start)
#             hours = total_seconds // 3600
#             minutes = (total_seconds % 3600) // 60
#             seconds = total_seconds % 60
#             timestamp = f"[{hours:02}:{minutes:02}:{seconds:02}]"
#             formatted_line = f"{timestamp} {snippet.text}"
#             formatted_lines.append(formatted_line)
        
#         full_transcript = " ".join(formatted_lines)
        
#         # Initialize cache structure for this video
#         if video_id not in video_cache:
#             video_cache[video_id] = {}
#         video_cache[video_id]["Transcript"] = full_transcript
        
#         return full_transcript
        
#     except Exception as e:
#         print(f"Unexpected error fetching transcript: {e}")
#         return ''

def fetch_transcript(video_id):
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    # }

    url = f'https://www.youtube.com/watch?v={video_id}'
    resp = requests.get(url)
    html = resp.text

    initial_data = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});', html)
    if not initial_data:
        raise Exception("Could not find ytInitialPlayerResponse")

    data = json.loads(initial_data.group(1))
    captions = data.get('captions')
    if not captions:
        raise Exception("No captions available")

    tracks = captions['playerCaptionsTracklistRenderer']['captionTracks']
    transcript_url = tracks[0]['baseUrl']
    transcript_xml = requests.get(transcript_url).text
    root = ET.fromstring(transcript_xml)

    transcript = []
    for elem in root.findall('text'):
        start = float(elem.attrib['start'])
        dur = float(elem.attrib.get('dur', 0))
        text = elem.text or ''
        transcript.append({
            'start': start,
            'duration': dur,
            'text': text.replace('\n', ' ')
        })

    return transcript

def get_transcript(video_id):
    """Fetch and cache video transcript with timestamps."""
    # Check if transcript is already cached
    if video_id in video_cache and "Transcript" in video_cache[video_id]:
        print(f"Using cached transcript for video ID: {video_id}")
        return video_cache[video_id]["Transcript"]
    
    try:
        captions = fetch_transcript(video_id)
        if not captions:
            print(f"No transcript found for video ID: {video_id}")
            return ''
        # print(caption)
        formatted_lines = []
        for snippet in captions:
            total_seconds = int(snippet['start'])
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            timestamp = f"({hours:02}:{minutes:02}:{seconds:02})"
            formatted_line = f"{timestamp} {snippet['text']}"
            formatted_lines.append(formatted_line)
        
        full_transcript = " ".join(formatted_lines)
        
        # Initialize cache structure for this video
        if video_id not in video_cache:
            video_cache[video_id] = {}
        video_cache[video_id]["Transcript"] = full_transcript
        
        return full_transcript
        
    except Exception as e:
        print(f"Unexpected error fetching transcript: {e}")
        return ''

def get_clean_transcript(video_id):
    """Get transcript without timestamps for better processing."""
    # Check if clean transcript is already cached
    if video_id in video_cache and "CleanTranscript" in video_cache[video_id]:
        print(f"Using cached clean transcript for video ID: {video_id}")
        return video_cache[video_id]["CleanTranscript"]
    
    # Get the full transcript (will use cache if available)
    full_transcript = get_transcript(video_id)
    if not full_transcript:
        return ''
    
    # Remove timestamps to get clean text
    clean_transcript = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', full_transcript)
    clean_transcript = re.sub(r'\s+', ' ', clean_transcript).strip()
    
    # Cache the clean transcript
    video_cache[video_id]["CleanTranscript"] = clean_transcript
    return clean_transcript

def chunk_transcript(transcript, chunk_size=1000, overlap=200):
    """Split transcript into overlapping chunks for better context preservation."""
    if not transcript:
        return []
    
    words = transcript.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append(Document(page_content=chunk_text))
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a helpful assistant tasked with summarizing YouTube video content.

The input is a transcript of the video formatted as a continuous string. Each sentence is preceded by a timestamp in the format [hh:mm:ss], followed by the spoken text. The entire transcript is space-separated without line breaks.

Example:
[00:00:00] So, I've been coding since 2012, and I [00:00:03] really wish someone told me these 10 [00:00:07] things before I wasted years figuring them out...

Your task is to:
1. **Summarize**: Provide a summary of the video content, focusing on the main points, key takeaways, and any important details that would help someone understand the video's purpose and content without watching it.
2. List the **Main Points Covered** in the video in details using bullet points. Also include timestamps for clarity.

**Transcript**:
{text}

**Summary**:
"""
)

def ensure_processed_transcript(video_id):
    """Ensure transcript chunks are processed and cached for a video."""
    if video_id not in video_cache:
        video_cache[video_id] = {}
    
    # Check if processed chunks are already cached
    if "TranscriptChunks" in video_cache[video_id]:
        return video_cache[video_id]["TranscriptChunks"]
    
    # Get clean transcript (will use cache if available)
    clean_transcript = get_clean_transcript(video_id)
    if not clean_transcript:
        return []
    
    # Create and cache transcript chunks
    chunks = chunk_transcript(clean_transcript)
    video_cache[video_id]["TranscriptChunks"] = chunks
    return chunks

async def summarize_video(video_id):
    """Summarize video transcript with caching."""
    # Check if summary is already cached
    if video_id in video_cache and "Summary" in video_cache[video_id]:
        print(f"Using cached video summary for video ID: {video_id}")
        return video_cache[video_id]["Summary"]
    
    # Get transcript (will use cache if available)
    transcript = get_transcript(video_id)
    if not transcript:
        return {"error": "No transcript found or unable to fetch transcript."}
    
    try:
        
        # Create document from transcript
        transcript_docs = Document(page_content=transcript)
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="stuff",
            prompt=summary_prompt
        )
        summary = summary_chain.run([transcript_docs])
        
        # Cache the summary
        if video_id not in video_cache:
            video_cache[video_id] = {}
        video_cache[video_id]["Summary"] = summary
        
        return summary
        
    except Exception as e:
        print(f"Error creating video summary: {e}")
        return {"error": f"Error creating summary: {str(e)}"}