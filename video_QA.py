import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
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

def get_video_qa_prompt(summary):
    """Create QA prompt template with video summary context."""
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
You are a helpful assistant analyzing YouTube video content.

Here is a summary of the video:
{summary}

Here are the most relevant transcript segments:
{{context}}

Question: {{question}}

Answer based on the video content. If the answer involves specific details, try to reference approximate timestamps when available.

Answer:
"""
    )
    return qa_prompt

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

async def answer_video_question(video_id, question):
    """Answer questions about video content using transcript and summary."""
    # Ensure we have summary (will create if not cached)
    if video_id not in video_cache or "Summary" not in video_cache[video_id]:
        summary = await summarize_video(video_id)
        if isinstance(summary, dict) and "error" in summary:
            return summary
    else:
        print(f"Using cached video summary for video ID: {video_id}")
        summary = video_cache[video_id]["Summary"]
    
    # Get processed transcript chunks (will process if not cached)
    chunks = ensure_processed_transcript(video_id)
    if not chunks:
        return {"error": "No transcript chunks found after processing."}
    
    # Check if vectorstore is already cached
    if "Vectorstore" not in video_cache[video_id]:
        print(f"Creating and caching vectorstore for video ID: {video_id}")
        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            video_cache[video_id]["Vectorstore"] = vectorstore
        except Exception as e:
            return {"error": f"Error creating vectorstore: {str(e)}"}
    else:
        print(f"Using cached vectorstore for video ID: {video_id}")
        vectorstore = video_cache[video_id]["Vectorstore"]
    
    try:
        # Create QA chain
        qa_prompt = get_video_qa_prompt(summary)
        retriever = vectorstore.as_retriever(search_type="similarity", k=4)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt},
        )

        answer = qa_chain(question)
        return answer
        
    except Exception as e:
        print(f"Error answering question: {e}")
        return {"error": f"Error processing question: {str(e)}"}

# Utility functions for cache management
def clear_video_cache(video_id=None):
    """Clear cache for a specific video or all videos."""
    global video_cache
    if video_id:
        if video_id in video_cache:
            del video_cache[video_id]
            print(f"Video cache cleared for video ID: {video_id}")
    else:
        video_cache.clear()
        print("All video cache cleared")

def get_video_cache_stats():
    """Get statistics about what's cached for videos."""
    stats = {}
    for video_id, data in video_cache.items():
        stats[video_id] = {
            "has_transcript": "Transcript" in data,
            "has_clean_transcript": "CleanTranscript" in data,
            "has_transcript_chunks": "TranscriptChunks" in data,
            "has_summary": "Summary" in data,
            "has_vectorstore": "Vectorstore" in data,
            "transcript_length": len(data.get("Transcript", "")),
            "clean_transcript_length": len(data.get("CleanTranscript", "")),
            "chunk_count": len(data.get("TranscriptChunks", []))
        }
    return stats

def get_transcript_preview(video_id, max_chars=500):
    """Get a preview of the transcript for debugging."""
    if video_id in video_cache and "Transcript" in video_cache[video_id]:
        transcript = video_cache[video_id]["Transcript"]
        if len(transcript) > max_chars:
            return transcript[:max_chars] + "..."
        return transcript
    return "No transcript found in cache"

# Advanced utility: Get transcript segments by time range
def get_transcript_segment(video_id, start_time=None, end_time=None):
    """Get transcript segment between specified timestamps (in seconds)."""
    if video_id not in video_cache or "Transcript" not in video_cache[video_id]:
        return "Transcript not found in cache"
    
    transcript = video_cache[video_id]["Transcript"]
    
    # If no time range specified, return full transcript
    if start_time is None and end_time is None:
        return transcript
    
    # Extract segments based on timestamps
    segments = []
    timestamp_pattern = r'\[(\d{2}):(\d{2}):(\d{2})\]([^[]*)'
    matches = re.findall(timestamp_pattern, transcript)
    
    for match in matches:
        hours, minutes, seconds, text = match
        timestamp_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        
        # Check if timestamp is within range
        if start_time is not None and timestamp_seconds < start_time:
            continue
        if end_time is not None and timestamp_seconds > end_time:
            break
            
        segments.append(f"[{hours}:{minutes}:{seconds}]{text}")
    
    return " ".join(segments)

# Example usage functions
def example_usage():
    """Example of how to use the video QA system."""
    video_id = "your_video_id_here"
    
    # Get summary
    summary = summarize_video(video_id)
    print("Summary:", summary)
    
    # Ask questions
    question = "What are the main points discussed in this video?"
    answer = answer_video_question(video_id, question)
    print("Answer:", answer)
    
    # Check cache stats
    stats = get_video_cache_stats()
    print("Cache stats:", stats)