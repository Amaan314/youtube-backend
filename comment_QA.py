from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import aiohttp
import re
import os
API_KEY_COMMENTS = os.getenv('API_KEY_COMMENTS')

API_KEY_VIDEO = os.getenv('API_KEY_VIDEO')
BASE_URL = "https://www.googleapis.com/youtube/v3"

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
embed_model = "models/text-embedding-004"

# === QA Chain ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# Enhanced cache structure
cache = {}

async def fetch_comments_data(video_id, max_results=100, order="relevance"):
    url = f"{BASE_URL}/commentThreads?part=snippet&videoId={video_id}&key={API_KEY_COMMENTS}&maxResults={max_results}&order={order}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return None

async def fetch_channel_details(channel_id):
    url = f"{BASE_URL}/channels?part=snippet%2CcontentDetails%2Cstatistics&id={channel_id}&key={API_KEY_VIDEO}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                if result["items"]:
                    return result["items"][0].get("statistics", {})
            return {}

async def fetch_video_details(video_id):
    video_details_url = f"{BASE_URL}/videos?part=snippet,statistics,contentDetails&id={video_id}&key={API_KEY_VIDEO}"
    async with aiohttp.ClientSession() as session:
        async with session.get(video_details_url) as response:
            if response.status == 200:
                result = await response.json()
                if result["items"]:
                    video_details = result["items"][0]
                    return video_details
            return None

async def extract_comments(video_id):
    if video_id in cache and "Comments" in cache[video_id]:
        print(f"Using cached comments for video ID: {video_id}")
        return cache[video_id]["Comments"]

    video = await fetch_video_details(video_id)
    if not video:
        return []

    com_cnt = int(video['statistics']['commentCount']) if 'commentCount' in video['statistics'] else 0
    
    all_comments = []

    # Fetch comments by relevance
    if com_cnt > 0:
        com_data_rel = await fetch_comments_data(video_id, min(100, com_cnt), "relevance")
        if com_data_rel and "items" in com_data_rel:
            for item in com_data_rel["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append({
                    "Author": snippet["authorDisplayName"],
                    "CommentText": snippet["textOriginal"],
                    "LikeCount": snippet["likeCount"],
                    "PublishDate": snippet["publishedAt"],
                    "AuthorLogoUrl": snippet["authorProfileImageUrl"]
                })

    # Fetch additional comments by time if needed
    if com_cnt > 100:
        remaining_comments = com_cnt - len(all_comments)
        if remaining_comments > 0:
            com_data_time = await fetch_comments_data(video_id, min(100, remaining_comments), "time")
            if com_data_time and "items" in com_data_time:
                for item in com_data_time["items"]:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    all_comments.append({
                        "Author": snippet["authorDisplayName"],
                        "CommentText": snippet["textOriginal"],
                        "LikeCount": snippet["likeCount"],
                        "PublishDate": snippet["publishedAt"],
                        "AuthorLogoUrl": snippet["authorProfileImageUrl"]
                    })

    # Initialize cache structure for this video
    if video_id not in cache:
        cache[video_id] = {}
    cache[video_id]["Comments"] = all_comments
    return all_comments

def remove_links(comment):
    return re.sub(r'https?://\S+|www\.\S+', '', comment).strip()

def has_multiple_timestamps(comment):
    timestamps = re.findall(r'\d{1,2}(:\d{2}){1,3}', comment)
    return len(timestamps) > 3

def has_char_timestamps(comment):
    pattern = r'^\s*\d{1,2}(:\d{2}){1,3}[\s\u200B]*[-:|]*[\s\u200B]+[a-zA-Z]{2,}.*$'
    matches = re.findall(pattern, comment, flags=re.MULTILINE)
    return len(matches) > 3

def is_code_heavy(comment):
    comment = comment.strip()
    high_signal_keywords = re.findall(
        r'\b(def|class|return|import|lambda|function|const|var|=>|try|except|elif)\b',
        comment
    )
    code_structures = re.findall(r'(==|===|{|}|\[|\]|::|->|=)', comment)
    indented_lines = re.findall(r'^\s{4,}', comment, re.MULTILINE)
    num_lines = comment.count('\n') + 1
    score = 0
    if len(high_signal_keywords) >= 2:
        score += 2
    if len(code_structures) >= 3:
        score += 2
    if len(indented_lines) >= 2:
        score += 2
    if num_lines >= 3:
        score += 1
    if len(high_signal_keywords) > 0 and len(code_structures) > 0:
        score += 1
    return score >= 5

def clean_and_filter_comment(comment):
    comment = remove_links(comment)
    if is_code_heavy(comment):
        return None
    if has_multiple_timestamps(comment) and not has_char_timestamps(comment):
        return None
    if len(comment.strip()) == 0:
        return None
    return comment.strip()

def process_comments(comment_list):
    cleaned_comments = []
    for comment in comment_list:
        cleaned = clean_and_filter_comment(comment)
        if cleaned:
            cleaned_comments.append(cleaned)
    return cleaned_comments

def format_comment(comment):
    return f"{comment['Author']}: {comment['CommentText']} (Likes: {comment['LikeCount']})"

custom_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an assistant tasked with summarizing YouTube comments.

Summarize the main points expressed in the following comments:
- Highlight common themes.
- Capture both praise and criticism.
- Also mention if you see anything special or surprising.

Comments with their author names starting with @:
{text}

Summary:
"""
)

def chunk_comments(comments, chunk_size=20):
    chunks = []
    for i in range(0, len(comments), chunk_size):
        chunk = comments[i:i + chunk_size]
        if chunk:
            chunks.append(Document(page_content="\n".join(chunk)))
    return chunks

embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=GOOGLE_API_KEY,
    model=embed_model
)

def get_qa_prompt(summary):
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
    You are a helpful assistant analyzing YouTube comments.

    Here is a summary of all comments:
    {summary}

    Here are the most relevant comment chunks along with their author names:
    {{context}}

    Question: {{question}}

    Answer:
    """
    )
    return qa_prompt

def ensure_processed_comments(video_id):
    """Ensure comments are processed and cached for a video."""
    if video_id not in cache:
        cache[video_id] = {}
    
    # Check if processed comments are already cached
    if "ProcessedComments" in cache[video_id]:
        return cache[video_id]["ProcessedComments"]
    
    # Get raw comments (will use cache if available)
    comments = cache[video_id].get("Comments", [])
    if not comments:
        return []
    
    # Process and cache the formatted/cleaned comments
    formatted_comments = [format_comment(comment) for comment in comments]
    cleaned_comments = process_comments(formatted_comments)
    
    cache[video_id]["ProcessedComments"] = cleaned_comments
    return cleaned_comments

async def summarize_comments(video_id):
    # Check if summary is already cached
    if video_id in cache and "Summary" in cache[video_id]:
        print(f"Using cached summary for video ID: {video_id}")
        return cache[video_id]["Summary"]
    
    # Ensure we have comments (will fetch if not cached)
    if video_id not in cache or "Comments" not in cache[video_id]:
        comments = await extract_comments(video_id)
        if not comments:
            # return {"error": "No comments found or unable to fetch comments."}
            print({"error": "No comments found or unable to fetch comments."})
            return 'No comment summary available.'
    
    # Get processed comments (will process if not cached)
    cleaned_comments = ensure_processed_comments(video_id)
    if not cleaned_comments:
        # return {"error": "No valid comments found after cleaning."}
        print({"error": "No valid comments found after cleaning."})
        return 'No comment summary available.'
    
    # Create summary
    comment_text = "\n\n".join(cleaned_comments)
    all_comments_docs = Document(page_content=comment_text)

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=custom_prompt
    )
    
    summary = summary_chain.run([all_comments_docs])
    
    # Cache the summary
    cache[video_id]["Summary"] = summary
    return summary

async def answer_question(video_id, question):
    # Ensure we have summary (will create if not cached)
    if video_id not in cache or "Summary" not in cache[video_id]:
        summary = await summarize_comments(video_id)
        if isinstance(summary, dict) and "error" in summary:
            return summary
    else:
        print(f"Using cached summary for video ID: {video_id}")
        summary = cache[video_id]["Summary"]
    
    # Get processed comments (will process if not cached)
    cleaned_comments = ensure_processed_comments(video_id)
    if not cleaned_comments:
        return {"error": "No valid comments found after cleaning."}
    
    # Check if vectorstore is already cached
    if "Vectorstore" not in cache[video_id]:
        print(f"Creating and caching vectorstore for video ID: {video_id}")
        chunked_docs = chunk_comments(cleaned_comments)
        vectorstore = FAISS.from_documents(chunked_docs, embedding_model)
        cache[video_id]["Vectorstore"] = vectorstore
    else:
        print(f"Using cached vectorstore for video ID: {video_id}")
        vectorstore = cache[video_id]["Vectorstore"]
    
    # Create QA chain
    qa_prompt = get_qa_prompt(summary)
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

# Optional: Function to clear cache for a specific video or all videos
def clear_cache(video_id=None):
    """Clear cache for a specific video or all videos."""
    global cache
    if video_id:
        if video_id in cache:
            del cache[video_id]
            print(f"Cache cleared for video ID: {video_id}")
    else:
        cache.clear()
        print("All cache cleared")

# Optional: Function to get cache statistics
def get_cache_stats():
    """Get statistics about what's cached."""
    stats = {}
    for video_id, data in cache.items():
        stats[video_id] = {
            "has_comments": "Comments" in data,
            "has_processed_comments": "ProcessedComments" in data,
            "has_summary": "Summary" in data,
            "has_vectorstore": "Vectorstore" in data,
            "comment_count": len(data.get("Comments", [])),
            "processed_comment_count": len(data.get("ProcessedComments", []))
        }
    return stats