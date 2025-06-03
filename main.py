from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from video_QA import get_transcript, summarize_video
app = FastAPI()

origins = [
    "http://localhost:3000", # Your Next.js app
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "YouTube Search API is running"}


@app.get(
    "/video/transcript/{video_id}",
    summary="Get Video Transcript",
    description="Fetches the transcript of a YouTube video with timestamps.",
    tags=["Video"]
)   
def get_video_transcript(video_id: str):
    """Get the full transcript of a YouTube video."""
    try:
        transcript = get_transcript(video_id)
        if not transcript:
            print(f"No transcript found or an issue occurred for video ID: {video_id}")
            return {"results": ''}
        return {
            "results": transcript,
            "length": len(transcript),
            "word_count": len(transcript.split())
        }
    except Exception as e:
        return {"error": str(e)}

@app.get(
    "/video/summarize/{video_id}",
    summary="Summarize Video Content",
    description="Summarizes the content of a YouTube video based on its transcript. Uses caching for improved performance.",
    tags=["Video"]
)
async def video_summary(video_id: str):
    """Generate a summary of YouTube video content."""
    try:
        summary = await summarize_video(video_id)
        if isinstance(summary, dict) and "error" in summary:
            return JSONResponse(
                status_code=404,
                content={"message": summary["error"], "results": None}
            )
        if not summary:
            return JSONResponse(
                status_code=404,
                content={"message": f"No summary available for video ID: {video_id}", "results": None}
            )
        return {"results": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating video summary: {str(e)}")