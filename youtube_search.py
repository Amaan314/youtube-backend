import aiohttp
import matplotlib
import pandas as pd
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns   
import isodate
import base64
import io
import os

matplotlib.use('Agg')
sns.set()

API_KEY_VIDEO = os.getenv('API_KEY_VIDEO')
BASE_URL = "https://www.googleapis.com/youtube/v3"
API_URL = os.getenv('MODEL_API_URL')

async def get_sentiments(comments):
    async with aiohttp.ClientSession() as session:
        try:
            payload = {"comments": comments}
            async with session.post(API_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("sentiments", [])
                else:
                    print(f"Error: Received {response.status} from API")
                    return []
        except Exception as e:
            print(f"Error in get_sentiment_async: {e}")
            return []


async def analyze_comments(comments):
    sentiment_labels = await get_sentiments(comments)
    sentiment_counts = Counter(sentiment_labels)
    result = {
        'Positive': sentiment_counts.get('Positive', 0),
        'Negative': sentiment_counts.get('Negative', 0),
        'Neutral': sentiment_counts.get('Neutral', 0)
    }
    return result


async def fetch_video_data(search_query, max_results, sort_by='relevance'):
    url = f"{BASE_URL}/search?part=snippet&type=video&q={search_query}&key={API_KEY_VIDEO}&maxResults={max_results}&order={sort_by}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response_body = await response.text()
            if response.status == 200:
                data = await response.json()
                if 'items' in data and data['items']:
                    return data
                else:
                    return {"items": []}
            else:
                print(f"Failed to fetch video data. Status code: {response.status}")
                print(f"Response body: {response_body}")
                return {"items": []}
            
async def fetch_channel_details(channel_id):
    url = f"{BASE_URL}/channels?part=snippet%2CcontentDetails%2Cstatistics&id={channel_id}&key={API_KEY_VIDEO}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                if result["items"]:
                    return result["items"][0].get("statistics", {}), result["items"][0].get("snippet", {})
            return {}, {}

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

async def get_data(search_query, max_videos, sort_by, max_com, ord):
    video_data = await fetch_video_data(search_query=search_query, max_results=max_videos, sort_by=sort_by)
    detailed_video_data = []
    if video_data and "items" in video_data:
        for video in video_data["items"]:
            if 'videoId' in video["id"]:
                video_id = video["id"]["videoId"]
            else:
                continue
            video_details = await fetch_video_details(video_id=video_id)
            detailed_video_data.append(video_details)
        return detailed_video_data
    return None

async def viz_combined(df, plot_type='total'):
    if len(df) <= 20:
        fig, ax = plt.subplots(figsize=(12, 8))
        bottom_margin = 0.25 + 0.07 * (len(df) // 5)
    elif len(df) <= 40:
        fig, ax = plt.subplots(figsize=(14, 10))
        bottom_margin = 0.17 + 0.05 * (len(df) // 5)
    else:
        fig, ax = plt.subplots(figsize=(18, 14))
        bottom_margin = 0.1 + 0.05 * (len(df) // 5)
        
    colors = ['#03045e','#023e8a','#0077b6','#0096c7','#00b4d8','#48cae4','#90e0ef','#ade8f4']
    sns.set_style('white')

    if plot_type == 'total':
        ax.bar(df.index + 1, df['Views'], color=colors)

        if plot_type == 'total':
            ax2 = ax.twinx()
            ax2.plot(df.index + 1, df['Likes'], color='crimson', marker='D')
            ax2.set_ylabel('Likes', fontsize=13, fontweight='bold')
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        else:
            raise ValueError("Invalid plot_type. Use 'total' or 'percent'.")

        ax.set_ylabel('Views', fontsize=13, fontweight='bold')
        plt.title('Views and Likes Graph' if plot_type == 'total' else 'Views and Likes Percentage by Views', fontsize=16, fontweight='bold')

    elif plot_type == 'engagement_rate':
        df['engagement_rate'] = (df['Likes'] + df['Comments']) / df['Views']
        ax.bar(df.index + 1, df['engagement_rate'], color=colors)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(0, df['engagement_rate'].max() * 1.1)
        ax.set_ylabel('Engagement Rate (%)', fontsize=13, fontweight='bold')
        plt.title('Engagement Rate by Video', fontsize=16, fontweight='bold')

    elif plot_type == 'composite_score':
        df['engagement_rate'] = (df['Likes'] + df['Comments']) / df['Views']
        df['composite_score'] = (df['Views'] * 0.4) + (df['Likes'] * 0.2) + (df['Comments'] * 0.2) + (df['engagement_rate'] * 0.1) + (df['Subscribers'] * 0.1)
        ax.bar(df.index + 1, df['composite_score'], color=colors)
        ax.set_ylabel('Composite Score', fontsize=13, fontweight='bold')
        plt.title('Composite Score by Video', fontsize=16, fontweight='bold')

    else:
        raise ValueError("Invalid plot_type. Use 'total', 'percent', 'engagement_rate', or 'composite_score'.")

    while len(colors) < len(df):
        colors.extend(colors)

    handles = [plt.Line2D([0], [0], color=c, marker='o', label=f"{num}: {name}", markersize=10, linestyle='None') 
               for num, name, c in zip(df.index + 1, df['Title'], colors)]

    legend = ax.legend(handles=handles, title='Title Mapping', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.setp(legend.get_title(), fontsize=13, fontweight='bold')

    ax.set_xlabel('Title Number', fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, len(df) + 1))
    ax.set_xticklabels(range(1, len(df) + 1))
    if plot_type == 'total' or plot_type == 'percent' or plot_type == 'composite_score':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2%}'))

    plt.rcParams['font.family'] = 'DejaVu Sans'
    if len(df) > 40:
        plt.tight_layout()

    fig.subplots_adjust(bottom=bottom_margin)
    
    for text in legend.get_texts():
        text.set_fontsize(10)
    legend.get_frame().set_linewidth(0.5)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

async def sentiment_viz(df, type='multiple'):
    if type == 'multiple':
        # Stacked bar chart for multiple videos
        ax = df.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#28a745', '#dc3545', '#bfbfbf'])
        plt.title('Sentiment Analysis for Multiple Videos')
        plt.xlabel('Video Index')
        plt.ylabel('Number of Comments')
        plt.xticks(ticks=range(len(df)), labels=[i + 1 for i in range(len(df))], rotation=0)
        plt.legend(['Positive', 'Negative', 'Neutral'])
    elif type == 'single':
        # Non-stacked bar chart for a single video (different bars for each sentiment)
        series = df.iloc[0] 
        ax = series.plot(kind='bar', figsize=(12, 6), color=['#28a745', '#dc3545', '#bfbfbf'])
        plt.title('Sentiment Analysis for a Single Video')
        plt.ylabel('Number of Comments')
        plt.xticks(rotation=0)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

async def search_youtube(query, sort_by='relevance', max_results=5, max_com=5, order='relevance'):
    videos_data = await get_data(search_query=query, 
                                max_videos=max_results, 
                                sort_by=sort_by,
                                max_com=max_com,
                                ord=order)
    # print(f"Videos data fetched: {len(videos_data) if videos_data else 0} videos")
    if not videos_data:
        return None
    structured_data = []
    for video in videos_data:
        try:
            if video:
                video_id = video["id"]
                snippet = video["snippet"]
                statistics = video.get("statistics", {})
                content_details = video.get("contentDetails", {})
                channel_title = snippet["channelTitle"]
                channel_details, channel_snippet = await fetch_channel_details(snippet["channelId"])
                video_link = f"https://www.youtube.com/watch?v={video_id}"

                thumbnails = snippet.get("thumbnails", {})
                thumbnail_url = thumbnails.get("high", {}).get("url", "")

                channel_thumbnails = channel_snippet.get("thumbnails", {})
                channel_thumbnail_url = channel_thumbnails.get("high", {}).get("url") or \
                                        channel_thumbnails.get("medium", {}).get("url") or \
                                        channel_thumbnails.get("default", {}).get("url") or ""

                duration_str = content_details.get('duration', 'PT0S')
                duration = isodate.parse_duration(duration_str)
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                formatted_duration = f"{hours:02}:{minutes:02}:{seconds:02}"
                video_info = {
                    "Title": snippet["title"],
                    "Views": int(statistics.get("viewCount", 0)),
                    "Likes": int(statistics.get("likeCount", 0)),
                    "Comments": int(statistics.get("commentCount", 0)),
                    "Upload_date": snippet.get("publishedAt", "N/A"),
                    "Duration": formatted_duration,
                    "Channel": channel_title,
                    "Subscribers": int(channel_details.get("subscriberCount", 0)),
                    'Video_link': video_link,
                    'Thumbnail': thumbnail_url,
                    "Channel_Thumbnail": channel_thumbnail_url,
                    "Description": snippet.get("description", "No description available.")
                }
                structured_data.append(video_info)
        except Exception as e:
            print(f"Error processing video data: {e}")
            continue
    if not structured_data:
        return None
    df = pd.DataFrame(structured_data)
    try:
        df['Upload_date'] = pd.to_datetime(df['Upload_date'].str.split('T').str[0])
        df['Likes(%)'] = (df['Likes']) / (df['Views']) * 100
        df = df[['Title', 'Channel', 'Subscribers', 'Views', 'Likes', 'Likes(%)', 'Duration', 'Upload_date', 'Comments', 'Video_link', 'Thumbnail', 'Channel_Thumbnail', 'Description']]
        df['Title'] = df.apply(lambda row: f'<a href="{row["Video_link"]}" target="_blank">{row["Title"]}</a>', axis=1)
        return df
    except Exception as e:
        print(f"An error occurred while processing the data: {e}")
        return None