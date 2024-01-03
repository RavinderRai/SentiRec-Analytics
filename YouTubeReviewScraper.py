from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

class YouTubeReviewData:    
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def search_videos(self, search_query, max_results=5):
        """
        Search for YouTube videos based on a given query and retrieve additional information including closed captions.

        Parameters:
        - search_query (str): The search query used to find relevant videos on YouTube.
        - max_results (int): The maximum number of videos to retrieve. Defaults to 5.

        Returns:
        List[dict]: A list of dictionaries, each containing information about a video, including:
            - 'video_id' (str): The unique identifier for the video.
            - 'title' (str): The title of the video.
            - 'video_link' (str): The YouTube link to the video.
            - 'channel_name' (str): The name of the channel that uploaded the video.
            - 'cc_text' (str): The closed captions text for the video. This is the review text.

        Note:
        - Videos with titles containing specific strings ('VS', 'vs', 'Vs') are excluded, as they indicate videos that aren't reviews specific to the  
        product in the search query.
        - The 'cc_text' field may contain an empty string if closed captions are not available.
        """       
        
        search_response = self.youtube.search().list(
            q=search_query,
            type='video',
            part='id, snippet',
            maxResults=max_results
        ).execute()
        
        
        """ old version
        video_ids = [result['id']['videoId'] for result in search_response.get('items', [])]
        titles = [result['snippet']['title'] for result in search_response.get('items', [])]
        
        #any videos with vs in the title means it is not a direct review of the specific headphone, so we will not want those
        strings_to_check = ["VS", "vs", "Vs"]
        indices_to_remove = [i for i, title in enumerate(titles) if any(s in title for s in strings_to_check)]
        # Remove items from the video_ids list using the indices
        video_ids = [video_ids[i] for i in range(len(video_ids)) if i not in indices_to_remove]
        """
        
        
        videos_info = []
        for result in search_response.get('items', []):
            video_id = result['id']['videoId']
            title = result['snippet']['title']
            video_link = f'https://www.youtube.com/watch?v={video_id}'
            channel_name = result['snippet']['channelTitle']

            # Check and remove unwanted titles
            strings_to_check = ["VS", "vs", "Vs"]
            if not any(s in title for s in strings_to_check):
                review_text = self.fetch_captions(video_id)
                videos_info.append({
                    'video_id': video_id,
                    'title': title,
                    'video_link': video_link,
                    'channel_name': channel_name,
                    'review_text': review_text
                })

        return videos_info
    
    def fetch_captions(self, video_id):
        """
        Get the closed captions. 

        Parameters:
        - video_id (str): The video id which is obtained in search_videos.
        
        Returns:
        String: Closed caption text of a youtube video
        """
        try:
            # Retrieve the transcript for the video
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            cc_text = ""

            # Concatenate the transcript text
            for entry in transcript:
                cc_text += ' ' + entry['text']
                
            cc_text = cc_text.replace('\n', ' ')
            return cc_text

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
    def review_lists(self, search_query, max_results=7):
        """
        Older function that gets the video data, extracts the video ids, and the produces a list of captions. This is done in the search_query function 
        anyways now so this shouldn't be needed anymore.
        
        Parameters:
        - search_query (str): The query you would enter in the youtube search bar.
        - max_results (int): The number of results you want to get video data for. Default is 7.
        
        Returns:
        List: List of the closed caption text of the youtube videos obtained from search_videos.
        """        
        # Get the video ids for the top x reviews
        video_info = self.search_videos(search_query=search_query, max_results=max_results)
        video_ids = [dct['video_id'] for dct in video_info]

        video_review_texts = [self.fetch_captions(video_ids[i]) for i in range(0, len(video_ids))]

        return video_review_texts
