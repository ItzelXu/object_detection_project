import requests, os

# Create directory for videos
os.makedirs("videos", exist_ok=True)

# Download Video 1
url1 = "https://www.pexels.com/download/video/856340/"
r = requests.get(url1)
with open("videos/video1.mp4", "wb") as f:
    f.write(r.content)

# Download Video 2
url2 = "https://www.pexels.com/download/video/854179/"
r = requests.get(url2)
with open("videos/video2.mp4", "wb") as f:
    f.write(r.content)
