import praw
import urllib

reddit = praw.Reddit(client_id='2Yo3b1SC6tQA1Q',
                     client_secret='IzLIcnVpZySXwiDNBRlmBMu-TEQ',
                     user_agent='image retrieval v1.0 by /u/JonestheOwner')

urls = []
for submission in reddit.subreddit('earthporn').top(limit=10):
	urls.append(submission.url)

for i, url in enumerate(urls):
	if '.jpg' in url:
		urllib.urlretrieve(url,'{:04d}.jpg'.format(i))