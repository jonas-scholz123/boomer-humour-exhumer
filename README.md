# boomer humour exhumer
A PyTorch Image classifier that detects boomer humour.

<p float="left" align="middle">
  <img src="/images/boomer.jpg" width="30%" /> 
  <img src="./images/non_boomer.jpg" width="30%" />
</p>
A boomer meme (left) with it's distinctive art-style and a generic twitter meme on the right.



Achieves ~80% accuracy.

## Training Data

We scrape reddit using RedditDownloader https://github.com/shadowmoose/RedditDownloader 
to gather 11000 boomer-memes from the subreddits /r/Boomerhumour and /r/Boomershumor.

We scrape a similar number of non-boomer (general purpose) memes from /r/me_irl, /r/meirl, /r/WhitePeopleTwitter, /r/BlackPeopleTwitter, /r/196.

## How To Run

Install requirements.txt using pip:
```
pip3 install -r requirements.txt
```

Currently needs data read in through dataOrganiser.py, will work on an API to query the model directly with a single image or a batch of images.
