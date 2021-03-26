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

To determine the boomer-ness of an image, naviagte to /src/ and run:

```
python3 exhume.py /path/to/image
```

For example,

```
python3 exhume.py ../images/boomer.jpg
>>> The image is 98.31% Boomerish
```

or 

```
python3 exhume.py ../images/non_boomer.jpg
>>> The image is 1.09% Boomerish
```


## To Do

- [x] Single image exhumation
- [x] Exhumation as a microservice for querying
- [x] OCR Capability for reading text
- [ ] Embeddings/RNN for encoding text into classification
- [ ] Reddit bot that comments the boomerness of a post.
- [ ] Cloud Hosted

