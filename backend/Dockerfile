FROM python:3.9-buster
RUN apt-get update
RUN apt-get -y install gcc g++ python3-dev unixodbc-dev ffmpeg libsm6 libxext6 tesseract-ocr-all libhdf5-dev

COPY requirements.txt /app/requirements.txt

RUN pip install -r app/requirements.txt

# For some reason numpy breaks sometimes?
RUN pip install --upgrade numpy


WORKDIR /app/src

COPY . /app

CMD ["python3", "exhume_app.py"]