FROM ubuntu
RUN apt-get update
RUN apt --fix-missing update
RUN apt-get -y install python3 python3-pip
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN cd home
COPY facenet-keras-face-match.py .
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ENTRYPOINT python3 facenet-keras-face-match.py
