FROM ubuntu
RUN apt-get update
RUN apt --fix-missing update
RUN apt-get -y install python3 python3-pip
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN cd home
#RUN cd face-match-json-dump
COPY * .
RUN pip3 install -r requirements.txt
#RUN pip3 install -r google-cloud-storage==1.32.0 tensorflow==2.3.1 opencv-python==4.4.0.46 annoy==1.17.0 Keras==2.4.3 fastapi==0.68.1 uvicorn==0.15.0 starlette==0.14.2 cmake dlib
ENTRYPOINT python3 facenet-keras-face-match.py