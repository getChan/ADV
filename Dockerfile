FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER 9511chn@gmail.com

COPY ./ADV/ /usr/src/app

WORKDIR /usr/src/app/
RUN pip3 install -r requirements.txt
WORKDIR /usr/src/app/server/

EXPOSE 80 
CMD python3 setup.py runserver --host=0.0.0.0 --port=80
