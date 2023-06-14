# syntax=docker/dockerfile:1
FROM cicirello/pyaction:latest

# install app dependencies
COPY requirements.txt / 
RUN pip install -r requirements.txt