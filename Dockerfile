FROM python:3.11

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000
ENV PORT 5000

WORKDIR /app

#CMD exec gunicorn --bind :$PORT main:app --workers 1 --threads 1 --timeout 0
CMD [ "python", "main.py" ]