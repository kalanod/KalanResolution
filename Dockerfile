FROM python:3.9

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY checkpoints /app
COPY models /app
COPY scripts /app
COPY static /app
COPY templates /app
COPY interface.py /app



EXPOSE 80

CMD ["gunicorn", "--bind", "0.0.0.0:80", "interface:app"]