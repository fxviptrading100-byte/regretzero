FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . /app
RUN chmod +x start.sh
EXPOSE 7860
CMD ["bash", "start.sh"]