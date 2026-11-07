# One container: downloads the data, builds the models, serves the website.
#   docker build -t career-mode-scout .
#   docker run -p 7860:7860 career-mode-scout          -> http://localhost:7860
# Works as-is on Hugging Face Spaces (Docker SDK), Render, Railway or Fly.io.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/run_all.py --quick
ENV PORT=7860
EXPOSE 7860
CMD ["python", "server.py"]
