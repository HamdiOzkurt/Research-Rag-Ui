FROM python:3.11-slim

WORKDIR /app

# Node.js (npx) is needed for MCP tools (firecrawl-mcp)
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
  && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.simple_copilot_backend:app", "--host", "0.0.0.0", "--port", "8000"]

