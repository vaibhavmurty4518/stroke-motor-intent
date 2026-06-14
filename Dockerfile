FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Retrain model fresh using THIS server's sklearn version
RUN python startup.py

EXPOSE 5000

CMD ["python", "app/app.py"]
