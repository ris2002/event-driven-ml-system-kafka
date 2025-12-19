FROM python:3.11.0-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade fastapi uvicorn
RUN pip install --no-cache-dir -r requirements.txt
COPY deployment_folder /app/deployment_folder  
COPY config /app/config 

#The config folder holds all the model pkl files and data pkl files since it contains a lot of data it is gitignored
EXPOSE 8000
CMD ["uvicorn", "deployment_folder.app:app", "--host", "0.0.0.0", "--port", "8000"]
