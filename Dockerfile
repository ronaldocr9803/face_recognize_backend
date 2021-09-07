FROM python:3.8

RUN pip install --upgrade pip

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN pip install opencv-python
RUN pip install elasticsearch>=7.8.0 aiohttp
RUN pip install tqdm

EXPOSE 8000

COPY . .

RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -U pip wheel setuptools \
    # && pip install -r requirements.txt
CMD ["python", "create_db.py"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
