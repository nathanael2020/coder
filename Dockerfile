FROM python:3.8-slim

WORKDIR /sandbox

COPY sandbox_code.py /sandbox/sandbox_code.py

CMD ["python", "sandbox_code.py"]
