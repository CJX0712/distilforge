FROM python:3.13-slim

WORKDIR /app

# 先装依赖锁定文件，利用层缓存
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# 冒烟：跑端到端基准并校验确定性
CMD ["python", "examples/run_demo.py"]
