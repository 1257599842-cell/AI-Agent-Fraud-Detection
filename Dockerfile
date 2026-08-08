# 工程层容器：只装依赖与代码，**不打包数据**。
# 理由：data/ 里是 590,540 行的 parquet + 知识库 + 应然档，几百 MB 且不是代码资产；
# 打进镜像会让镜像巨大、且每次换数据都要重建。运行时用挂载卷。
FROM python:3.12-slim

# LightGBM 需要 libgomp（arm64 wheel 自带，但 slim 基础镜像上仍显式装稳妥）
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# 数据以只读卷挂入：docker run -v $(pwd)/data:/app/data:ro ...
VOLUME ["/app/data"]

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# 启动即加载 59 万行元数据（lifespan），故给足健康检查的启动宽限
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
