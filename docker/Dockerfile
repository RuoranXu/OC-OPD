FROM vllm/vllm-openai:v0.10.2
WORKDIR /workspace/OC-OPD

# Cache the pinned training dependencies separately from repository source.
# examples/test intentionally retains a separate vLLM 0.8.5 environment.
COPY requirements.txt requirements-train.txt requirements-vllm.txt ./
RUN python -m pip install --no-cache-dir \
    -r requirements.txt -r requirements-train.txt -r requirements-vllm.txt

COPY . .
RUN python -m pip install --no-cache-dir --no-deps -e .
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENTRYPOINT []
CMD ["bash"]
