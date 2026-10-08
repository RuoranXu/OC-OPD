FROM vllm/vllm-openai:v0.10.2
WORKDIR /workspace/OC-OPD
COPY . .
RUN python -m pip install --no-cache-dir -e '.[train,vllm]'
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENTRYPOINT []
CMD ["bash"]
