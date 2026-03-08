IMG=baseten/attn:base
CONTAINER_NAME=attn_dev
VENV=.venv
PYTHON=$(VENV)/bin/python
GPU_MODEL := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | grep -oE '[HB][0-9]+' | tr '[:upper:]' '[:lower:]')
DIST=dist_$(GPU_MODEL)
FLASH_WHL=$(DIST)/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
FLASH3_WHL=$(DIST)/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl
SAGE_SRC=../SageAttention

.PHONY: docker_build docker_run docker_push install_dep install_dep_best_effort install_optional_baselines bench bench_causal clean

docker_build:
	docker build -t $(IMG) -f docker/Dockerfile .

docker_run:
	docker rm -f $(CONTAINER_NAME) || true
	docker run -d \
		-v $(realpath ../):/workspace \
		-it --ipc=host --shm-size 32g \
		--entrypoint bash \
		--gpus all \
		--name $(CONTAINER_NAME) \
		$(IMG) \
		-c 'sleep infinity'

docker_push:
	docker push $(IMG)

install_dep:
	uv sync
	uv pip install --python $(PYTHON) $(FLASH_WHL)
	if [ "$(DIST)" = "dist_h100" ]; then \
		TORCH_CUDA_ARCH_LIST="9.0"; \
		uv pip install --python $(PYTHON) $(FLASH3_WHL); \
	fi
	if [ "$(DIST)" = "dist_b200" ]; then \
		TORCH_CUDA_ARCH_LIST="10.0;10.0a"; \
		uv pip install --python $(PYTHON) -e ../cutile-python/; \
		uv pip install --python $(PYTHON) ../cutile-python/experimental/; \
	fi
	MAX_JOBS=32 EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" \
		uv pip install --python $(PYTHON) -e $(SAGE_SRC) --no-build-isolation

install_optional_baselines:
	@true

install_dep_best_effort: install_dep install_optional_baselines

bench:
	$(PYTHON) benchmark_sage_vs_flash.py

bench_causal:
	$(PYTHON) benchmark_sage_vs_flash.py --causal-only

clean:
	rm -rf *.mp4 && rm -rf *.json && rm -rf *.json.gz
