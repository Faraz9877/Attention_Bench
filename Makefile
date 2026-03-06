IMG=baseten/attn:base
CONTAINER_NAME=attn_dev
PYTHON_VERSION=3.11
VENV=.venv
PYTHON=$(VENV)/bin/python
DIST=dist_h100
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
	uv pip install --python $(PYTHON) $(FLASH3_WHL)
	TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" \
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
