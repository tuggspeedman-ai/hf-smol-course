.PHONY: setup setup-notebooks fmt lint test test-unit test-all smoke clone-course hf-login help

help:
	@echo "Common targets:"
	@echo "  setup            uv sync (base deps)"
	@echo "  setup-notebooks  uv sync + jupyter extras"
	@echo "  fmt              ruff format"
	@echo "  lint             ruff check"
	@echo "  test             fast unit tests (skips slow/integration)"
	@echo "  test-all         all tests incl. slow + integration"
	@echo "  smoke            load a tiny SmolLM2 model to verify the env works"
	@echo "  clone-course     clone the HF smol-course repo into course-materials/"
	@echo "  hf-login         interactive hf login (uses HF_TOKEN if set)"

setup:
	uv sync

setup-notebooks:
	uv sync --extra notebooks

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

test: test-unit

test-unit:
	uv run pytest -v -m "not slow and not integration"

test-all:
	uv run pytest -v

smoke:
	uv run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
	m='HuggingFaceTB/SmolLM2-135M'; \
	tok=AutoTokenizer.from_pretrained(m); \
	mdl=AutoModelForCausalLM.from_pretrained(m); \
	print('loaded', m, 'params=', sum(p.numel() for p in mdl.parameters()))"

clone-course:
	@mkdir -p course-materials
	@if [ -d course-materials/smol-course ]; then \
		echo "course-materials/smol-course already exists — skipping"; \
	else \
		git clone https://github.com/huggingface/smol-course.git course-materials/smol-course; \
	fi

hf-login:
	@. .env 2>/dev/null; \
	if [ -n "$$HF_TOKEN" ]; then \
		uv run hf auth login --token $$HF_TOKEN --add-to-git-credential; \
	else \
		uv run hf auth login; \
	fi
