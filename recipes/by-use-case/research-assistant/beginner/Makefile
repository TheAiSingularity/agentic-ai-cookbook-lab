.PHONY: run smoke test install clean help

PY       ?= python3.12
VENV     ?= .venv
PYBIN    := $(VENV)/bin/python
PIPBIN   := $(VENV)/bin/pip
REPO_ROOT := $(abspath ../../../..)

help:
	@echo "Targets:"
	@echo "  install  — create .venv and install requirements"
	@echo "  run      — run the recipe (requires OPENAI_API_KEY)"
	@echo "  smoke    — live end-to-end check with a canned question"
	@echo "  test     — mocked unit tests (no API key needed)"
	@echo "  clean    — remove .venv"

$(PYBIN):
	$(PY) -m venv $(VENV)
	$(PIPBIN) install -q --upgrade pip
	$(PIPBIN) install -q -r requirements.txt

install: $(PYBIN)

run: $(PYBIN)
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) main.py "$(Q)"

smoke: $(PYBIN)
	@test -n "$$OPENAI_API_KEY" || (echo "OPENAI_API_KEY not set" && exit 1)
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) main.py "What is Anthropic's contextual retrieval and why does it reduce retrieval failures?"

test: $(PYBIN)
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) -m pytest test_main.py -v

clean:
	rm -rf $(VENV) .pytest_cache __pycache__
