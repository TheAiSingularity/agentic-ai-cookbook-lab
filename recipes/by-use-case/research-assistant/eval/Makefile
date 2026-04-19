.PHONY: eval eval-simpleqa eval-browsecomp ablate ablate-simpleqa ablate-browsecomp pareto clean help

PY     ?= python3.12
VENV   := ../beginner/.venv
PYBIN  := $(VENV)/bin/python
REPO_ROOT := $(abspath ../../../..)

help:
	@echo "Targets:"
	@echo "  eval          — run seed dataset through the default (beginner) config + print scores"
	@echo "  eval-simpleqa      — beginner config over simpleqa_seed.jsonl"
	@echo "  eval-browsecomp    — beginner config over browsecomp_plus_seed.jsonl"
	@echo "  ablate        — run full ablation matrix over BOTH seed datasets (results.jsonl)"
	@echo "  ablate-simpleqa    — full matrix, SimpleQA seeds only"
	@echo "  ablate-browsecomp  — full matrix, BrowseComp-Plus seeds only"
	@echo "  pareto        — score results.jsonl + emit aggregate table + pareto.png"
	@echo "  clean         — remove results.jsonl and pareto.png"
	@echo ""
	@echo "Requires: EXA/GOOGLE not needed; OPENAI_API_KEY for the judge;"
	@echo "          full stack (OPENAI_BASE_URL + SEARXNG_URL + models) for the agent."

eval: eval-simpleqa eval-browsecomp

eval-simpleqa:
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) scorer.py datasets/simpleqa_seed.jsonl

eval-browsecomp:
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) scorer.py datasets/browsecomp_plus_seed.jsonl

ablate: ablate-simpleqa ablate-browsecomp

ablate-simpleqa:
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) ablation.py \
	  --dataset datasets/simpleqa_seed.jsonl \
	  --configs $(CONFIGS)

ablate-browsecomp:
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) ablation.py \
	  --dataset datasets/browsecomp_plus_seed.jsonl \
	  --configs $(CONFIGS)

CONFIGS ?= A1,A2,B1,B3

pareto:
	@test -f results.jsonl || (echo "No results.jsonl — run 'make ablate' first." && exit 1)
	PYTHONPATH=$(REPO_ROOT) $(PYBIN) pareto.py

clean:
	rm -f results.jsonl pareto.png
