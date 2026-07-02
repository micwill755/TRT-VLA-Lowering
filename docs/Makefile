# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= $(if $(wildcard .venv/bin/sphinx-build),.venv/bin/sphinx-build,sphinx-build)
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html clean setup

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"
