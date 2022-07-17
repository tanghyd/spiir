# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# build docs for github pages
# 1. generate .rst files from spiir package structure
# 2. clean old build artifacts (safe) and create html from .rst
# 3. clear docs/ directory
# 4. copy built html files to docs/
# 5. add empty .nojekyll file so github pages does not use jekyll to render
docs:
	@sphinx-apidoc -fME -o ./source ../src/spiir
	@make clean && make html
	@rm -r ../docs
	@cp -a build/html/. ../docs
	@touch ../docs/.nojekyll