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

# build docs for github pages in docs/ from /docsrc
# 1. (optional) remove stale spiir.rst files from previous builds 
# 2. clear docs/ directory
# 3. generate .rst files from spiir package structure
# 4. build html files from .rst
# 5. copy built html files to docs/
# 6. add empty .nojekyll file so github pages does not use jekyll to render
# docs:
# 	@rm -f ./source/spiir*.rst
# 	@rm -rf ../docs
# 	@sphinx-apidoc -fME -o ./source ../src/spiir
# 	@make html
# 	@cp -a build/html/. ../docs
# 	@touch ../docs/.nojekyll

# https://www.sphinx-doc.org/en/master/tutorial/deploying.html
# build docs for github pages from main branch docs/ and push to gh-pages
# 1. (optional) remove stale spiir.rst files from previous builds 
# 2. generate .rst files from spiir package structure
# 3. create html from .rst
#    a .nojeykll file is created via actions-gh-pages with enable_jekyll: False
docs:
	@rm -r ./source/spiir*.rst
	@sphinx-apidoc -fE -o ./source ../src/spiir
	@make html