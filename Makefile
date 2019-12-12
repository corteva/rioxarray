.PHONY: clean clean-test clean-pyc clean-build docs help test
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 rioxarray test --max-line-length 88
	black --target-version py36 --check .

check:
	###### FLAKE8 #####
	# No unused imports, no undefined vars,
	flake8 --ignore=E731,W503,W504 --exclude --max-complexity 10 --max-line-length 88 rioxarray/
	flake8 --max-line-length 88 tests/unit/ tests/functional/ tests/integration
	black --target-version py36 --check .

pylint:
	###### PYLINT ######
	pylint --rcfile .pylintrc rioxarray
	# Run our custom linter on test code.
	pylint --load-plugins tests.linter --disable=I,E,W,R,C,F --enable C9999,C9998 tests/

test: ## run tests quickly with the default Python
	py.test

docs: ## generate Sphinx HTML documentation, including API docs
	# rm -f sphinx/rioxarray*.rst
	# rm -f sphinx/modules.rst
	# sphinx-apidoc -o sphinx/ rioxarray
	$(MAKE) -C sphinx clean
	$(MAKE) -C sphinx html
	$(BROWSER) docs/html/index.html

release: dist ## package and upload a release
	twine upload dist/*

report: install-dev coverage ## clean, install development version, run all tests, produce coverage report

install: clean ## install the package to the active Python's site-packages
	python setup.py install

install-dev: clean ## install development version to active Python's site-packages
	pip install -U -r requirements.txt -r requirements_dev.txt
	pip install -e .[all]
