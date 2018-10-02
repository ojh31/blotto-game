clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*.py~' -exec rm --force {} +
	find . -name '*.un~' -exec rm --force {} +
	find . -name '*.sh~' -exec rm --force {} +

run:
	python blotto.py 10000
