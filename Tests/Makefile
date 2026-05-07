test:
	set -e; for i in test_*.py; do set -e; python $$i; done

coverage:
	coverage erase
	set -e; for i in test_*.py; do set -e; coverage run -a --source=../dolfin_mech $$i; done
	coverage report -m > coverage.txt
