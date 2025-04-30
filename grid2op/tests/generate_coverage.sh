#/bin/bash
coverage run --source=.. -m unittest discover
coverage report -m -i
coverage html -i

