all: output.txt

output.txt: linearfit.py
	python3 $< > $@
