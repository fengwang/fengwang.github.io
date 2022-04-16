all:
	xelatex -shell-escape main
	bibtex main
	xelatex -shell-escape main
	pdf2htmlEX main.pdf ../../index.html
