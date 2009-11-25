report.pdf:	report.tex report.bib deepnet.pdf auto-encoder.pdf likeicml.sty
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

%.pdf:	%.svg
	inkscape --without-gui --export-pdf $@~ $<
	mv $@~ $@

