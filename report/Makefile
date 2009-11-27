report.pdf:	report.tex report.bib deepnet.pdf auto-encoder.pdf likeicml.sty betterblender.pdf gated.pdf
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

%.pdf:	%.svg
	inkscape --without-gui --export-area-drawing --export-pdf $@~ $<
	mv $@~ $@

