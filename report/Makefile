MAIN = report

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib
	pdflatex $(MAIN).tex
	bibtex $(MAIN) || true
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out $(MAIN).toc
