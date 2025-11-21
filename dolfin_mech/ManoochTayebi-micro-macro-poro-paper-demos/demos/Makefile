#!/bin/bash

.PHONY: convert_notebooks clean_notebooks clean_output_files

NOTEBOOKS := $(wildcard *.ipynb)
LATEX_FILES := $(NOTEBOOKS:.ipynb=.tex)

convert_notebooks: $(LATEX_FILES)

%.tex: %.ipynb
	jupyter nbconvert --to latex $<

clean_notebooks:
	for file in *.tex; do sed -i '' '1,/^\\section/d' $${file}; done
	for file in *.tex; do sed -i '' '/\\end{document}/,$$d' $${file}; done
	for file in *.tex; do sed -i '' 's/section{/section*{/g' $${file}; done

OUTPUT_FORMATS := aux fdb_latexmk fls h5 log msh out pdf sta synctex.gz tex tex.bu vtk xdmf xml
OUTPUT_FILES := $(foreach format,$(OUTPUT_FORMATS),$(wildcard *.$(format)))
DAT_FILES := $(filter-out Fig5-voronoi-seeds.dat, $(wildcard *.dat))

clean_output_files:
	rm -f $(OUTPUT_FILES) $(DAT_FILES)
