all: build/Vortrag.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/Vortrag.pdf: FORCE | build
	latexmk $(texoptions) Vortrag.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc Vortrag.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
