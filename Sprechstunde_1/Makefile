all: build/Notizen.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/Notizen.pdf: FORCE | build
	latexmk $(texoptions) Notizen.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc Notizen.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
