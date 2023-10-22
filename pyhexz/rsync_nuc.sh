rsync -v -c -rtp --exclude=__pycache__ --exclude=build/ --exclude=*.so --exclude=*.c --exclude=.pytest* --exclude=.ipynb* ./ dw@nuc:/home/dw/git/github.com/dnswlt/hexz/pyhexz/
