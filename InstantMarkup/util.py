import os


def lines(file):
    for line in file: yield line
    yield os.linesep


def blocks(file):
    block = []
    for line in lines(file):
        if line.strip():
            block.append(line)
        elif block:
            yield ''.join(block).strip()
            block = []

