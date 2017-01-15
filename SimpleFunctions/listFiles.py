import os
import sys
import codecs
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics import renderPDF
from chardet.universaldetector import UniversalDetector


# sys.setdefaultencoding('utf8')

detector = UniversalDetector()
base_dir = '/Users/Tian/Downloads/txt2016'
BLOCKSIZE = 1024

def convertFileWithDetection():
    for file in os.listdir(base_dir):
        if file.endswith('.txt'):
            full_path = os.path.join(base_dir, file)
            file_name = os.path.splitext(file)[0]
            new_file_name = file_name + '01'
            get_encoding_type(full_path)
            print_100(full_path)
            with open(full_path, 'rb') as sourceFile:
                # with codecs.open(os.path.join(base_dir, new_file_name + '.txt'), 'w', 'utf-8') as targetFile:
                #     # while True:
                #     #     contents = sourceFile.read(BLOCKSIZE)
                #     #     if not contents:
                #     #         break
                #     #     targetFile.write(contents)
                #
                #     for line in sourceFile:
                #         targetFile.write(line.decode('utf-8').encode('utf-8'))
                pass

            # print(file_name + ' is done.')

def get_encoding_type(current_file):
    detector.reset()
    # for line in open(current_file):
    #     detector.feed(line)
    #     if detector.done: break
    try:
        with open(current_file) as f:
            for line in f:
                detector.feed(line)
                if detector.done: break
        detector.close()
        print(detector.result['encoding'])
        return detector.result['encoding']

    except Exception as e:
        print('e')
        return 'GB2312'

def print_100(file):
    d = Drawing(1000,1000)
    with open(file, 'rb') as f:

        for _ in range(3):
            print(f.readline().decode('utf-8'))
            sentence = str(f.readline().decode('utf-8'))
            s = String(10,10, sentence)
            d.add(s)

    renderPDF.drawToFile(d, 'test.pdf', 'A simple PDF file')




if __name__ == '__main__':
    # print(sys.getdefaultencoding())

    convertFileWithDetection()