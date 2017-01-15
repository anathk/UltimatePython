from openpyxl import Workbook
from openpyxl.compat import range
from openpyxl.utils import get_column_letter
import datetime

wb = Workbook()

#txt_file = open('/Users/Tian/Desktop/Dec21st_2016_Fe_50C_Polarization_1.txt', 'r')
dest_filename = 'empty_book.xlsx'

ws1 = wb.active
ws1.title = 'range names'

linenum = 0
# start_read = datetime.datetime.now()
# with open('/Users/Tian/Desktop/Dec21st_2016_Fe_50C_Polarization_1.txt', 'r') as txt_file:
#     print(sum(1 for _ in txt_file))
# end_read = datetime.datetime.now()
# print("Read duration: %s. " % str(end_read-start_read))



start_read = datetime.datetime.now()
with open('/Users/Tian/Desktop/Dec21st_2016_Fe_50C_Polarization_1.txt', 'r') as txt_file:
    with open('test.csv', 'w') as csv_file:

        for line in txt_file:
            #ws1.append(line.split())
            csv_file.write(",".join(line.split()) + "\n")
            linenum += 1
            if linenum % 10000 == 0:
                #print(linenum % 10000)
                print("Line %d appended.\n" % linenum)
        end_read = datetime.datetime.now()
        print("Totle line: %d." % linenum)
        print("Read duration: %s. " % str(end_read-start_read))






# print("Start to save excel file...")
# start_save = datetime.datetime.now()
# wb.save(filename=dest_filename)
# end_save = datetime.datetime.now()
# print("Save duration: %s. " % str(end_save-start_save))
print('Done.')