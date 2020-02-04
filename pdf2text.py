# python 3.7
# -*- coding: utf-8 -*-
# @Time    : 2020-02-03 17:46
# @Author  : Xueli
# @File    : pdf2text.py
# @Software: PyCharm

# required packages:pdfminer.six
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter, TextConverter, XMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os
import os.path

def convert_pdf_to_txt(file):
    # input: a pdf file
    pdfFilePath = file
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(file, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()

    # get pdf file name and then create a new text file to store the text content of a paper
    pdfFileName = os.path.basename(pdfFilePath)
    portion = os.path.splitext(pdfFileName)
    if portion[1] == ".pdf":
        txtFileName = portion[0] + ".txt"
    # write text into txtFileName and save to current directory()
    f = open(txtFileName, "w+")
    f.write(text)
    f.close()
    return txtFileName

