"""Export a .docx to .pdf via Word COM for visual verification."""
import sys
from pathlib import Path

import win32com.client

src = Path(sys.argv[1]).resolve()
pdf = src.with_suffix(".pdf")
word = win32com.client.Dispatch("Word.Application")
word.Visible = False
d = word.Documents.Open(str(src))
d.SaveAs2(str(pdf), FileFormat=17)  # wdFormatPDF
d.Close(False)
word.Quit()
print(pdf)
