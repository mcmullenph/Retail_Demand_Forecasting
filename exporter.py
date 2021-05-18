import re

READ_FILE = "./Paul_McMullen_Report.html"
WRITE_FILE = "./Paul_McMullen_Report.html"

with open(READ_FILE, 'r', encoding = 'utf8') as html_file:
    content = html_file.read()

# Get rid off prompts and source code
# content = content.replace("div.input_area {","div.input_area {\n\tdisplay: none;")    
# content = content.replace(".prompt {",".prompt {\n\tdisplay: none;")
content = content.replace("jp-InputArea {","jp-InputArea {\n\tdisplay: none;")    
# content = content.replace("jp-OutputArea-prompt\">Out[4]: {","-prompt {\n\tdisplay: none;")
content = re.sub(r'>Out\[\d{1,5}]:<', '><', content)
# jp-OutputArea-prompt">Out[4]:

f = open(WRITE_FILE, 'w', encoding = 'utf8')
f.write(content)
f.close()