from typing import List

from tokenizer import tokenize, Token
from token_parser import Parser, Section, Document
from dom_parser import  parse_document, DocumentDOM
from drawio import draw_diagram

from N2G import drawio_diagram

def render_code(code: str, drawio_file_name: str, drawio_filde_folder: str):
    tokens: List[Token] = tokenize(code)
    parser = Parser(tokens)
    document: Document = parser.parse_doc()
    document_dom: DocumentDOM = parse_document(document)
    diagram = drawio_diagram()
    diagram.add_diagram('Page-1')
    draw_diagram(document_dom, diagram)
    diagram.dump_file(drawio_file_name, folder=drawio_filde_folder)

if __name__ == '__main__':
    # code = '==hello==\nhow are you\n===sub section===\nbye bye'
    with open('/home/sp/Desktop/Cheat Sheet Generator/Driving School 技 Cheat Sheet.txt', 'r') as handle:
        code: str = handle.read()
    code = '''
==Sec 1==
Some Text
===Sec 1.1===
Hello
===Sec1.2===
Bye
==Sec 2==
Hi'''
    render_code(code, 'demo.drawio', '/home/sp/Desktop/Cheat Sheet Generator')