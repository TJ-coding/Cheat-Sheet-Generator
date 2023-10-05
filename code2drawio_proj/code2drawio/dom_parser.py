from typing import List, Optional
from dataclasses import dataclass

from token_parser import SubSec, TextGroup, SecGroup, Section, Document

@dataclass
class TextGroupDOM():
    text_lines: List[str]
    
@dataclass
class SubSecDOM():
    subsec_title: str
    text_group: Optional[TextGroupDOM]

@dataclass
class SectionDOM():
    section_title: str
    children: List[TextGroupDOM|SubSecDOM]
    
@dataclass
class DocumentDOM():
    children: List[SectionDOM|TextGroupDOM]
    
from types import NoneType

def parse_subsec_dom(subsec: SubSec)->SubSecDOM:
    # Takes Subsection and Returns SubSectionDOM
    if subsec.text_group is None:
        return SubSecDOM(subsec.subsec_title, None)
    return SubSecDOM(subsec.subsec_title, parse_text_group_dom(subsec.text_group))

def parse_text_group_dom(text_group: TextGroup)->TextGroupDOM:
    # Takes a TextGroup and Returns TextGroupDOM
    text_group.item1
    if text_group.item2 == None:
        return TextGroupDOM([text_group.item1])
    text_group_dom: TextGroupDOM = parse_text_group_dom(text_group.item2)
    return TextGroupDOM([text_group.item1]+text_group_dom.text_lines)

def parse_sec_group(sec_group: SecGroup)-> List[SubSecDOM|TextGroupDOM]:
    # Takes Section Group Instance Returns List[SubSectionDOM|TextGroupDOM]
    first_element: SubSecDOM|TextGroupDOM
    if isinstance(sec_group.item1, SubSec):
        first_element = parse_subsec_dom(sec_group.item1)
    elif isinstance(sec_group.item1, TextGroup):
        first_element = parse_text_group_dom(sec_group.item1)
    else:
        raise ValueError('First Item of SecGroup must either be a SubSec or TextGroup. Was '+str(type(sec_group.item1)))        
    if isinstance(sec_group.item2, NoneType):
        # Did not contain second parameter
        return [first_element]
    return [first_element] + parse_sec_group(sec_group.item2)

def parse_section(section: Section)->SectionDOM:
    # Takes Section Instance as an Input and Returns a SectionDOM instance
    if section.sec_group is None:
        return SectionDOM(section.sec_title, [])
    return SectionDOM(section.sec_title, parse_sec_group(section.sec_group))

def parse_document(document: Document)->DocumentDOM:
    # Takes Document Instance as an Input and Returns a DocumentDOM instance
    item1: SectionDOM|TextGroupDOM
    if isinstance(document.item1, Section):
        item1 = parse_section(document.item1)
    elif isinstance(document.item1, TextGroup):
        item1 = parse_text_group_dom(document.item1)
    else:
        raise ValueError('The item1 of Document object instance must either be Section or TextGroup')
    
    if document.item2 is None:
        return DocumentDOM([item1])
    item2:List[SectionDOM|TextGroupDOM] = parse_document(document.item2).children
    return DocumentDOM([item1]+item2)

'''
Traveres the parsed AbstractSyntaxTree and creates a DOM representation.

==Example==

from .tokenizer import tokenize, Token
from .token_parser import Parser
tokens: List[Token] = tokenize("==hello==")
parser = Parser(tokens)
parsed_section: Section = parser.parse_sec()
section_dom: SectionDOM = parse_section(parsed_section)
'''