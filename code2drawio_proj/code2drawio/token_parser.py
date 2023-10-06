from multiprocessing import Value
from typing import Optional, List
from dataclasses import dataclass

from .tokenizer import Token, TokenType

@dataclass
class TextGroup:
    item1: str
    item2: Optional['TextGroup']

@dataclass
class SubSec:
    subsec_title: str
    text_group: Optional[TextGroup]
    
@dataclass
class SecGroup:
    item1: TextGroup | SubSec
    item2: Optional['SecGroup']

@dataclass
class Section:
    sec_title: str
    sec_group: Optional[SecGroup]

@dataclass
class Document:
    item1: TextGroup| Section
    item2: Optional['Document']
    
class Parser():
    ''' Converts List of Tokens to Abstract Parse Tree.
    ==Example==
    from .tokenizer import tokenize, Token
    tokens: List[Token] = tokenize("==hello==")
    parser = Parser(tokens)
    parsed_section: Section = parser.parse_sec()
    '''
    
    def __init__(self, _tokens: List[Token]):
        self.tokens: List[Token] = _tokens
        self.cursor_index: int = 0
        self.next_token: Token = self.tokens[0]
        
    def scan_next_token(self):
        if self.cursor_index > len(self.tokens):
            raise ValueError('No more tokens to scan')
        # Updates the next token
        self.cursor_index += 1 
        if self.cursor_index >= len(self.tokens):
            self.next_token = Token(TokenType.end, '')
            return
        self.next_token = self.tokens[self.cursor_index]
    
    # Doc-> Sec|TextGroup|Sec Doc| TextGroup Doc
    def parse_doc(self)->Optional[Document]:
        item1: Optional[TextGroup | Section] = self.parse_sec()
        if item1 is None:   # Could not parse as Section
            item1 = self.parse_text_group()
        if item1 is None:
            # The first item of a Document must either be TextGroup or Section.
            return None
        item2: Optional[Document] = self.parse_doc()
        return Document(item1, item2)
    
    # Sec -> SecTitle SecGroup
    def parse_sec(self)->Optional[Section]:
        if self.next_token.type is not TokenType.sec_title:
            # First token of section must be section title
            return None
        section_title: str = self.next_token.content
        self.scan_next_token()
        sec_group: Optional[SecGroup] = self.parse_sec_group()
        return Section(section_title, sec_group)
    
    # SecGroup -> TextGroup | SubSec | SubSec SecGroup | TextGroup SecGroup
    def parse_sec_group(self)->Optional[SecGroup]:
        '''If Next Tokens are parsable as SecGroup, returns SecGroup. If not, returns None.'''
        item1: Optional[TextGroup|SubSec] = self.parse_text_group()
        if item1 is None:
            item1 = self.parse_subsec()
        if item1 is None:
            print('first token was not text group or subsec')
            return None
        item2 = self.parse_sec_group()
        return SecGroup(item1, item2)
        
    # SubSec -> SubSecTitle | SubSecTitle TextGroup
    def parse_subsec(self)->Optional[SubSec]:
        if self.next_token.type is not TokenType.subsec_title:
            return None
        subsec_title: str = self.next_token.content
        self.scan_next_token()
        text_group: Optional[TextGroup] = self.parse_text_group()
        return SubSec(subsec_title, text_group)
    
    # TextGroup -> TextLine | TextLine TextGroup
    def parse_text_group(self)->Optional[TextGroup]:
        if self.next_token.type is not TokenType.text_line:
            return None
        text_line: str = self.next_token.content
        self.scan_next_token()
        item2: Optional[TextGroup] = self.parse_text_group()
        return TextGroup(text_line, item2)
    
'''Context Free Grammar
Doc-> Sec|TextGroup|Sec Doc| TextGroup Doc
Sec -> SecTitle SecGroup
SecGroup -> TextGroup | SubSec | SubSec SecGroup | TextGroup SecGroup
SubSec -> SubSecTitle | SubSecTitle TextGroup
TextGroup -> TextLine | TextLine TextGroup
'''