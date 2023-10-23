from typing import Any, NamedTuple, List, Optional
from enum import Enum
import re

class TokenType(Enum):
    end = -1
    sec_title = 0
    subsec_title = 1
    text_line = 2

class Token(NamedTuple):
    type: TokenType
    content: Any
    
def parse_tokens(line: str)->Token:
    subsec_title: Optional[re.Match] = re.search("===(.*)===", line)
    if subsec_title:
        return Token(TokenType.subsec_title, subsec_title.group(1))
    subsec_title: Optional[re.Match] = re.search("##(.*)", line)
    if subsec_title:
            return Token(TokenType.subsec_title, subsec_title.group(1))
    
    sec_title: Optional[re.Match] = re.search("==(.*)==", line)
    if sec_title: 
        return Token(TokenType.sec_title, sec_title.group(1))
    sec_title: Optional[re.Match] = re.search("#(.*)", line)
    if sec_title:
            return Token(TokenType.sec_title, sec_title.group(1))
    
    return Token(TokenType.text_line, line)

def tokenize(code: str) -> List[Token]:
    '''Tokenizes String Code into List of Tokens.
    ===Example===
    Input
    "==hello=="
    Output
    [Token(type=TokenType.sec_title, content="hello")]
    '''
    code_per_line: List[str] = code.strip().split('\n')
    # Remove all empty lines
    while '' in code_per_line:
        code_per_line.remove('')
    code_tokens: List[Token] = [parse_tokens(line) for line in code_per_line]
    return code_tokens