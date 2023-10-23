from typing import List
from code2drawio.tokenizer import Token, TokenType, tokenize
import unittest
from code2drawio.tokenizer import tokenize
from code2drawio.token_parser import Parser, Section, SecGroup


class TestParser(unittest.TestCase):
    # Sec -> SecTitle SecGroup
    def test_parse_sec(self):
        tokens = tokenize('==Hello==')
        parser = Parser(tokens)
        section: Section = parser.parse_sec()
        self.assertEqual(isinstance(section, Section), True)
        self.assertEqual(section.sec_title, 'Hello')

    # SecGroup -> TextGroup | SubSec | SubSec SecGroup | TextGroup SecGroup
    def test_parse_sec_group(self):
        # Test TextGroup
        tokens = tokenize('hello')
        parser = Parser(tokens)
        sec_group: SecGroup = parser.parse_sec_group()
        self.assertEqual(isinstance(sec_group, SecGroup), True)
        self.assertEqual(sec_group.item1.item1, 'hello')

        # Test SubSec
        tokens = tokenize('===subsec heading===\ntext')
        parser = Parser(tokens)
        sec_group: SecGroup = parser.parse_sec_group()
        self.assertEqual(isinstance(sec_group, SecGroup), True)
        self.assertEqual(sec_group.item1.subsec_title, 'subsec heading')

        # Test SubSec SecGroup
        tokens = tokenize(
            '===subsec heading===\ntext\n===subsec heading2===\ntext2')
        parser = Parser(tokens)
        sec_group: SecGroup = parser.parse_sec_group()
        print(sec_group)
        self.assertEqual(isinstance(sec_group, SecGroup), True)
        self.assertEqual(sec_group.item1.subsec_title, 'subsec heading')
        self.assertEqual(sec_group.item2.item1.subsec_title, 'subsec heading2')


class TestTokenizer(unittest.TestCase):
    def test_heading(self):
        # Equal Form
        code = '==Title1== \n apple'
        tokens: List[Token] = tokenize(code)
        self.assertEqual(tokens[0].type, TokenType.sec_title)
        self.assertEqual(tokens[0].content, 'Title1')
        # Hashtag Form
        code = '#Title2\n apple'
        tokens: List[Token] = tokenize(code)
        self.assertEqual(tokens[0].type, TokenType.sec_title)
        self.assertEqual(tokens[0].content, 'Title2')

    def test_hashtag(self):
        # Equal Form
        code = '===Title1=== \n apple'
        tokens: List[Token] = tokenize(code)
        self.assertEqual(tokens[0].type, TokenType.subsec_title)
        self.assertEqual(tokens[0].content, 'Title1')
        # Hashtag Form
        code = '##Title2\n apple'
        tokens: List[Token] = tokenize(code)
        self.assertEqual(tokens[0].type, TokenType.subsec_title)
        self.assertEqual(tokens[0].content, 'Title2')


# unittest.main(argv=[''], verbosity=2, exit=False)
if __name__ == '__main__':
    unittest.main()
