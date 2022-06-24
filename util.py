import os
import nltk
from nltk.grammar import DependencyGrammar
from nltk.parse import DependencyGraph, ProjectiveDependencyParser, NonprojectiveDependencyParser
from nltk.parse.corenlp import CoreNLPDependencyParser

class sentence_processor():
    def __init__(self,text_sentence):
        self.sentence = text_sentence
    def dependency_tree_build(self):
        print(self.sentence)
        parser = CoreNLPDependencyParser(url='http://localhost:9000')
        return parser.raw_parse(self.sentence)

if __name__ == "__main__":
    test = sentence_processor("The quick brown fox jumps over the lazy dog.")
    result, = test.dependency_tree_build()
    # result = list(result)
    # result.pretty_print()
    for head, rel, dep in result.triples():
        print(head,rel,dep)
    # print(result.to_conll(4))
    # print()