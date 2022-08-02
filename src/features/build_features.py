from docx.api import Document
import os
import pandas as pd
from typing import List

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)

head = "Текст1. ГРП.docx"


def generate_ngrams(tokens: list, number: int, lemmatize: bool = False,
                    concatenate: bool = True) -> list:
    if lemmatize:
        tokens = [token.lemma for token in tokens]
        ngrams = zip(*[tokens[i:] for i in range(number)])
    else:
        tokens = [token.text for token in tokens]
        ngrams = zip(*[tokens[i:] for i in range(number)])
    if concatenate:
        return [" ".join(ngram) for ngram in ngrams]
    else:
        return list(ngrams)


def parse_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    doc.parse_syntax(syntax_parser)
    return doc


def get_doc_tables(document: Document) -> List[pd.DataFrame]:
    tables = list()
    for table in document.tables:

        data = []

        keys = None
        for i, row in enumerate(table.rows):
            text = (cell.text for cell in row.cells)

            if i == 0:
                keys = tuple(text)
                continue

            row_data = dict(zip(keys, text))
            data.append(row_data)

        df = pd.DataFrame(data)
        df.index = df.iloc[:, 0].values
        df.drop(df.columns[0], axis=1, inplace=True)

        tables.append(df)

    return tables


def get_doc_text(document: Document):

    return ' '.join([para.text for para in document.paragraphs])
