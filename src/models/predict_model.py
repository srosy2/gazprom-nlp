# This is a sample Python script.
import json
import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from typing import List, Tuple

from src.features.build_features import parse_text
from ..data.make_dataset import load_data
from ..features.build_features import generate_ngrams
from natasha import Doc
from natasha.doc import DocToken
from tqdm import tqdm
import tensorflow as tf

import tensorflow_text
import tensorflow_hub as hub
import torch.nn.functional as F
from torch import tensor
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from ontology_base import CreateServer
import datefinder

data_path = 'data'

# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
model = hub.load(module_url)

# excel_file = 'Кластеры данных для эталонного сравнения (Бенчмаркинг ТЭП + СМБ + ОЭДК).xlsx'
excel_file = 'файл по категоризации.xlsx'
metrics_column = 'Метрики'
type_metrics_column = 'Тип данных'

nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli').eval()
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

prefix: str = r"<http://www.semanticweb.org/кристина/ontologies/2021/3/untitled-ontology-8#>"


def prepare_triple(value: str):
    return value.split('#')[-1][:-1].replace('_', ' ')


def prepare_sparql_response(input: List[str]):
    return [prepare_triple(value) for value in input]


def embed(input):
    return model(input)


def load_file(file_name: str) -> str:
    with open(file_name, encoding="utf-8") as file:
        text = file.read()
    return text


def check_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def closeness_ngram(metric_ngram: list, ngram: list):
    return True if metric_ngram == ngram else False


def closeness_text_metric(metrics_ngram: list, ngrams: list):
    equal_ngram = list(set([ngram for metric_ngram in metrics_ngram
                            for ngram in ngrams if closeness_ngram(metric_ngram, ngram)]))
    return equal_ngram if len(equal_ngram) / len(metrics_ngram) >= 0.8 \
        else False


def get_equal_ngram(text: Doc, metrics):
    detect_metrics = dict()
    number_ngram = 2

    ngrams = generate_ngrams(text.tokens, number_ngram, lemmatize=True, concatenate=False)

    for metric in metrics:

        ngram_metric = generate_ngrams(parse_text(metric).tokens, number_ngram, lemmatize=True, concatenate=False)

        if ngram_metric:
            equal_texts_metric = closeness_text_metric(ngram_metric, ngrams)

        else:
            metric_lemma = parse_text(metric).tokens[0].lemma
            equal_texts_metric = {metric: metric_lemma} if metric_lemma in \
                                                           [token.lemma for token in text.tokens] else False

        if equal_texts_metric:
            equal_texts_metrics = detect_metrics.get(metric, list())
            equal_texts_metrics.append(equal_texts_metric)
            detect_metrics.update({metric: equal_texts_metrics})

    return detect_metrics


def equal_ngrams(text: Doc, server: CreateServer) -> Tuple[dict, dict]:
    # find by values and metrics categorical values
    detect_metrics = dict()
    detect_metrics_value = dict()
    metrics = prepare_sparql_response(server.find_object(f"prefix : {prefix}"
                                                         r"SELECT ?s { ?s ?p :Метрики . }")['s'].values)

    categorical_metrics = prepare_sparql_response(sparql_request(server, f"prefix : {prefix}"
                                                                         "SELECT * "
                                                                         "WHERE { "
                                                                         "?s ?p :str ."
                                                                         "?s :Может_принимать_значения ?o ."
                                                                         "}")['s'].values)

    categorical_values = prepare_sparql_response(sparql_request(server, f"prefix : {prefix}"
                                                                        "SELECT * "
                                                                        "WHERE { "
                                                                        "?s ?p :str ."
                                                                        "?s :Может_принимать_значения ?o ."
                                                                        "}")['o'].values)

    recognise_values = sparql_request(server, f"prefix : {prefix}"
                                              "SELECT * "
                                              "WHERE { ?s :Опзнать_по_значению ?o .}")['s'].values

    recognise_values = prepare_sparql_response(sparql_request(server, f"prefix : {prefix}"
                                                                      "SELECT * "
                                                                      "WHERE { "
                                                                      "?s :Может_принимать_значения ?o."
                                                                      f" FILTER (?s IN ({', '.join(recognise_values)}))"
                                                                      "}")['o'].values)

    equal_values = list(get_equal_ngram(text, categorical_values).keys())
    detect_metrics.update(get_equal_ngram(text, metrics))

    for value in equal_values:
        if value in recognise_values:
            metric = prepare_sparql_response(sparql_request(server, f"prefix : {prefix}"
                                                                    "SELECT * "
                                                                    "WHERE { "
                                                                    f" ?s :Может_принимать_значения :{'_'.join(value.split(' '))} ."
                                                                    "}")['s'].values)[0]
            metrics_value = detect_metrics_value.get(metric, list())
            metrics_value.append(value)
            detect_metrics_value.update({metric: metrics_value})
            equal_values.pop(equal_values.index(value))

    for metric in detect_metrics.keys():
        for value in equal_values:
            if sparql_ask(server, f"prefix : {prefix}"
                                  "ASK WHERE {"
                                  f":{'_'.join(metric.split(' '))} ?p :{'_'.join(value.split(' '))} ."
                                  "}") and metric in categorical_metrics:
                metrics_value = detect_metrics_value.get(metric, list())
                metrics_value.append(value)
                detect_metrics_value.update({metric: metrics_value})

    return detect_metrics, detect_metrics_value


def syntax_chunk(token: DocToken, sentence: Doc, iteration: int) -> list:
    if token.rel == 'root' or iteration == 10:
        return [token]

    iteration += 1
    head_id_sent = int(token.head_id.split('_')[0]) - 1
    head_id_token = int(token.head_id.split('_')[1]) - 1
    chunk = syntax_chunk(sentence.sents[head_id_sent].tokens[head_id_token], sentence, iteration)
    chunk.append(token)
    return chunk


def complete_syntax_chunk(syntax: list, sentence: Doc, limit: int = 10):
    complete_syntax = syntax.copy()
    tokens_id = [token.id for token in syntax]
    [complete_syntax.append(token) for token in sentence.tokens
     if (token.head_id in tokens_id and len(complete_syntax) < limit)]

    return complete_syntax


def embed_similarity(metric: str, text: str) -> float:
    metric_embeding = embed([metric])
    text_embeding = embed([text])
    cos_similarity = cos_sim(metric_embeding, text_embeding)
    return cos_similarity


def text_similarity(metric, syntax) -> float:
    syntax_lemma = list(set([_.lemma for _ in syntax]))
    parse_metric = parse_text(metric)
    parse_metric = [_.lemma for _ in parse_metric.tokens]
    similarity = sum([True for _ in parse_metric if _ in syntax_lemma]) / len(parse_metric)
    return similarity


def sentence_prepare(syntax):
    syntax_tokens = []
    syntax_numbers = [_.id for _ in syntax]
    syntax = [token[1] for token in sorted(zip(syntax_numbers, syntax), key=lambda x: int(x[0].split('_')[1]))]
    [syntax_tokens.append(token) for token in syntax if token not in syntax_tokens]
    return syntax_tokens


def sentence_prepare_tokens(syntax):
    syntax_lemma = [_.lemma for _ in syntax]
    return syntax_lemma


def get_lemma_text(text: str) -> list:
    text = parse_text(text).tokens
    return [token.lemma for token in text]


def get_table_metrics(tables: List[pd.DataFrame], server: CreateServer):
    table_metrics = dict()
    metrics = prepare_sparql_response(server.find_object(f"prefix : {prefix}"
                                                         r"SELECT ?s { ?s ?p :Метрики . }")['s'].values)
    detect_metrics = list()
    for table in tables:

        indexes = table.index

        for index in indexes:
            find_metrics = list()

            for metric in metrics:
                if closeness_text_metric(get_lemma_text(metric), get_lemma_text(index)):
                    find_metrics.append(metric)

            if find_metrics:
                detect_metrics.append((index, sorted(find_metrics, key=lambda x: len(parse_text(x).tokens),
                                                     reverse=True)[0]))

        for index, metric in detect_metrics:
            value = table_metrics.get(metric, list())
            value.extend(table.loc[index].values)
            table_metrics.update({metric: value})

    return table_metrics


def sparql_ask(server: CreateServer, request: str) -> bool:
    return server.check_object(request)


def sparql_request(server: CreateServer, request: str) -> pd.DataFrame:
    return server.find_object(request)


def near_chunk(token, text: Doc, radius: int = 2) -> list:
    id_sent = int(token.id.split('_')[0]) - 1
    id_token = int(token.id.split('_')[1]) - 1
    start = id_token - radius if id_token > radius else 0
    end = id_token + radius
    return text.sents[id_sent].tokens[start: end]


def intersection(value_1: tuple, value_2: tuple) -> bool:
    return len(set(range(*value_1)) - set(range(*value_2))) != len(set(range(*value_1)))


def get_date_time(text: Doc) -> list:
    adj_tokens = [token for token in text.tokens if token.pos == 'ADJ']
    indexes = [index[1] for index in datefinder.find_dates(text.text, index=True)]
    data_tokens = [adj_token for adj_token in adj_tokens for index in indexes
                   if intersection((adj_token.start, adj_token.stop), index)]
    return data_tokens


def get_syntax_detect_metrics(text: Doc, detect_tokens: list, metrics: list, find_metrics: list):
    detect_metrics = dict()
    chunks = {token.lemma: syntax_chunk(token, text, 0) for token in detect_tokens}
    chunks = {token: complete_syntax_chunk(chunks.get(token)[::-1], text, 10) + near_chunk(chunks.get(token)[-1], text)
              for token in chunks.keys()}
    chunks: dict = {token: sentence_prepare(chunks.get(token)) for token in chunks.keys()}
    chunks_text: dict = {token: ' '.join([_.text for _ in chunks.get(token)]) for token in chunks.keys()}
    # chunks_lemma = {token: sentence_prepare_tokens(chunks.get(token)) for token in chunks.keys()}
    if chunks_text:
        embed_chunk_text = embed(list(chunks_text.values()))
        embed_chunk_metrics = embed(list(metrics))
        embed_text = {list(chunks_text.keys())[vec]: embed_chunk_text[vec] for vec in range(embed_chunk_text.shape[0])}
        embed_metrics = {metrics[vec]: embed_chunk_metrics[vec] for vec in range(embed_chunk_metrics.shape[0])}
    else:
        embed_text, embed_metrics = None, None

    for token in chunks.keys():

        possible_metrics = dict()

        for metric in metrics:

            if embed_metrics is not None and embed_text is not None:

                similarity = text_similarity(metric, chunks.get(token))

                cos_similarity = cos_sim(np.array(embed_metrics.get(metric)).reshape(-1, 512),
                                         np.array(embed_text.get(token)).reshape(-1, 512))

                if cos_similarity >= 0.8 or similarity >= 0.3:
                    add_value = 1 if metric in find_metrics else 0
                    possible_metrics.update({metric: cos_similarity + similarity * 2 + add_value})

        if possible_metrics:
            possible_metrics = sorted(possible_metrics.items(), key=lambda item: item[1], reverse=True)
            tokens = detect_metrics.get(possible_metrics[0][0], list())
            tokens.append(token)
            detect_metrics.update({possible_metrics[0][0]: tokens})

    return detect_metrics


def syntax_detection(text: Doc, server: CreateServer, find_metrics: list) -> dict:
    detect_metrics = dict()

    number_metrics = prepare_sparql_response(sparql_request(server, f"prefix : {prefix}"
                                                                    "SELECT * "
                                                                    "WHERE { "
                                                                    "?s ?p ?o."
                                                                    " FILTER (?o IN (:int, :float ) )"
                                                                    "}")['s'].values)

    data_metrics = prepare_sparql_response(sparql_request(server, f"prefix : {prefix}"
                                                                  "SELECT * "
                                                                  "WHERE { "
                                                                  "?s ?p :data."
                                                                  "}")['s'].values)

    num_tokens = [token for token in text.tokens if token.pos == 'NUM']
    # data_tokens = [token for token in text.tokens if token.pos == 'ADJ']
    data_tokens = get_date_time(text)

    detect_metrics.update(
        get_syntax_detect_metrics(text, num_tokens, number_metrics, find_metrics))
    detect_metrics.update(
        get_syntax_detect_metrics(text, data_tokens, data_metrics, find_metrics))

    return detect_metrics


def closeness_objects(find_metrics: list, server: CreateServer) -> list:
    objs_metrics: pd.DataFrame = server.find_object(f"prefix : {prefix}"
                                                    r"SELECT * WHERE { ?s :Имеет_метрику ?o . }"
                                                    )[['s', 'o']]

    objs_metrics = pd.DataFrame(objs_metrics.groupby('s')['o'].apply(list))
    objs_metrics.index = prepare_sparql_response(list(objs_metrics.index))
    objs_metrics['o'] = objs_metrics['o'].apply(lambda metrics: prepare_sparql_response(metrics))
    objs_metrics['o'] = objs_metrics['o'].apply(lambda metrics: sum([x in find_metrics for x in metrics]) / len(
        metrics))

    return list(objs_metrics[objs_metrics['o'] >= 0.5].index)


def zero_shot_classification(text: Doc) -> dict:
    detect_metrics = dict()
    data = load_data(os.path.join(data_path, 'raw',
                                  excel_file))
    metrics = data[metrics_column].values
    ngrams = generate_ngrams(text.tokens, 5)
    multi_zero_shot_classification_model(ngrams, metrics)
    # for metric in tqdm(metrics):
    #     for ngram in tqdm(ngrams):
    #         if zero_shot_classification_model(ngram, metric):
    #             detect_metrics.update({metric: ngram})
    return detect_metrics if detect_metrics else False


def tokenize_premise_hypothesis(sequence, label, device):
    premise = sequence
    hypothesis = f'Этот текст про {label}.'
    with torch.no_grad():
        return tokenizer.encode(premise, hypothesis, return_tensors='pt',
                                truncation_strategy='only_first').to(device)


def prepare_premise_hypothesis_data(x):
    x_max_len = max([tokenize.shape[1] for tokenize in x])
    x = [F.pad(input=tokenize, pad=(0, x_max_len - tokenize.shape[1]), mode='constant', value=0) for tokenize in x]
    x = torch.vstack(x)
    attention = x.detach().clone()
    attention[attention != 0] = 1
    attention[:, 0] = 1
    return x, attention


def predict_hypothesis(x: torch.tensor, attention: torch.tensor):
    with torch.no_grad():
        logits = nli_model(input_ids=x, attention_mask=attention)[0].to('cpu')

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:, 1].numpy()

    return prob_label_is_true > 0.5


def multi_zero_shot_classification_model(sequences: list, labels: list):
    device = check_device()
    predicts = []

    x = [tokenize_premise_hypothesis(sequence, label, device) for sequence in sequences for label in labels]
    x, attention = prepare_premise_hypothesis_data(x)

    # run through model pre-trained on MNLI
    x.to(device)
    nli_model.to(device)

    for number in range(x.shape[0] // 100 + 1):
        start_number = number * 100
        end_number = (number + 1) * 100
        predict = predict_hypothesis(x[start_number:end_number], attention[start_number:end_number])
        predicts.append(predict)

    predicts = np.hstack(predicts)
    out_put = dict(np.array([(sequence, label) for sequence in sequences for label in labels])[predicts])
    return out_put
