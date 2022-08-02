import json
from src.models.predict_model import zero_shot_classification, equal_ngrams, syntax_detection
from src.features.build_features import parse_text
from tqdm import tqdm
import os
import logging
from ontology_base import CreateServer

data_path = 'data/processed'
data_save = 'data/external'

#create a logger
logger = logging.getLogger(__name__)
#set logger level
logger.setLevel(logging.INFO)
#or you can set the following level
#logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('reports/mylog.log')
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)

AGRAPH_HOST: str = 'localhost'
AGRAPH_PORT: int = 10035
AGRAPH_USER: str = 'test'
AGRAPH_PASSWORD: str = 'xyzzy'


def load_data(file_name: str) -> list:

    with open(file_name) as file:
        data = json.load(file)
    return data


def save_content(title: str, content: str) -> None:

    with open(os.path.join(data_save, f'{title}.txt'), 'w', encoding='utf-8') as file:
        file.write(content)


def save_data(data: dict, file_name: str) -> None:

    with open(file_name, 'w') as file:
        json.dump(data, file)


def run(server: CreateServer, save: bool = False):

    title_metric = dict()
    list_content_title = load_data(os.path.join(data_path, 'lessons.json'))

    for number, content_title in tqdm(enumerate(list_content_title)):
        title = content_title.get('title')
        content = parse_text(content_title.get('content'))
        title_metric.update({f'{title}_{number}': equal_ngrams(content, server)})

    content_quantity = [len(title_metric.get(title)) for title in title_metric.keys()]
    content_metrics = sorted(zip(content_quantity, list_content_title), key=lambda cont: cont[0], reverse=True)[:5]

    if save:
        for number, cont_tl in enumerate(content_metrics):
            try:

                save_content(f"{cont_tl[1].get('title')}_{number}", cont_tl[1].get('content'))
            except FileNotFoundError:

                save_content(str(number), cont_tl[1].get('content'))
        # save_data(title_metric, os.path.join(data_path, 'title_metric.json'))


if __name__ == '__main__':
    logger.info('start program')
    server = CreateServer(host=AGRAPH_HOST, port=AGRAPH_PORT,
                          user=AGRAPH_USER, password=AGRAPH_PASSWORD)
    server.access_repo('repo')
    server.find_object()

    run(server)

    server.quit()

    logger.info('finish program')
