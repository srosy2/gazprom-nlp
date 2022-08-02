# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import json
import os
import re
import openpyxl
from tqdm import tqdm
import pandas as pd


sheet_name = 'Лист1'
data_path = 'data'
regular_expression_content = r'"content" : \[.*?\]'
regular_expression_title = r'"title" : \".*?\"'
regular_expression = r'"title" : \".*?\",  "content" : \[.*?\]'
delete_text = r"Располагайте  объекты в рамках модульной сетки,  заданной направляющими.  Для отображения/ скрытия " \
              r"направляющих используйте Alt+F9. Для смены уровней текста используйте  выделение+Tab. Для возврата на " \
              r"предыдущий уровень выделите  строку и нажмите  Shift+Tab. Первый уровень Второй уровень Третий " \
              r"уровень  ‹#› Газпром нефть Gazprom Neft "
replace_text = [('\t', ' '), ('\n', ' '), ('""" ]', '"]'), ('[ """', '["'),
                ('"""', '"'), ('""', '"'), (delete_text, '')]


def load_data(file_name: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_excel(file_name, sheet_name=sheet_name)
    return df


def prepare_text(text: str) -> str:
    new_text = text.split(' ')
    new_text = ' '.join(list(filter(lambda x: x != '', new_text)))
    for func in replace_text:
        new_text = new_text.replace(*func)
    return new_text


def prepare_title(text: str) -> str:
    return text[11: -1]


def prepare_content(text: str) -> str:
    text = text[17: -4]
    return text


def get_content_title(text: str) -> list:
    title_content = re.findall(regular_expression, text)
    title_content = [{'title': prepare_title(re.findall(regular_expression_title, x)[0]),
                      'content': prepare_content(re.findall(regular_expression_content, x)[0])}
                     for x in tqdm(title_content)]
    return title_content


def load_file(file_name: str) -> str:
    with open(file_name, encoding="utf-8") as file:
        text = file.read()
    return text


def save_file(file_name: str, file_obj: list) -> None:
    with open(file_name, 'w') as file:
        json.dump(file_obj, file)


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    content = load_file(os.path.join(input_filepath, 'lessons.json'))
    content = prepare_text(content)
    cnt_title = get_content_title(content)
    save_file(os.path.join(output_filepath, 'lessons.json'), cnt_title)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(os.path.join(project_dir, 'data', 'raw'), os.path.join(project_dir, 'data', 'processed'))
