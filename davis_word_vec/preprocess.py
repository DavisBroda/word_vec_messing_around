from typing import *
import re
import glob


class LineIterator(Iterable[List[str]]):
    def __init__(self, file_absolute_path: str):
        self.f_name = file_absolute_path

    def __iter__(self):
        for file in glob.glob(self.f_name):
            for raw_line in open(file, mode='r', encoding='utf-8', errors='replace'):
                yield self.parse_line(raw_line)

    @staticmethod
    def parse_line(raw_line: str):
        if LineIterator.is_xml_tag_only(raw_line):
            return []
        return LineIterator.preprocess_line(raw_line)

    @staticmethod
    def is_xml_tag_only(line: str) -> bool:
        open_tag = (line.startswith("<") or line.startswith("<\\"))
        close_tag = line.endswith(">")
        if open_tag and close_tag:
            return True
        else:
            return False

    @staticmethod
    def preprocess_line(line: str) -> List[str]:
        # include spaces for word splitting later, and apostrophe for contractions
        only_aplha_num = re.sub(r'[^a-zA-Z0-9 \']+', '', line)

        return only_aplha_num.split(" ")


def read_input_file(absolute_path: str) -> Iterable[List]:
    return LineIterator(absolute_path)

def get_file_line_count(absolute_path: str) -> int:
    count = 0
    for file in glob.glob(absolute_path):
        for line in open(file, mode='r', encoding='utf-8', errors='replace'):
            count = count + 1
    print("count: ", count, " for path: ", absolute_path)
    return count








