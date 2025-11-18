import json
import os
from pathlib import Path

import click


@click.command
@click.option("--data_folder")
def create_dataset(data_folder):
    sub = os.listdir(os.path.join(data_folder, "wavs"))
    if len(sub) == 0:
        print("invalid dataset!!")
        return
    multi_speakers = os.path.isdir(sub[0])
    fwrite = open(os.path.join(data_folder, "metadata.jsonl"), 'w', encoding='utf-8')
    for line in open(os.path.join(data_folder, "metadata.csv"), 'r', encoding='utf-8').read().split("\n"):
        parts = line.split("|")
        if multi_speakers:
            fwrite.write(json.dumps({
                "audio": parts[0],
                "text": parts[1],
                "source": Path(parts[0]).as_posix().split("/")[0]
            }, ensure_ascii=False) + "\n")
        else:
            fwrite.write(json.dumps({
                "audio": parts[0],
                "text": parts[1],
            }, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    create_dataset()
