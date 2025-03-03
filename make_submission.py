import json
import os
import shutil
from datetime import datetime
from distutils import dir_util
from os import path

import fire
import kaggle

from dpfk import util


def make_submission(inference_config="config/infer.yml",
                    run_base_dir="wandb",
                    save_local=False):
    submission_folder = datetime.now().strftime("%Y%m%d-%H%M")
    submission_dir = path.join("submissions", submission_folder)

    dir_util.mkpath(submission_dir)
    dir_util.copy_tree('src', path.join(submission_dir, 'src'))
    dir_util.copy_tree('dpfk', path.join(submission_dir, 'src', 'dpfk'))

    with open(path.join(submission_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(
            {
                "title": submission_folder,
                "id": f"xvrhlt/{submission_folder}",
                "licenses": [{
                    "name": "CC0-1.0"
                }]
            }, f)

    dst_run_dir = path.join(submission_dir, "runs")
    dir_util.mkpath(dst_run_dir)

    shutil.copy(inference_config, submission_dir)
    config = util.read_config(inference_config)

    for run in config['runs']:
        src_path = path.join(run_base_dir, run)
        dst_path = path.join(dst_run_dir, run)
        dir_util.copy_tree(src_path, dst_path)

    kaggle.api.dataset_create_new(submission_dir, dir_mode='zip')
    if save_local:
        shutil.make_archive(submission_dir, 'zip', root_dir='submissions')
    dir_util.remove_tree(submission_dir)


if __name__ == "__main__":
    fire.Fire(make_submission)
