from msk_envs.utils.pdf_log_builder import create_pdf_output
from msk_envs.utils.frame_parser import FrameData
import gzip
import json


def main():
    file = "/home/marth/Documents/msk_envs/deploy_frame_data_0.json.gz"
    with gzip.open(file, 'rt') as f:
        frame_data_raw = json.load(f)
    frame_data = []
    for frame_dict in frame_data_raw:
        frame = FrameData.from_dict(frame_dict)
        frame_data.append(frame)
    create_pdf_output(frame_data, "test_output.pdf")


if __name__ == "__main__":
    main()
