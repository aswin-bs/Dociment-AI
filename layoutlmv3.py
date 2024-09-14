import json
import os
import ast
from pathlib import Path
import datasets
from PIL import Image
import pandas as pd

logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={}
}
"""
_DESCRIPTION = """\
This is a sample dataset for training layoutlmv3 model on custom annotated data.
"""

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

data_path = r'./'

class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for InvoiceExtraction Dataset"""
    def __init__(self, **kwargs):
        super(DatasetConfig, self).__init__(**kwargs)

class InvoiceExtraction(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DatasetConfig(name="EntityExtraction", version=datasets.Version("1.0.0"), description="EntityExtraction dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'employerName', 'employerAddressStreet_name', 'employerAddressCity',
                                'employerAddressState', 'employerAddressZip', 'einEmployerIdentificationNumber',
                                'employeeName', 'ssnOfEmployee', 'box1WagesTipsAndOtherCompensations',
                                'box2FederalIncomeTaxWithheld', 'box3SocialSecurityWages',
                                'box4SocialSecurityTaxWithheld', 'box16StateWagesTips', 'box17StateIncomeTax',
                                'taxYear', 'OTHER'
                            ]
                        )
                    ),
                    "image_path": datasets.Value("string"),
                    "image": datasets.features.Image()
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="",
        )

    def _split_generators(self, dl_manager):
        dest = os.path.join(data_path, 'layoutlmv3')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": os.path.join(dest, "train.txt"), "dest": dest}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": os.path.join(dest, "test.txt"), "dest": dest}
            ),
        ]

    def _generate_examples(self, filepath, dest):
        df = pd.read_csv(os.path.join(dest, 'class_list.txt'), delim_whitespace=True, header=None)
        id2labels = dict(zip(df[0].tolist(), df[1].tolist()))

        logger.info("⏳ Generating examples from = %s", filepath)

        item_list = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item_list.append(line.rstrip('\n\r'))

        for guid, fname in enumerate(item_list):
            try:
                data = ast.literal_eval(fname)
                image_path = os.path.join(dest, data['file_name'])
                image, size = load_image(image_path)
                boxes = data['bboxes']
                text = data['tokens']
                label = data['ner_tags']

                boxes = [normalize_bbox(box, size) for box in boxes]
                flag = any(j > 1000 for i in boxes for j in i)
                if flag:
                    logger.warning(f"Bounding box values exceed 1000 in {image_path}")

                yield guid, {
                    "id": str(guid), "tokens": text, "bboxes": boxes, "ner_tags": label,
                    "image_path": image_path, "image": image
                }
            except Exception as e:
                logger.error(f"Error processing {fname}: {e}")
