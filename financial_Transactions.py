# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transactions Dataset"""


import csv

import datasets


# TODO: Add transaction citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{mccreery2020effective,
      title={Effective Transfer Learning for classifying Transactions},
      author={AK},
      year={2020},
      eprint={2008.13546},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
"""


_DESCRIPTION = """\
This dataset consists of 378 transactions performed on account and categorised according to the description of the transaction.
"""

_HOMEPAGE = "https://github.com/alokkulkarni/transactions"

_LICENSE = ""


_URL = "https://raw.githubusercontent.com/alokkulkarni/transactions/master/transactions.csv"


class MedicalQuestionsPairs(datasets.GeneratorBasedBuilder):
    """Transactions Dataset"""

    def _info(self):
        features = datasets.Features(
            {
                "Account": datasets.Value("string"),
                "Date": datasets.Value("string"),
                "Amount": datasets.Value("string"),
                "Description": datasets.Value("String"),
                "Location": datasets.Value("string"),
                "Category": datasets.features.ClassLabel(num_classes=16, names=[ "Fuel", "Income", "Credit_Card_Payment", "Entertainment", "Shopping", "Rent", "Subscriptions", "Healthcare", "Groceries", "Cash_Withdrawal", "Loan_Payment", "Utilities", "Automotive", "Online_Shopping", "Dining_Out", "Miscellaneous" ]),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_file = dl_manager.download_and_extract(_URL)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_file})]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = csv.reader(f)
            for id_, row in enumerate(data):
                yield id_, {
                    "Account": row[0],
                    "Date": row[1],
                    "Amount": row[2],
                    "Description": row[3],
                    "Location" : row[4],
                    "Category": row[5]
                }

