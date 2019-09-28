# Gendered Pronoun Resolution

This is an another one approach to solve the competition from kaggle
[Gendered Pronoun Resolution](https://www.kaggle.com/c/gendered-pronoun-resolution).

18th place over 263 (silver medal) with 0.21618 log-loss score.

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

First download the train data with command

```bash
bash ./download_data.sh
```

and test data from the competition link.

To train the model run

```python
python ./src/train.py
```

### Approach

Detailed solution see in
[this presentation](presentation/Gendered_Pronoun_Resolution_Pair_pronouns_to_their_correct_entities.pdf) (russian).
