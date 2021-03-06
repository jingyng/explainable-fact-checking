# [Explainable Fact Checking through Question Answering](https://arxiv.org/abs/2110.05369)

We use wandb for recording all experiments, you can find the details from [here](https://wandb.ai/fakejing/icassp2022).
## Dataset
The dataset is available at: [Fool Me Twice](https://github.com/google-research/fool-me-twice)

## Question and Answer Generation
We adopt the approach to generate questions and answers from claims simultaneously. Specifically, we follow the instruction from [Nan et al.](https://github.com/amazon-research/fact-check-summarization) to fine-tune the `BART-large` model the *SQuAD* and *NewsQA* datasets to generation question-answer pairs. 

We make the generated questions and answers for the dataset used in this paper available through [Google Drive](https://drive.google.com/file/d/14SB_pyzwBAM7x4dIHYmFvV5ftUP4oKsQ/view?usp=sharing).

## Question Answering 
We utilize FARM framework from [deepset.ai](https://github.com/deepset-ai/FARM) to generate answers from evidence, and the question-answering model is `deepset/electra-base-squad2`, which is an extractive QA model. The code to generate the answers from evidence is: `Fool-me-twice-answering.py`. The generated answers are also available through [Google Drive](https://drive.google.com/file/d/14wSjRvnqlsq9PIFzoUK-kHJ3TnAKpg0-/view?usp=sharing).

## Answer Comparison
We have generated questions and answers from claims, and extracted answers from evidence. Now we will compare the answer pairs to predict the label of the claim. 

To train our attention model, we first need to prepare the data in the right format, which is provided in `Data-organization.py`.

You need to change the data directory to your own, then run `python Data-organization.py`.

After organizing input data, for for encoding all input representation, we use [`microsoft/mpnet-base`](microsoft/mpnet-base). The code for training is in `qa-additive-attention.py`.

Example: `python qa-additive-attention.py --run_name final-fm2-nocat-cq-aa-gold --bert1 microsoft/mpnet-base --bsz 32 --epochs 5 --num_questions 10`

## Baselines
We compare our answer comparison model with five baselines:
- Blackbox: the baseline method from the orignial FM2 dataset paper.
- QUALS score: automatic metric for checking factual consistency proposed by [Nan et al.](https://arxiv.org/abs/2105.04623), QUALS scores of our results are available at [Google Drive](https://drive.google.com/file/d/1GiCOJCnfcFHANMlHu5INkI-JoxGo7cBN/view?usp=sharing).
- Token level F1-score: a standard metric for question-answertasks, which counts words overlap between two answers.
- BERTscore:a common metric for measuring the similarity of two sentences. We use the default model `roberta-large`.
- Cosine similarity: a metric also used for measuring sentence similarity. We use sentence transformer `all-mpnet-base-v2`.

We provide a jupyter notebook named `Baselines.ipynb` for the implementation of baselines.

## Ablation Study
For the ablation study, we removed the attention layer of our proposed attention model. To reproduce, first organize the data following the code in `Data-organization-ablation.py`.

Five different inputs are compared:
- C: claim only (code can be found in `ablation-c.py`)
- Q: questios only (all questions concatenated,code can be found in `ablation-q.py`)
- AA: answer-pairs only (all answers concatenated,code can be found in `ablation-aa.py` )
- Q-AA: questions and answer pairs (code can be found in `ablation-q-aa.py`)
- CQ-AA: claim, questions and answer pairs (code can be found in `ablation-cq-aa.py`)

## Citation
If you find our work interesting and would like to adapt to your research, please kindly cite our paper (pre-print, official published paper coming soon).
```
@misc{yang2021explainable,
      title={Explainable Fact-checking through Question Answering}, 
      author={Yang, Jing and Vega-Oliveros, Didier and Seibt, Ta??s and Rocha, Anderson},
      eprint={2110.05369},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      year={2021}
}
```
