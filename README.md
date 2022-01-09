# [Explainable Fact Checking through Question Answering](https://arxiv.org/abs/2110.05369)
Code will be available soon

## Dataset
The dataset is available at: [Fool Me Twice](https://github.com/google-research/fool-me-twice)

### Question and Answer Generation
We adopt the approach to generate questions and answers from claims simultaneously. Specifically, we follow the instruction from [Nan et al.](https://github.com/amazon-research/fact-check-summarization) to fine-tune the `BART-large` model to generation question-answer pairs. 

We also make the generated questions and answers for the dataset used in this paper available through [Google Drive](https://drive.google.com/file/d/14SB_pyzwBAM7x4dIHYmFvV5ftUP4oKsQ/view?usp=sharing).

### Question Answering 
To extract answers from evidence, we utilize FARM framework from [deepset.ai](https://github.com/deepset-ai/FARM) to generate answers from evidence, and the question-answering model is `deepset/electra-base-squad2`. The code to generate the answers from evidence is: `Fool-me-twice-answering.py`. The generated answers are also available through [Google Drive](https://drive.google.com/file/d/14wSjRvnqlsq9PIFzoUK-kHJ3TnAKpg0-/view?usp=sharing).

### Answer Comparison
Given a claim, we have generated its associated questions and answers, we also extracted answers from the evidence of the claim. Now we will compare the answer pairs to predict the label of the claim. For for encoding all input representation, we use `[microsoft/mpnet-base](microsoft/mpnet-base)`.

### Baselines
We compare our answer comparison model with five baselines:
- Blackbox: the baseline method from the orignial FM2 dataset paper
- QUALS score: automatic metric for checking factual consistency proposed by [Nan et al.](https://arxiv.org/abs/2105.04623)
- Token level F1-score: a standard metric for question-answertasks, which counts words overlap between two answers
- BERTscore:a common metric for measuring the similarity of two sentences. We use the default model `roberta-large`
- Cosine similarity: a metric also used for measuring sentence similarity. We use sentence transformer `all-mpnet-base-v2`

### Ablation Study
For the ablation study, we removed the attention layer of our proposed attention model. Five different inputs are compared:
- C: claim only
- Q: questios only (all questions concatenated)
- AA: answer-pairs only (all answers concatenated)
- Q-AA: questions and answer pairs
- CQ-AA: claim, questions and answer pairs
