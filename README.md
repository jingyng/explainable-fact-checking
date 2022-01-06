# [Explainable Fact Checking through Question Answering](https://arxiv.org/abs/2110.05369)
Code will be available soon

## Dataset
The dataset is available at: [Fool Me Twice](https://github.com/google-research/fool-me-twice)

### Question and Answer Generation
We adopt the approach to generate questions and answers from claims simultaneously. Specifically, we follow the instruction from [Nan et al.](https://github.com/amazon-research/fact-check-summarization) to fine-tune the BART-large model to generation question-answer pairs.

### Question Answering 
To extract answers from evidence, we utilize FARM framework from [deepset](https://github.com/deepset-ai/FARM) to generate answers from evidence, and the question-answering model is *deepset/electra-base-squad2*.
### Answer Comparison
After having 
