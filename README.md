# [Explainable Fact Checking through Question Answering](https://arxiv.org/abs/2110.05369)
Code will be available soon

## Dataset
The dataset is available at: [Fool Me Twice](https://github.com/google-research/fool-me-twice)

### Question and Answer Generation
We adopt the approach to generate questions and answers from claims simultaneously. Specifically, we follow the instruction from [Nan et al.](https://github.com/amazon-research/fact-check-summarization) to fine-tune the BART-large model to generation question-answer pairs. 

We also make the generated questions and answers for the dataset used in this paper available through [google drive](https://drive.google.com/file/d/14SB_pyzwBAM7x4dIHYmFvV5ftUP4oKsQ/view?usp=sharing)

### Question Answering 
To extract answers from evidence, we utilize FARM framework from [deepset.ai](https://github.com/deepset-ai/FARM) to generate answers from evidence, and the question-answering model is *deepset/electra-base-squad2*.

### Answer Comparison
Given a claim, we have generated its associated questions and answers, we also extracted answers from the evidence of the claim. Now we will compare the answer pairs to predict the label of the claim. For for encoding all input representation, we use *[microsoft/mpnet-base](microsoft/mpnet-base)*.
