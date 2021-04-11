# DCR
Deep Conversational Recommender in Travel

## Main Idea
Using the MultiWOZ data set, a model TCR for cross-domain conversation recommendation is proposed. TCR is composed of three parts. Firstly, the topic model is used to capture the current topic according to the context of the conversation; secondly, a corresponding method is used to construct a graph using the venue and slot value information in the data set, and then the GCN is used to obtain the embedded representation of the venue entity. Use the groundtruth tag in the conversation to train the recommendation model. Finally, use the pointed integration mechanism, combined with the recommendation and dialogue models obtained by the previous pre-training, to generate the final response.

## run DCR

Under the DCR folder

```
cd data
python generate_data.py

cd ..
mkdir logs
mkdir results/gcn_no_activation

python main.py
```

```
@article{liao2019deep,
  title={Deep conversational recommender in travel},
  author={Liao, Lizi and Takanobu, Ryuichi and Ma, Yunshan and Yang, Xun and Huang, Minlie and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:1907.00710},
  year={2019}
}
```

Please unzip the [``DCRdata.zip``](https://drive.google.com/open?id=1bv-AcHOM6hZV6sLrBWLY5jmc792rNoYS) under the subdirectory ``DCR/data``

