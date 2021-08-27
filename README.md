# IA-Net
Code for EMNLP 2021 paper

**Progressively Guide to Attend: An Iterative Alignment Framework for Temporal Sentence Grounding** <br />
**[Paper]** <br />


### Prerequisites
* Python 3.6
* Pytorch >= 0.4.0

### Preparation
* Download Pretrained [Glove Embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* Download Extracted Features of [Three Datasets](https://github.com/liudaizong/CSMGAN) or the Enhanced Features of [Three Datasets](https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw)

### Training
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset ActivityNet --feature-path /yourpath/ActivityCaptions/ActivityC3D --train-data data/activity/train_data_gcn.json --val-data data/activity/val_data_gcn.json --test-data data/activity/test_data_gcn.json --max-num-epochs 20 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-7 --train --model-saved-path models_activity
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset TACOS --feature-path /yourpath/TACOS/TACOS --train-data data/tacos/TACOS_train_gcn.json --val-data data/tacos/TACOS_val_gcn.json --test-data data/tacos/TACOS_test_gcn.json --max-num-epochs 60 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --train --model-saved-path models_tacos --batch-size 64
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset Charades --feature-path /yourpath/Charades --train-data data/charades/train.json --val-data data/charades/test.json --test-data data/charades/test.json --max-num-epochs 80 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --train --model-saved-path models_charades --batch-size 64 --max-num-frames 64

### Evaluation
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset ActivityNet --feature-path /yourpath/ActivityCaptions/ActivityC3D --train-data data/activity/train_data_gcn.json --val-data data/activity/val_data_gcn.json --test-data data/activity/test_data_gcn.json --max-num-epochs 20 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-7 --evaluate --model-load-path /your/model/path
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset TACOS --feature-path /yourpath/TACOS/TACOS --train-data data/tacos/TACOS_train_gcn.json --val-data data/tacos/TACOS_val_gcn.json --test-data data/tacos/TACOS_test_gcn.json --max-num-epochs 40 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --evaluate --batch-size 64 --model-load-path /your/model/path
    $ python main.py --word2vec-path /yourpath/glove_model.bin --dataset Charades --feature-path /yourpath/Charades --train-data data/charades/train.json --val-data data/charades/test.json --test-data data/charades/test.json --max-num-epochs 40 --dropout 0.2 --warmup-updates 300 --warmup-init-lr 1e-07 --lr 4e-4 --num-heads 4 --num-gcn-layers 2 --num-attn-layers 2 --weight-decay 1e-8 --evaluate --batch-size 64 --max-num-frames 64 --model-load-path /your/model/path

### Citation
If you use this code please cite:

```
@inproceedings{liu2021progressively,
    title={Progressively Guide to Attend: An Iterative Alignment Framework for Temporal Sentence Grounding},
    author={Liu, Daizong and Qu, Xiaoye and Zhou, Pan},
    booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2021}
}

@inproceedings{liu2020jointly,
    title={Jointly Cross- and Self-Modal Graph Attention Network for Query-Based Moment Localization},
    author={Liu, Daizong and Qu, Xiaoye and Liu, Xiaoyang and Dong, Jianfeng and Zhou, Pan and Xu, Zichuan},
    booktitle={Proceedings of the 28th ACM International Conference on Multimedia (MM’20)},
    year={2020}
}
```

## Acknowledgements
This code borrows several code from [CSMGAN](https://github.com/liudaizong/CSMGAN). If you use our code, please consider citing the original papers as well.