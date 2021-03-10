# eComTag dataset
Thanks for you interest in our WSDM 2021 Paper [Abstractive Opinion Tagging](https://arxiv.org/pdf/2101.06880.pdf).

## Introduction 
eComTag is a new dataset collected from several Chinese e-commerce websites for abstractive opinion tagging research.

It includes nine domains: **Cosmetic** (18740 items), **Electronics** (13249), **Food** (5296), **Entertainment** (4137), **Books** (4067), **Sports** (1987), **Clothes** (1615), **Medical** (821), and **Furniture** (156).
We put the different domain data

Besides, we annotate each review with a binary salience label via human judgment. If a review is item-related, we label the review as 1, otherwise 0. 

Each item sample in eComTag consists of:
* a set of reviews
* a set of corresponding salience labels
* a sequence of opinion tags

Here is the statistics of eComTag:

|  Data   | Sample  |
|  :----  | :----  |
| Training  | 40,162 |
| Validation  | 4,953 |
| Test  | 4,953 |

You can take a look at the eComTag dataset by running ```python3 eComTag```.


## Reference
If you find our dataset useful, please cite our paper as follows:
```bibtex
@inproceedings{li-etal-2020-aot,
  title={Abstractive Opinion Tagging},
  author={Qintong Li and Piji Li and Xinyi Li and Zhaochun Ren and Zhumin Chen and Maarten de Rijke},
  booktitle={WSDM},
  year={2021},
}
```