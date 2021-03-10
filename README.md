# Abstractive Opinion Tagging
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is the official implementation of the WSDM 2021 Paper [Abstractive Opinion Tagging](https://github.com/qtli/AOT).

If you have any question, please open an issue or contact <qtleo@outlook.com>.

## eComTag Dataset 
Please visit this [URL](https://qtli.github.io/qtli.github.io/project/ecomtag/) to read the details of the eComTag dataset.

Please use this [Google form](https://drive.google.com/drive/folders/1Vk0SjQbNdxL8_D9l6pJMkoI8KyDpPnLt?usp=sharing) to submit your information and request access to eComTag.

## Setup
Check the packages needed or simply run the command:
```console
pip3 install -r requirements.txt
```
For reproducibility purposes, we place the model checkpoints at [**Google Drive**](https://drive.google.com/drive/folders/1Vk0SjQbNdxL8_D9l6pJMkoI8KyDpPnLt?usp=sharing). You could download and move it under `/output/`.

> We would now set up our directories like this:

```
.
└── model
    └── ...
└── baselines
    └── ...
└── eComTag
    └── ...
└── utils
    └── ...
└── README.md
```

## Experiments
> Training models associated with AOT-Net will take more than one day. 
You can also run the trained models to test by changing "False" to "True" in the bash commands.

AOT-Net
```bash
bash script/AOTNet.sh 0 False
```

w/o SSE
```bash
bash script/woSSE.sh 0 False
```

w/o RCR
```bash
bash script/woRCR.sh 0 False
```

w/o AF
```bash
bash script/woAF.sh 0 False
```

w/o AL
```bash
bash script/woAL.sh 0 False
```

## Baselines

RNN
```bash
bash script/RNN.sh 0 False
```

PG-Net
```bash
bash script/PGNet.sh 0 False
```

Transformer
```bash
bash script/Transformer.sh 0 False
```


## Reference
If you find our code useful, please cite our paper as follows:
```bibtex
@inproceedings{li-etal-2020-aot,
  title={Abstractive Opinion Tagging},
  author={Qintong Li and Piji Li and Xinyi Li and Zhaochun Ren and Zhumin Chen and Maarten de Rijke},
  booktitle={WSDM},
  year={2021},
}
```





