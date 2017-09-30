# Spk-Dialogue: Speaker Role Contextual Modeling for Language Understanding and Dialogue Policy Learning
*An implementation of the IJCNLP 2017 paper:
[Speaker Role Contextual Modeling for Language Understanding and Dialogue Policy Learning](#).*

## Content
* [Data](#data)
* [Reference](#reference)

## Data
the data used in the paper is [DSTC4](http://www.colips.org/workshop/dstc4/)

## Requirements
Tensorflow ver1.2 CUDNN ver5.1
Python 2.7

## Usage
* Change the path in **slu\_preprocess.py** line 29 to your custom GloVe file path.
* **Note that user should add a 200 dims 0.0 at the end of the GloVe file.**
* cd into every sub-directory and run `python2.7 slu.py` will reproduce the results.

## Reference

Main papers to be cited
```
@inproceedings{chen2017speaker,
  author    = {Ta-Chung Chi and Po-Chun Chen and Shang-Yu Su and Yun-Nung Chen},
  title	    = {Speaker Role Contextual Modeling for Language Understanding and Dialogue Policy Learning},
  booktitle = {Proceedings of 2017 International Joint Conference on Natural Language Processing},
  year	    = {2017},
  address   = {Taipei, Taiwan}
}
