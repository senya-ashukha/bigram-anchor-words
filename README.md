# Bigram Anchor Words Topic Model

Implementation for the [Bigram Anchor Words Topic Model](https://link.springer.com/chapter/10.1007/978-3-319-52920-2_12) paper. 
Bag of words is very poor text representation, since that, in traditional topic models, we are losing a lot of information. 
The project goal is to combine linguistic with statistical topic models. 
We propose new Anchor Words Topic Model [1] such as bigrams also could be anchor words.

[1] Sanjeev A., Rong G.: A Practical Algorithm for Topic Modeling with Provable Guarantees (NIPS, 2012) 

# Results

Here are an example of anchor words. 
Metrics are also good and could be found in the paper.
<p align="center">
<img height="318" src="http://ars-ashuha.ru/images/anchors.png"/>
</p>

# Experiments 

You could use following code to repeat published results. A simple way to repeat experiments is to try to understand examples =) I'm sorry that documentation is absent.  

```bash
cd bigram-anchor-words
ipython ./examples/{corpus}/{model}.py
```

# Citation

If you found this code useful please cite our paper

```
@inproceedings{ashuha2016bigram,
  title={Bigram Anchor Words Topic Model},
  author={Ashuha, Arseniy and Loukachevitch, Natalia},
  booktitle={International Conference on Analysis of Images, Social Networks and Texts},
  pages={121--131},
  year={2016},
  organization={Springer}
}
```
