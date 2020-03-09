# Scholarly Paper Recommendation

The project is about recommending relevant scholarly papers to researchers based on their research interests which can be inferred from their published papers. We used a pretrained word2vec model to get feature vectors of user's profile and candidate paper to recommend


## Dataset description
The project use the [dataset 2](https://scholarbank.nus.edu.sg/handle/10635/146027) released by Kazunari Sugiyama and Min-Yen Kan in National University of Singapore. We appreciated the creator of the dataset for sharing the raw pdf files .

**Raw dataset includes:**

* Research interests of 50 researchers
	* Each researcher's published papers in DBLP list (full text in pdf format)
	* Paper IDs relevant to each researcher's interests (ground truth)
* Candidate papers to recommend:
	* full text in pdf format: 95,238
	* full text in txt format: 95,125 (some pdf files were not readable by python packages pdfminer.six, therefore were failed to be converted to txt files)
 
**Statistics of user profiles:**

|      |   number of publications  | number of relevant papers|
|:-------------: | :-------------: | :-------------: |
| Min            | 2               |8              |
| Max            | 31              |241            |
| Avg            | 10.7           |74.56          |
|Total|535|3728|



	




## Prerequisites

Python packages: pdfminer.six

Accessing to some data that could not provide in the repo via google drive, downloads ths files and alter the path when using this file in script:

* [google news word2vec model](https://drive.google.com/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM): load this model from local file or via API
* [corpus for training my own word2vec model](https://drive.google.com/file/d/12wYxounFPHThUgITpqq-ViGsWBLrpUy3/view?usp=sharing)
* [candidate papers to be recommended](https://drive.google.com/file/d/1iVFhC7bcgls8o6PwIRTmlxXzFyI4Y4Qv/view?usp=sharing)
* [my own word2vec model](https://drive.google.com/open?id=1-47kS8UgQAIKv6sEuDUlvWwvR53L7I84)

## Experiment pipelines

1. Data preprocessing
	* Converting all papers (researchers's published papers and candidate papers to recommend) from pdf format to plain text format (pdf2text.py)
	* Text cleaning (text_cleaning.py)
	* Using all cleaned candidate papers to generate a corpus for word2vec model training
		* The corpus contains 95,125 candidate papers after text cleaning (see corpus.txt via google drive link)
		* There's one document/candidate paper per line, tokens separated by whitespace
	
2. Pretraining a  word2vec model based on the corpus of all candidate papers to recommend
	* Corpus (see corpus.txt via google drive link)
	* Key training parameters setting: **min_count=5, size=300, window=5**
		* **min_count** It is for pruning the internal dictionary. Words that appear only once or twice in a billion-word corpus are probably uninteresting typos and garbage. In addition, there’s not enough data to make any meaningful training on those words, so it’s best to ignore them. Default value of min_count=5
		* **size**: The number of dimensions (N) of the N-dimensional space that gensim Word2Vec maps the words onto. Bigger size values require more training data, but can lead to better (more accurate) models. Reasonable values are in the tens to hundreds.
		* **window**: The maximum distance between a target word and words around the target word.
	* Saving model

3. Candidate papers representation ( *i from 1 to 95125* )
	* Using the pretrain word2vec model to create feature vector for each candidate paper ( *FV<sub>pi</sub>* )
	* using a normal word2vec model to create feature vector for each candidate paper ( *FV<sub>ni</sub>* )
	
4. Researchers interest representation ( *n from 1 to 50* ): generating feature vectors to represent each researcher's interest
	* Using pretrain word2vec model
		* Using the most recent publication of a specific researchers as input to get feature vector the researcher's interest ( *FV<sub>pmn</sub>*, )
		* Using researcher's all publications with the same weight as input( *FV<sub>psn</sub>* )
		* Using researcher's all publications with the different weight stategy as input, the more recent the publication is, the more important ( *FV<sub>pdn</sub>* )
		
	* Using normal word2vec model
		* Using the most recent publication of a specific researchers as input to get feature vector the researcher's interest ( *FV<sub>nmn</sub>* )
		* Using researcher's all publications with the same weight as input( *FV<sub>nsn</sub>* )
		* Using researcher's all publications with the different weight stategy as input, the more recent the publication is, the more important ( *FV<sub>ndn</sub>* )
		
5. Calculating similarity between *FV* for different candidate papers and *FV* for different researchers', get most 10 relervant candidate papers for each type of researchers' interest representation (see similarity\_calculation_mr\_CP.py)

6. Evaluateing similar ranking results by metrics NDCG@10, P@10,MRR
	* result of using researcher's most recent publication (see evaluation\_metrics.py and evaluation4mr_GoogleNews.py)

|     |vector representation model| NDCG@10 |  P@10  |MRR     |
|:---:|:-------------: | :-------------: | :----: | :-----:|
| using most recent paper| CP model|0.3665 |**0.1340** |0.2736 |
| using most recent paper| GN model|0.3389|  0.1080 |0.2747 |
| using all papers| CP model|**0.3731** |0.114 |**0.3282** |
| using all papers| GN model| 0.2376 | 0.074  |0.1921 |
| using all papers with different weight| CP model| 0.3709 | 0.132  |0.2969 |
| using all papers with different weight| GN model| 0.3270 | 0.1119  |0.2666 |





## Contributior
* **Name**
* **Name**
## Authors

* **Xueli Pan** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used

