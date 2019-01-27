# BreakDown Detection TF

Bad task, which is not worth put too much efforts, because the baseline is better than all teams.
Recommend try another downstream task: https://github.com/yhs-968/pyGRU4REC/tree/34514ddc4339717abb02da41762d9330643bf31d

## Enviroment
Python 3.6

tensorflow==1.4.0

Other packages: see `requirements.txt`

## Usage

First run
```
python train.py --learn_embed_matrix --decay_steps 1 --steps 3 --print_interval 1 --save_interval 1 --word_thre 2 --hidden_size 32
```

The training loss will be recorded by the tensorboard in `model/tfboard`
```
tensorboard --logdir="model/tfboard/" --host=127.0.0.1
```

Then run the test script
```
python test.py --model_file model.ckpt-1
```

Then use the evaluation script
```
python eval.py -p eval/test_jsons/ -o eval/textRNN/
```


## Baseline program for the task of dialogue breakdown detection

The baseline program uses words included in each utterance as features (Bag-of-Words) and detects dialogue breakdowns by using Conditional Random Fields (CRFs).
This program outputs three kinds of labels, O (not a breakdown), T (possible breakdown,) and X (breakdown) with probability distributions.
However, each distribution is deterministic, i.e., 1.0 for the output label, 0.0 for the others.

### Requirements

JavaSE-1.7+

### Launching baseline program

$java -jar DBDBaseline.jar -l (Directory containing training data) [-p (Directory containing test data)] -o (output directory) -t (threshold of concordance rate)


The model is trained using JSON files in the directory containing the training data specified with -t option. When training is finished, the model detects dialogue breakdowns on the JSON files in the directory containing the test data specified by -p and outputs the results to the directory specified by -o in JSON format.

If -p option is not specified, the model uses training data as test data.

-t is the option for determining the correct label for each instance from multiple annotated labels in training the model. The correct label for each instance is basically the majority of the labels given to the instance. Breakdown labels ("T" and "X") are, however, used only if the concordance rate among annotators is greater than or equal to the specified threshold, otherwise not-a-breakdown label ("O") is used. For example, if two annotators give "O", three give "T", and five give "X", then the concordance rate of label "X" is 5 / (2 + 3 + 5) = 0.5 which is the largest of the three. If the threshold specified by "-t" is 0.5 or less, the label of the utterance is determined to be "X", but if it is larger than 0.5 it to be "O".

 
### Acknowledgments

In this program, the MALLET is used as an implementation of Conditional Random Fields.
 McCallum, Andrew Kachites.  "MALLET: A Machine Learning for Language Toolkit." http://mallet.cs.umass.edu. 2002.
 
And lucene-gosen is used as a Japanese morphological analyzer.
 https://code.google.com/p/lucene-gosen/
 

## Evaluation script

### About
  This script evaluates the output of a dialogue breakdown detector. In addition to the accuracy, recall rate, F values are calculated.

### Execution environment
  Python 2.7.x

### How to run
  ~~~~~~~
  $python eval.py -p (Directory containing test data) -o (output directory) -t (threshold of concordance rate)
  ~~~~~~~
  When executed, it compares the dialogue data specified by -p with the detection result data specified by -o,
  and outputs the dialogue breakdown detection results.
  -t is the option for determining the reference label.
  The reference label is basically the majority of the labels given to an utterance.
  However, breakdown labels ("T" and "X") become reference labels 
  only if the ratio for these labels is greater than or equal to the specified threshold, otherwise not-a-breakdown label ("O") becomes the reference label.
  For example, if two annotators give "O", three give "T", and five give "X", then the ratio of label "X" is 5 / (2 + 3 + 5) = 0.5 which is the largest of the three. 
  If the threshold specified by "-t" is 0.5 or less, the label of the utterance is determined to be "X", but if it is larger than 0.5 it becomes "O".

### Evaluation items

  Accuracy: 
  The number of correctly classified labels divided by the total number of labels to be classified.

  Precision, Recall, F-measure (X): 
  The precision, recall, and F-measure for the classification of the X labels.

  Precision, Recall, F-measure (T+X): 
  The precision, recall, and F-measure for the classification of T + X labels; that is, PB and B labels are treated as a single label.

  JS divergence (O, T, X):
  Distance between the predicted distribution of the three labels and that of the gold labels calculated by Jensen-Shannon Divergence.

  JS divergence (O, T+X):
  JS divergence when T and X are regarded as a single label.

  JS divergence (O+T, X):
  JS divergence when O and T are regarded as a single label.

  Mean squared error (O, T, X):
  Distance between the predicted distribution of the three labels and that of the gold labels calculated by mean squared error.

  Mean squared error (O, T+X):
  Mean squared error when T and X are regarded as a single label.

  Mean squared error (O+T, X):
  Mean squared error when O and T are regarded as a single label. 

### File format

  The format of the breakdown detection result file is described as below. The evaluation script can only accept this format. 

  * Naming convention for input/output files

  The input dialogue file for dialogue breakdown detection is <dialogue-id>.log.json. The output result file will be <dialogue-id>.labels.json. 
  For example, if an input dialogue file is 1408278293.log.json, then the output result file will be 1408278293.labels.json.

  * Format of the output result file

  The output result file should be in a JSON format．An example is given below:

  ~~~~~~~
  ##### Example #####
  {
    "dialogue-id" : "1408278293",
    "turns" : [ {
      "turn-index" : 0,
      "labels" : [ {
        "breakdown" : "O",
        "prob-O" : 0.8,
        "prob-T" : 0.2,
        "prob-X" : 0.0
      } ]
    }, {
      "turn-index" : 2,
      "labels" : [ {
        "breakdown" : "O",
        "prob-O" : 0.75,
        "prob-T" : 0.15,
        "prob-X" : 0.10
      } ]
    }, {
      "turn-index" : 4,
      "labels" : [ {
        "breakdown" : "X",
        "prob-O" : 0.0,
        "prob-T" : 0.0,
        "prob-X" : 1.0
      } ]
    }, {
           ・・・
    }, {
      "turn-index" : 18,
      "labels" : [ {
        "breakdown" : "T",
        "prob-O" : 0.2,
        "prob-T" : 0.6,
        "prob-X" : 0.2
      } ]
    }, {
      "turn-index" : 20,
      "labels" : [ {
        "breakdown" : "O",
        "prob-O" : 1.0,
        "prob-T" : 0.0,
        "prob-X" : 0.0
      } ]
    } ]
  }

  ##########
  ~~~~~~~

### Description of each field

  - dialogue-id ： This ID should match that of the input dialogue file.  
  - turns ： This field contains all the dialogue breakdown detection results for all turns.  
  - turn-index ： This field denotes the index of a turn and should match that of the input dialogue file.  
  - labels ： This field contains the dialogue breakdown detection results for a turn.  
  - breakdown ： This can be one of O or T or X, corresponding to not a breakdown, possible breakdown, and breakdown, corresponding to the hard-decision made by a dialogue breakdown detector.  
  - prob-O ： Probability that the dialogue breakdown detection result is O (not a breakdown).  
  - prob-T ： Probability that the dialogue breakdown detection result is T (possible breakdown).  
  - prob-X ： Probability that the dialogue breakdown detection result is X (breakdown).  
