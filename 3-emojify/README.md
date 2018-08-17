## Emoji ##

This is an attempt to create DL4J version of Emoji LSTM example from Coursera DL specialization course.

The model is simplified to trouble shoot shape mismatch errors. The problem was fixed after following suggestion from DL4J gitter channel.

Need to copy the Glove vector file glove.6B.50d.txt extracted from http://nlp.stanford.edu/data/wordvecs/glove.6B.zip to src/main/resources/data folder.


### Goal ###
  Given variable sized sentences (max 5) like the following
	"funny lol"
	"lets play baseball"
	"food is ready for you"
   predict the emoji identifier (0 to 4).

      +-----------+
 -->  | Embedding |  // maps word index to - Glove vector 
      +-----------+
            |
            V   
      +-----------+
      |   LSTM    |  // (5)
      +-----------+
            |
            V   
      +-----------+
      | RNN Output|  // many to one softmax
      +-----------+

### Specs: ###
	Use an Embedding layer initialized with prebuilt Glove.
	Time series Tx = 5
	train samples m = 132
	max # of features n_x = 5
	label shape = one hot vector (1,5)



To create the model
  - copy glove vector file "glove.6B.50d.txt" (the file distributed in the class worked the best) to src/main/resources/data
  You may try the file bundled [here](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)
  - open project in an IDE
  - run demo.rnn.Emojify
  - open [http://localhost:9000](http://localhost:9000) to view training progress.

This creates the trained model /tmp/emojify-model.zip. Kill the process to stop the monitor.
