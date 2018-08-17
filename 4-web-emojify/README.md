# Web Emojify
Web interface to Emojify.
It accepts a sentance and pdericts a matching emoji.

To use this code,
  -  run the emojify java app first to generate a trained model file /tmp/emojify-model.zip.
  - copy that file and the Glove vector file glove.6B.50d.txt extracted from http://nlp.stanford.edu/data/wordvecs/glove.6B.zip to conf/data folder.
  - sbt run
  - open [http://localhost:900](0http://localhost:9000)
