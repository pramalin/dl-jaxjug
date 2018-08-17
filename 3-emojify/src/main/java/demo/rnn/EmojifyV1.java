package demo.rnn;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class EmojifyV1 {

	static int Series_Length = 5;

	public static void main(String[] args) throws Exception {

		Utils.init();

		// model
		MultiLayerNetwork model = Utils.Emojify_V2();
/*
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;
        model.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
*/
        // read train data
		List<String> X_train = new ArrayList<String>();
		List<Integer> Y_train = new ArrayList<Integer>();
		Utils.readCsv(new ClassPathResource("data/train_emoji.csv").getFile().getPath(), X_train, Y_train);
		
		INDArray X_train_indices = Utils.sentencesToIndices(X_train, Series_Length);
		INDArray Y_train_oh = Utils.convertToOneHot(Y_train, 5, Series_Length);

		// many to one output mask - 00001
		INDArray labelsMask = Nd4j.zeros(Y_train.size(), 5);
		INDArray lastColumnMask = Nd4j.ones(Y_train.size(), 1);
		labelsMask.putColumn(4, lastColumnMask);

		// read test data
		List<String> X_test = new ArrayList<String>();
		List<Integer> Y_test = new ArrayList<Integer>();
		Utils.readCsv(new ClassPathResource("data/tesss.csv").getFile().getPath(), X_test, Y_test);
		
		INDArray X_test_indices = Utils.sentencesToIndices(X_test, Series_Length);
		INDArray Y_test_oh = Utils.convertToOneHot(Y_test, 5, Series_Length);

		ListDataSetIterator<DataSet> trainData =
				new ListDataSetIterator<DataSet>((new DataSet(X_train_indices, Y_train_oh, null, labelsMask)).asList(), 32);
		ListDataSetIterator<DataSet> testData =
				new ListDataSetIterator<DataSet>(new DataSet(X_test_indices, Y_test_oh).asList());

		// 50 epochs
		String str = "Test set evaluation at epoch %d: Score: %.4f Accuracy = %.2f, F1 = %.2f";

		for (int i = 0; i < 70; i++) {
			model.fit(trainData);
            
			//Evaluate on the test set:
            Evaluation evaluation = model.evaluate(testData);
            System.out.println(String.format(str, i, model.score(), evaluation.accuracy(), evaluation.f1()));
		}

		// save model
		File location = new File("/tmp/emojify-model.zip");
		ModelSerializer.writeModel(model, location, true);

		// This code allows you to see the mislabeled examples
		INDArray pred3 = model.output(X_test_indices);

		double miss = 0;
		for (int i = 0; i < X_test.size(); i++) {
		    int num = Nd4j.argMax(pred3.getRow(i).getColumn(4), 0).getInt(0,0);
		    if(num != Y_test.get(i)) {
		      miss = miss + 1;
		      System.out.println("Expected emoji: " + Utils.emoji(Y_test.get(i)) +
		    		  " prediction: " + X_test.get(i) + Utils.emoji(num));
		    }
		  }
		  
		System.out.println("Missed " + miss + " out of " + X_test.size() +
				" acc: " + ((X_test.size() - miss) / X_test.size()) * 100 + " percent"); 
		  
		// Change the sentence below to see your prediction. Make sure all the words are
		// in the Glove embeddings.
		List<String> X_test2 = new ArrayList<String>();
		X_test2.add("not feeling happy");

		X_test_indices = Utils.sentencesToIndices(X_test2, 5);
		INDArray pred4 = model.output(X_test_indices);
		int num4 = Nd4j.argMax(pred4.getRow(0).getColumn(4), 0).getInt(0, 0);

		System.out.println(" Test: " + X_test2.get(0) + Utils.emoji(num4));
	}

}
