package services;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class EmojifyService {
    static MultiLayerNetwork model = null;

    public EmojifyService() throws Exception {
        this.init();
    }

    public void init() throws Exception {
        Utils.init();
        model = ModelSerializer.restoreMultiLayerNetwork(new ClassPathResource("data/emojify-model.zip").getFile().getPath());
    }

    public String suggestedEmoji(String text) {

        List<String> X_test2 = new ArrayList<String>();
        X_test2.add(text);

        INDArray X_test_indices = Utils.sentencesToIndices(X_test2, 5);
        INDArray pred4 = model.output(X_test_indices);
        int num = Nd4j.argMax(pred4.getRow(0).getColumn(4), 0).getInt(0, 0);
        return Utils.emoji(num);

    }

    public static void main(String[] args) throws Exception {
        EmojifyService es = new EmojifyService();
        String text = "great weather outside";
        String emoji = es.suggestedEmoji(text);
        System.out.println(" Test: " + text + " " + emoji);

        es.testModel();
    }

    public void testModel() throws Exception {
        // read test data
        List<String> X_test = new ArrayList<String>();
        List<Integer> Y_test = new ArrayList<Integer>();
        Utils.readCsv(new ClassPathResource("data/tesss.csv").getFile().getPath(), X_test, Y_test);

        INDArray X_test_indices = Utils.sentencesToIndices(X_test, 5);
        INDArray Y_test_oh = Utils.convertToOneHot(Y_test, 5, 5);

        // This code allows you to see the mislabeled examples
        INDArray pred3 = model.output(X_test_indices);

        double miss = 0;
        for (int i = 0; i < X_test.size(); i++) {
            int num = Nd4j.argMax(pred3.getRow(i).getColumn(4), 0).getInt(0,0);
            if(num != Y_test.get(i)) {
                miss = miss + 1;
            }

            System.out.println("Expected emoji: " + Utils.emoji(Y_test.get(i)) +
                    " prediction: " + X_test.get(i) + Utils.emoji(num));

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
