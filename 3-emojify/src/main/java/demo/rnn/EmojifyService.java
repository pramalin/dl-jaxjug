package demo.rnn;


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class EmojifyService {
    static MultiLayerNetwork model = null;

    public static void init() throws Exception {
        Utils.init();
        model = ModelSerializer.restoreMultiLayerNetwork("/tmp/emojify-model.zip");
    }

    public static String suggestedEmoji(String text) {

        List<String> X_test2 = new ArrayList<String>();
        X_test2.add(text);

        INDArray X_test_indices = Utils.sentencesToIndices(X_test2, 5);
        INDArray pred4 = model.output(X_test_indices);
        int num = Nd4j.argMax(pred4.getRow(0).getColumn(4), 0).getInt(0, 0);
        return Utils.emoji(num);

    }

    public static void main(String[] args) throws Exception {
        EmojifyService.init();
        String text = "i am annoyed";
        String emoji = EmojifyService.suggestedEmoji(text);
        System.out.println(" Test: " + text + " " + emoji);
    }
}
