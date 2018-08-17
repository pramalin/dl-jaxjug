package services;

import com.vdurmont.emoji.EmojiParser;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Utils {

    static Map<String, Integer> word_to_index = new HashMap<String, Integer>();
    static Map<Integer, String> emoji_dictionary = new HashMap<Integer, String>();

    public static void init() throws Exception {
        // initialize emoji dictionary
        emoji_dictionary.put(0, "\u2764\uFE0F");    // :heart: prints a black instead of red heart depending on the font
        emoji_dictionary.put(1, ":baseball:");
        emoji_dictionary.put(2, ":smile:");
        emoji_dictionary.put(3, ":disappointed:");
        emoji_dictionary.put(4, ":fork_and_knife:");

        System.out.println("Read glove file ...");
        readGloveVecs(new ClassPathResource("data/words.txt").getFile().getPath());
    }

    public static String emoji(int number) {
        return EmojiParser.parseToUnicode(emoji_dictionary.get(number));
    }

    static void readGloveVecs(String filename) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(filename));

        int index = -1;
        for (String line : lines) {
            index = index + 1;
            String[] values = line.toLowerCase().split(" ");

            String word = values[0];
            double[] vecs = new double[values.length - 1];

            for (int i = 0; i < vecs.length; i++) {
                vecs[i] = Double.parseDouble(values[i + 1]);
            }
            word_to_index.put(word, index);
        }
    }

    static void readCsv(String filename, List<String> X, List<Integer> Y) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(filename));

        for (String line : lines) {
            String[] tokens = line.split(",");
            X.add(tokens[0].replace("\"","").trim());
            Y.add(Integer.valueOf(tokens[1]));
        }
    }

    public static INDArray sentencesToIndices(List<String> X, int maxLen) {
        int m = X.size(); // number of training examples

        // # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1
        // line)
        INDArray XIndices = Nd4j.zeros(m, 1, maxLen);

        for (int i = 0; i < m; i++) { // loop over training examples
            // Convert the ith training sentence in lower case and split is into words. You
            // should get a list of words.
            String[] sentenceWords = X.get(i).toLowerCase().split(" ");

            // Initialize j to 0
            int j = 0;

            // # Loop over the words of sentence_words
            for (String w : sentenceWords) {
                // Set the (i,j)th entry of X_indices to the index of the correct word.
                if (j < maxLen) {
                    if (word_to_index.containsKey(w)) {
                        XIndices.putScalar(new int[] { i, 0, j }, word_to_index.get(w));
                    }
                    // Increment j to j + 1
                    j = j + 1;
                }
            }

        }
        return XIndices;
    }

    public static INDArray convertToOneHot(List<Integer>Y, int C, int seriesLen) {
        INDArray labels = Nd4j.create(new int[]{Y.size(), C, seriesLen}, 'f');

        for (int i = 0; i < Y.size(); i++) {
            labels.putScalar(new int[]{i, Y.get(i), seriesLen - 1}, 1.0);
        }
        return labels;
    }

}
