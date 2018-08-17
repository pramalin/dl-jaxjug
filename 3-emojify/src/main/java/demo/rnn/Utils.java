package demo.rnn;

import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Utils {

    static Map<String, Integer> word_to_index = new HashMap<String, Integer>();
    static Map<Integer, String> index_to_word = new HashMap<Integer, String>();
    static Map<String, double[]> word_to_vec_map = new HashMap<String, double[]>();

    static Map<Integer, String> emoji_dictionary = new HashMap<Integer, String>();

    public static void init() throws Exception {
        // initialize emoji dictionary
        emoji_dictionary.put(0, "\u2764\uFE0F");    // :heart: prints a black instead of red heart depending on the font
        emoji_dictionary.put(1, ":baseball:");
        emoji_dictionary.put(2, ":smile:");
        emoji_dictionary.put(3, ":disappointed:");
        emoji_dictionary.put(4, ":fork_and_knife:");

        System.out.println("Read glove file ...");
        // readGloveVecs(new ClassPathResource("data/vectors.txt").getFile().getPath());
        readGloveVecs(new ClassPathResource("data/glove.6B.50d.txt").getFile().getPath());

    }

    public static String emoji(int number) {
        return emoji_dictionary.get(number);
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
            word_to_vec_map.put(word, vecs);

            word_to_index.put(word, index);
            index_to_word.put(index, word);
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


    public static INDArray sentencesToAvg(List<String> X) {
        int m = X.size(); // number of training examples
        int vecSize = word_to_vec_map.get("cucumber").length;
        INDArray Xavg = Nd4j.zeros(m, vecSize);

        for (int i = 0; i < m; i++) { // loop over training examples
            // Convert the ith training sentence in lower case and split is into words. You
            // should get a list of words.
            String[] sentenceWords = X.get(i).toLowerCase().split(" ");

            // # Loop over the words of sentence_words
            int j = 0;
            // Initialize the average word vector, should have the same shape as your word vectors.
            INDArray avg = Nd4j.zeros(vecSize);

            for (String w : sentenceWords) {
                // Set the (i,j)th entry of X_indices to the index of the correct word.
                if (word_to_index.containsKey(w)) {
                    double[] vec = word_to_vec_map.get(w);
                    INDArray newArray = Nd4j.create(vec);
                    avg.addi (newArray);
                    j = j + 1;
                }
            }
            Xavg.putRow(i, avg.divi(j));
        }

        return Xavg;
    }

    public static INDArray convertToOneHot(List<Integer>Y, int C, int seriesLen) {
        INDArray labels = Nd4j.create(new int[]{Y.size(), C, seriesLen}, 'f');

        for (int i = 0; i < Y.size(); i++) {
            labels.putScalar(new int[]{i, Y.get(i), seriesLen - 1}, 1.0);
        }
        return labels;
    }


    public static INDArray convertToOneHot2d(List<Integer>Y, int C) {
        INDArray labels = Nd4j.create(new int[]{Y.size(), C});

        for (int i = 0; i < Y.size(); i++) {
            labels.putScalar(new int[]{i, Y.get(i)}, 1.0);
        }
        return labels;
    }

    public static MultiLayerNetwork Emojify_V2() {
        int vocab_len = index_to_word.keySet().size(); //

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(0)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.2))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(vocab_len).nOut(50).activation(Activation.IDENTITY).build())
                .layer(1, new LSTM.Builder().nIn(50).nOut(32).activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder().nIn(32).nOut(5).activation(Activation.SOFTMAX)

                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .setInputType(InputType.recurrent(1))
                .pretrain(false)
                .backprop(true)
                .build();

        System.out.println(conf.toJson());
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer embeddingLayer =
                (org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer) net.getLayer(0);

        // Retrieving the weights
        INDArray weights = embeddingLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);

        // putting pre-trained weights into rows
        INDArray rows = Nd4j.createUninitialized(new int[] { vocab_len, weights.size(1) }, 'c');

        for (int i = 0; i < vocab_len; i++) {
            String word = index_to_word.get(i);

            double[] embeddings = word_to_vec_map.get(word);
            if (embeddings != null) {
                INDArray newArray = Nd4j.create(embeddings);
                rows.putRow(i, newArray);
            } else { // if there is no pre-trained embedding value for that specific entry
                rows.putRow(i, weights.getRow(i));
            }
        }

        // finally put rows in place of weights
        embeddingLayer.setParam("W", rows);

        return net;
    }

    public static MultiLayerNetwork Emojify_V1() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(0)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.04))
                .l2(1e-5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(50).nOut(1)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(1).nOut(5).build())
                .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }

}
