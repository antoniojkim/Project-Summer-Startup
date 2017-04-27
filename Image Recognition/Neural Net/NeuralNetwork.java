
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**

@author Antonio
*/
public class NeuralNetwork{
    
    private int num_layers;
    private List<Integer> layer_sizes;
    private List<double[]> weights;
    private List<double[]> biases;
    
    public NeuralNetwork(List<Integer> sizes){
        num_layers = sizes.size();
        layer_sizes = sizes;
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        for (int a = 0; a<num_layers; a++){
            double[] layer_weights = new double[sizes.get(a)];
            double[] layer_biases = new double[sizes.get(a)];
            if (a != 0){
                for (int b = 0; b<sizes.get(a); b++){
                    layer_weights[b] = Math.random();
                    layer_biases[b] = Math.random();
                }
            }
            weights.add(layer_weights);
            biases.add(layer_biases);
        }
    }
    
    private double[] get(double[] a){
        return feedForward(a);
    }
    private double[] feedForward(double[] a){
        for (int i = 1; i<weights.size(); i++){
            a = activation(weights.get(i), a, biases.get(i));
        }
        return a;
    }
    
    public double[] activation(double[] z){
        double[] a = new double[z.length];
        for (int i = 0; i<a.length; i++){
            a[i] = 1/(1+Math.exp(-z[i]));
        }
        return a;
    }
    public double[] zip (double[] w, double[] z, double[] b){
        if (w.length != b.length){
            System.out.println("Error:  Failed to zip  => w.length == "+w.length+", b.length == "+b.length);
            System.exit(1);
        }
        double[] zipped = new double[w.length];
        for (int i = 0; i<w.length; i++){
            for (int j = 0; j<z.length; j++){
                zipped[i] = w[i]*z[j]+b[i];
            }
        }
        return zipped;
    }
    public double[] activation(double[] w, double[] z, double[] b){
        if (z.length == w.length && z.length == b.length){
            return activation(zip(w, z, b));
        }
        if (w.length != b.length){
            System.out.println("Error:  Failed to activate  => w.length == "+w.length+", b.length == "+b.length);
            System.exit(1);
        }
        double[] a = new double[w.length];
        for (int i = 0; i<a.length; i++){
            for (int j = 0; j<z.length; j++){
                a[i] += activation(w[i], z[j], b[i]);
            }
        }
        return a;
    }
    public double activation(double w, double z, double b){
        //Sigmoid/Logistic Function
        return 1/(1+Math.exp(-1*w*z-b));
    }
    public double[] activation_derivative(double[] z){
        //Sigmoid/Logistic Function
        double[] activation = activation(z);
        for (int i = 0; i<activation.length; i++){
            activation[i] *= (1-activation[i]);
        }
        return activation;
    }
    public double activation_derivative(double w, double z, double b){
        //Sigmoid/Logistic Function
        double activation = activation(w, z, b);
        return activation*(1-activation);
    }
    
    // Stochastic Gradient Descent
    public void train(List<TrainingData> training_data, int epochs, int mini_batch_size, double eta){
        int n = training_data.size();
        for (int j = 0; j<epochs; j++){
            shuffle(training_data);
            List<List<TrainingData>> mini_batches = new ArrayList<>();
            for (int k = 0; (k+mini_batch_size)<n; k += mini_batch_size){
                mini_batches.add(new ArrayList<>(training_data.subList(k, k+mini_batch_size)));
            }
            for (List<TrainingData> mini_batch : mini_batches){
                update_mini_batch(mini_batch, eta);
            }
        }
    }
    
    private void update_mini_batch(List<TrainingData> mini_batch, double eta){
        List<double[]> nabla_b = new ArrayList<>();
        List<double[]> nabla_w = new ArrayList<>();
        for (int i = 0; i<weights.size(); i++){
            double[] zeroes_b = new double[biases.get(i).length];
            nabla_b.add(zeroes_b);
            double[] zeroes_w = new double[weights.get(i).length];
            nabla_w.add(zeroes_w);
        }
        for (int i = 0; i<mini_batch.size(); i++){
            List<List<double[]>> delta_nabla = backpropagate(mini_batch.get(i));
            for (int j = 0; j<nabla_b.size(); j++){
                nabla_b.set(j, add(nabla_b.get(j), delta_nabla.get(0).get(j)));
                nabla_w.set(j, add(nabla_w.get(j), delta_nabla.get(1).get(j)));
            }
        }
        double factor = eta/mini_batch.size();
        for (int i = 0; i<weights.size(); i++){
            weights.set(i, subtract(weights.get(i), dot(factor, nabla_w.get(i))));
            biases.set(i, subtract(biases.get(i), dot(factor, nabla_b.get(i))));
        }
    }
    
    private List<List<double[]>> backpropagate(TrainingData data){
        List<double[]> nabla_b = new ArrayList<>();
        List<double[]> nabla_w = new ArrayList<>();
        for (int i = 0; i<weights.size(); i++){
            double[] zeroes_b = new double[biases.get(i).length];
            nabla_b.add(zeroes_b);
            double[] zeroes_w = new double[weights.get(i).length];
            nabla_w.add(zeroes_w);
        }
        double[] activation = data.input;
        List<double[]> activations = new ArrayList<>(Arrays.asList(data.input));
        List<double[]> zs = new ArrayList<>();
        for (int i = 0; i<weights.size(); i++){
            double[] z = zip(weights.get(i), activation, biases.get(i));
            zs.add(z);
            activation = activation(z);
            activations.add(activation);
        }
        double[] delta = dot(cost_derivative(activations.get(activations.size()-1), data.expectedOutput), activation_derivative(zs.get(zs.size()-1)));
        nabla_b.set(nabla_b.size()-1, delta);
        delta = dot(activations.get(activations.size()-2), delta);
        nabla_w.set(nabla_w.size()-1, delta);
        for (int l = 2; l<num_layers; l++){
            double[] z = zs.get(zs.size()-l);
            double[] sp = activation_derivative(z);
            delta = dot(weights.get(weights.size()-l+1), delta);
            
            delta = dot(delta, sp);
            nabla_b.set(nabla_b.size()-l, delta);
            delta = dot(activations.get(activations.size()-l-1), delta);
            nabla_w.set(nabla_w.size()-l, delta);
        }
        return Arrays.asList(nabla_b, nabla_w);
    }
    private double[] cost_derivative(double[] output_activations, double[] expected){
        try{
            for (int a = 0; a<output_activations.length; a++){
                output_activations[a] -= expected[a];
            }
        }catch(ArrayIndexOutOfBoundsException e){}
        return output_activations;
    }
    
    private double[] add (double[] x, double[] y){
        if (x.length != y.length){
            System.out.println("Error:  Failed to add => x.length == "+x.length+", y.length == "+y.length);
        }
        double[] a = new double[x.length];
        for (int i = 0; i<a.length; i++){
            a[i] = x[i]+y[i];
        }
        return a;
    }
    private double[] subtract (double[] x, double[] y){
        double[] a = new double[x.length];
        for (int i = 0; i<a.length; i++){
            a[i] = x[i]-y[i];
        }
        return a;
    }
    private double[] dot (double[] x, double[] y){
        double[] a = new double[y.length];
        for (int i = 0; i<a.length; i++){
            for (int j = 0; j<x.length; j++){
                a[i] += x[j]*y[i];
            }
        }
        return a;
    }
    private double[] dot (double x, double[] y){
        double[] a = new double[y.length];
        for (int i = 0; i<a.length; i++){
            a[i] = x*y[i];
        }
        return a;
    }
    
    private void shuffle(List<TrainingData> training_data){
        List<TrainingData> shuffled = new ArrayList<>();
        while(!training_data.isEmpty()){
            int r = randomint(0, training_data.size()-1);
            shuffled.add(training_data.get(r));
            training_data.remove(r);
        }
        training_data.addAll(shuffled);
        shuffled.clear();
    }
    public static double round(double num, int place){
        double rounded = Math.round(num*Math.pow(10, place))/Math.pow(10, place);
        return rounded;
    }
    private int randomint(int low, int high){
        return (int)((high-low+1)*Math.random()+low);
    }
    
    public void printWeights(){
        System.out.println("Weights:");
        for (int a = 0; a<weights.size(); a++){
            System.out.print("          ");
            for (int b = 0; b<weights.get(a).length; b++){
                System.out.print(round(weights.get(a)[b], 3)+"     ");
            }
            System.out.println("");
        }
    }
    public void printBiases(){
        System.out.println("Biases:");
        for (int a = 0; a<biases.size(); a++){
            System.out.print("          ");
            for (int b = 0; b<biases.get(a).length; b++){
                System.out.print(round(biases.get(a)[b], 3)+"     ");
            }
            System.out.println("");
        }
    }
    
    public List<TrainingData> createTrainingData(List<double[]> input, List<double[]> expectedOutput){
        List<TrainingData> data = new ArrayList<>();
        if (input.size() == expectedOutput.size()){
            int expectedExpectedSize = layer_sizes.get(layer_sizes.size()-1);
            for (int a = 0; a<expectedOutput.size(); a++){
                if (expectedOutput.get(a).length == expectedExpectedSize){
                    data.add(new TrainingData(input.get(a), expectedOutput.get(a)));
                }
            }
        }
        return data;
    }
    
    private class TrainingData{
        double[] input;
        double[] expectedOutput;
        
        public TrainingData(double[] input, double[] expectedOutput) {
            this.input = input;
            this.expectedOutput = expectedOutput;
        }
        
    }
    
}
