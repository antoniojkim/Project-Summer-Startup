/*
* To change this license header, choose License Headers in Project Properties.
* To change this template file, choose Tools | Templates
* and open the template in the editor.
*/
import static java.lang.Double.NaN;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**

@author Antonio
*/
public class NeuralNetwork{
    
    public static void main (String[] args){
        NeuralNetwork nn = new NeuralNetwork(3, 5, 7, 5, 1);
        double[][] trainingData =    {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 0, 0}};
        double[][] trainingOutput =  {{0},       {1},       {1},       {1},       {1},       {0},       {0}};
        print(nn.get(trainingData));
        for (int i = 0; i<=1000; i++){
            double loss = nn.train(trainingData, trainingOutput);
            if (i%100 == 0){
                System.out.println("Epoch:  "+i+"       Loss:  "+loss);
            }
        }
        System.out.println("");
        print(transpose(nn.get(trainingData)));
    }
    
    private double reg = 1;       // Regularization Strength
    private double eta = 0.5;        // Learning Rate
    
    private int numLayers;
    private List<Integer> numNeurons;
    private List<double[][]> weights;
    private List<double[]> biases;
    
    private boolean errorProof = false;
    
    public NeuralNetwork(Integer... sizes){
        
        numNeurons = new ArrayList<>(Arrays.asList(sizes));
        numLayers = numNeurons.size();
        
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        
        for (int i = 1; i<numNeurons.size(); i++){
            weights.add(createWeights(numNeurons.get(i-1), numNeurons.get(i)));
            double[] temp = new double[numNeurons.get(i)];
            biases.add(temp);
        }
    }
    
    public double[][] createWeights(int x, int y){
        double[][] weights = new double[x][y];
        
        double weightRegulator = Math.sqrt(2.0/y);
        
        for (int i = 0; i<weights.length; i++){
            for (int j = 0; j<weights[i].length; j++){
                weights[i][j] = (2*Math.random()-1)*weightRegulator;
            }
        }
        //print(weights);
        
        return weights;
    }
    
    public double[][] get(double... input){
        double[][] input2d = {input};
        return feedForward(transpose(input2d));
    }
    public double[][] get(double[][] input){
        return feedForward(input);
    }
    public double[][] get(int[][] input){
        return feedForward(toDoubleArray(input));
    }
    public double[][] feedForward(double[][] input){                            // Input must be (M, D) for some M
        double[][] a = input;
        for (int i = 0; i<weights.size()-1; i++){
            a = activation(add(dot(a, weights.get(i)), biases.get(i)));
        }
        a = output_activation(add(dot(a, weights.get(weights.size()-1)), biases.get(biases.size()-1)));
        return a;
    }
    
    public static double[][] activation(double[][] z){
        //Tanh
        double[][] a = new double[z.length][z[0].length];
        for (int i = 0; i<a.length; i++){
            for (int j = 0; j<a[i].length; j++){
                a[i][j] = Math.tanh(z[i][j]);
            }
        }
        return a;
    }
    private double[][] activation_prime(double[][] z){
        //Tanh Prime
        double[][] a = new double[z.length][z[0].length];
        for (int i = 0; i<a.length; i++){
            for (int j = 0; j<a[i].length; j++){
                a[i][j] = 1/Math.pow(Math.cosh(z[i][j]), 2);
            }
        }
        return a;
    }
    public static double[][] output_activation(double[][] z){
        //Tanh
        double[][] a = new double[z.length][z[0].length];
        for (int i = 0; i<a.length; i++){
            for (int j = 0; j<a[i].length; j++){
                a[i][j] = Math.tanh(z[i][j]);
            }
        }
        return a;
    }
    private double[][] output_activation_prime(double[][] z){
        //Tanh Prime
        double[][] a = new double[z.length][z[0].length];
        for (int i = 0; i<a.length; i++){
            for (int j = 0; j<a[i].length; j++){
                a[i][j] = 1/Math.pow(Math.cosh(z[i][j]), 2);
            }
        }
        return a;
    }
    
    public double dot(double[] x, double[] y){
        if (x.length == y.length){
            double dot = 0;
            for (int i = 0; i<x.length; i++){
                dot += x[i]*y[i];
            }
            return dot;
        }
        return NaN;
    }
    public double[][] dot(double[][] x, double[][] y){
		if (x[0].length != y.length){
			System.out.println("Dot Error 2:  "+x[0].length+", "+y.length);
			System.exit(1);
		}
		y = transpose(y);
		double[][] dot = new double[x.length][y.length];
		for (int i = 0; i<dot.length; i++){
			for (int j = 0; j<dot[i].length; j++){
				dot[i][j] = dot(x[i], y[j]);
			}
		}

		return dot;
    }
    public double[][] add(double[][] z, double[] b){
        if (z.length == b.length){
            for (int i = 0; i<z.length; i++){
                for (int j = 0; j<z[i].length; j++){
                    z[i][j] += b[i];
                }
            }
        }
        return z;
    }
    
    
    public double train(int[][] trainingInput, double[][] expectedOutput){   // Gradient Descent
        return train(toDoubleArray(trainingInput), expectedOutput);
    }
    public double train(double[][] trainingInput, double[][] expectedOutput){   // Gradient Descent
        if (trainingInput[0].length == numNeurons.get(0)
                && expectedOutput[0].length == numNeurons.get(numLayers-1)){
            int M = trainingInput.length;                                    // Input:  (M, D)
            // Output: (M, K)
            
            List<double[][]> integrations = new ArrayList<>();
            List<double[][]> activations = new ArrayList<>();
            List<double[][]> activations_prime = new ArrayList<>();
            
            activations.add(trainingInput);
            for (int i = 0; i<(weights.size()-1); i++){
                integrations.add(add(dot(activations.get(activations.size()-1), weights.get(i)), biases.get(i)));
                activations.add(activation(integrations.get(integrations.size()-1)));
                activations_prime.add(activation_prime(integrations.get(integrations.size()-1)));
            }
            integrations.add(add(dot(activations.get(activations.size()-1), weights.get(weights.size()-1)), biases.get(biases.size()-1)));
            activations.add(output_activation(integrations.get(integrations.size()-1)));
            activations_prime.add(output_activation_prime(integrations.get(integrations.size()-1)));
            
            double[][] output = activations.get(activations.size()-1);
            
            double loss = cost(expectedOutput, output);
            if (loss == NaN){
                System.out.println("Invalid Loss");
                System.exit(1);
            }
            
            double[][] delta = multiply(costPrime(expectedOutput, output), activations_prime.get(activations_prime.size()-1));
            List<double[][]> dWeights = new ArrayList<>();
            
            dWeights.add(dot(transpose(activations.get(activations.size()-2)), delta));
            
            for (int i = 2; i<numLayers; i++){
                delta = multiply(dot(delta, transpose(weights.get(weights.size()-i+1))), activations_prime.get(activations_prime.size()-i));
                dWeights.add(dot(transpose(activations.get(activations.size()-i-1)), delta));
            }
            
            dWeights = reverse(dWeights);
            
            for (int i = 0; i<weights.size(); i++){
                weights.set(i, parameterUpdate(weights.get(i), dWeights.get(i)));
            }
            
            return loss;
        }
        else{
            System.out.println("Invalid Training Input or Expected Output");
            System.out.println("Training Input:  "
                    + "("+trainingInput[0].length+", "+trainingInput.length+")");
            System.out.println("Expected Output:  "
                    + "("+expectedOutput[0].length+", "+expectedOutput.length+")");
        }
        return NaN;
    }
    public double cost(double[][] expected, double[][] output){
        // Mean Squared Cost
        double loss = 0;
        if (expected.length != output.length){
            return NaN;
        }
        for (int i = 0; i<expected.length; i++){
            if (expected[i].length != output[i].length){
                return NaN;
            }
            for (int j = 0; j<expected[i].length; j++){
                loss += Math.pow(output[i][j]-expected[i][j], 2);
            }
        }
        return 0.5*loss;
    }
    public double[][] costPrime(double[][] expected, double[][] output){
        // Derivative of Mean Squared Cost
        if (expected.length != output.length){
            System.out.println("Invalid Cost Prime");
            System.exit(1);
        }
        double[][] costPrime = new double[expected.length][];
        for (int i = 0; i<expected.length; i++){
            if (expected[i].length != output[i].length){
                System.out.println("Invalid Cost Prime");
                System.exit(1);
            }
            double[] v = new double[expected[i].length];
            for (int j = 0; j<v.length; j++){
                v[j] = (output[i][j]-expected[i][j]);
            }
            costPrime[i] = v;
        }
        return costPrime;
    }
    
    public double regularize(double[][] weights){
        double regularization = 0;
        for (int i = 0; i<weights.length; i++){
            for (int j = 0; j<weights[i].length; j++){
                regularization += 0.5*reg*weights[i][j]*weights[i][j];
            }
        }
        return regularization;
    }
    public static double[][] multiply(double[][] x, double[][] y){
        if (x.length == y.length && x[0].length == y[0].length){
            double[][] m = new double[x.length][x[0].length];
            for (int i = 0; i<m.length; i++){
                for (int j = 0; j<m[i].length; j++){
                    m[i][j] = x[i][j]*y[i][j];
                }
            }
            return m;
        }
        return null;
    }
    public static double[][] transpose(double[][] x){
        double[][] t = new double[x[0].length][x.length];
        for (int i = 0; i<t.length; i++){
            for (int j = 0; j<t[i].length; j++){
                t[i][j] = x[j][i];
            }
        }
        return t;
    }
    public List<double[][]> reverse(List<double[][]> list){
        for (int i = 0; i<(list.size()-i-1); i++){
            double[][] temp = list.get(i);
            list.set(i, list.get(list.size()-i-1));
            list.set(list.size()-i-1, temp);
        }
        return list;
    }
    
    public double[][] parameterUpdate(double[][] w, double[][] dW){
        for (int i = 0; i<dW.length; i++){
            for (int j = 0; j<dW[i].length; j++){
                w[i][j] -= eta*dW[i][j];
            }
        }
        return w;
    }
    public double[] parameterUpdate(double[] b, double[] dB){
        for (int i = 0; i<dB.length; i++){
            b[i] -= eta*dB[i];
        }
        return b;
    }
    
    public void setLearningRate(double eta){
        this.eta = eta;
    }
    
    public void checkError(String var, double[][] a, double x, double y){
        if (a.length != x || a[0].length != y){
            System.out.println(var+":  "+a[0].length+", "+a.length+
                    "\nShould be:     "+x+", "+y);
            System.exit(1);
        }
    }
    public static double[][] toDoubleArray(int[][] array){
        double[][] newArray = new double[array.length][];
        for (int i = 0; i<newArray.length; i++){
            double[] temp = new double[array[i].length];
            for (int j = 0; j<temp.length; j++){
                temp[j] = array[i][j];
            }
            newArray[i] = temp;
        }
        return newArray;
    }
    
    public static void print(double[] a){
        System.out.print("[");
        for (int i = 0; i<a.length; i++){
            if (i != 0){
                System.out.print(", ");
            }
            System.out.print(a[i]);
        }
        System.out.println("]\n");
    }
    public static void print(double[][] a){
        System.out.print("[");
        for (int i = 0; i<a.length; i++){
            if (i != 0){
                System.out.print("\n [");
            }
            else{
                System.out.print("[");
            }
            for (int j = 0; j<a[i].length; j++){
                if (j != 0){
                    System.out.print(", ");
                }
                System.out.print(a[i][j]);
            }
            System.out.print("]");
        }
        System.out.println("]\n");
    }
    
}
