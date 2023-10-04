package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.random.Uniform;

import java.io.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import static com.psi.ActivationFunction.relu;
import static com.psi.ActivationFunction.reluDerivative;
import static com.psi.U.multiplyElements;

public class ArtificialIntelligence {
	private final static double defaultAlpha = 0.01;
	private final static double chance = 0.5;
	List<MatrixStore<Double>> weights;
	int w = 0;

	public ArtificialIntelligence() {
		weights = new LinkedList<>();
	}

	public ArtificialIntelligence(int rows, int columns, Double weightMinVal, Double range) {
		weights = new LinkedList<>();
		MatrixStore<Double> temp = Primitive64Store.FACTORY.makeFilled(rows, columns, Uniform.of(weightMinVal, range));
		weights.add(temp);
		w++;
	}

	public void addLayer(int n, Double weightMinVal, Double range) {
		MatrixStore<Double> prev = weights.get(w - 1);
		Primitive64Store temp = Primitive64Store.FACTORY.makeFilled(n, prev.countRows(), Uniform.of(weightMinVal, range));
		weights.add(temp);
		w++;
	}

	public MatrixStore<Double> predict(MatrixStore<Double> input) {
		return deepNeuralNetwork(input, true, null, defaultAlpha, false);
	}

	public void fit(MatrixStore<Double> input, MatrixStore<Double> expected, double alpha, boolean dropout) {
		deepNeuralNetwork(input, true, expected, alpha, dropout);
	}

	public static Double neuron(MatrixStore<Double> input, MatrixStore<Double> weights, Double bias) {
		int n = (int) input.countColumns();
		double res = 0;
		for (int i = 0; i < n; ++i) {
			res += input.get(i) * weights.get(i) + bias;
		}
		return res;
	}

	public static MatrixStore<Double> neuralNetwork(MatrixStore<Double> input, MatrixStore<Double> weight) {
		Primitive64Store res = Primitive64Store.FACTORY.make(1, weight.countRows());
		for (int i = 0; i < weight.countRows(); ++i) {
			res.set(0, i, neuron(input, weight.row(i), (double) 0));
		}
		return res;
	}

	public MatrixStore<Double> deepNeuralNetwork(MatrixStore<Double> input, boolean useRelu, MatrixStore<Double> expected, double alpha, boolean dropout) {
		MatrixStore<Double> layerHidden = input;
		if (dropout) {
			layerHidden = U.multiplyElements(layerHidden, dropout((int) layerHidden.countRows(), chance));
			layerHidden.multiply(1.0 / chance);
		}

		List<MatrixStore<Double>> layerHiddenList = new LinkedList<>();
		layerHiddenList.add(layerHidden);
		int size = weights.size() - 1;
		if (!useRelu) {
			for (int i = 0; i < size; i++) {
				Primitive64Store weight = (Primitive64Store) weights.get(i);
				layerHidden = weight.multiply(layerHidden);
				layerHiddenList.add(layerHidden);
			}
		} else {
			for (int i = 0; i < size; i++) {
				Primitive64Store weight = (Primitive64Store) weights.get(i);
				layerHidden = weight.multiply(layerHidden);
				layerHidden = relu(layerHidden);
				layerHiddenList.add(layerHidden);
			}
		}

		MatrixStore<Double> layerOutput = weights.get(weights.size() - 1).multiply(layerHidden);
		if (expected != null) {
			MatrixStore<Double> layerOutputDelta = (layerOutput.subtract(expected)).multiply(2.0 / expected.countRows());
			MatrixStore<Double> layerOutputWeightDelta = layerOutputDelta.multiply(layerHidden.transpose());
			var x1 = layerOutputWeightDelta.multiply(alpha);
			weights.set(weights.size() - 1, weights.get(weights.size() - 1).subtract(x1));
			for (int i = weights.size() - 2; i >= 0; --i) {
				MatrixStore<Double> layerHiddenDelta = multiplyElements(weights.get(i + 1).transpose().multiply(layerOutputDelta), reluDerivative(layerHiddenList.get(i + 1)));
				MatrixStore<Double> layerHiddenWeightDelta = layerHiddenDelta.multiply(layerHiddenList.get(i).transpose());
				weights.set(i, weights.get(i).subtract(layerHiddenWeightDelta.multiply(alpha)));
				layerOutputDelta = layerHiddenDelta;
			}
		}

		return layerOutput;
	}

	public void miniBatchGradientDescent(MatrixStore<Double> input, MatrixStore<Double> expected, double alpha, boolean softmax, ActivationFunctionType activation) {
		var layerHidden = input;
		layerHidden = weights.get(0).multiply(layerHidden);
		layerHidden = ActivationFunction.apply(layerHidden, activation);

		var layerOutput = weights.get(1).multiply(layerHidden);
		if (softmax) {
			ActivationFunction.softmax(layerOutput);
		}

		var layerOutputDelta = (layerOutput.subtract(expected)).multiply(2.0 / layerOutput.countRows());
		layerOutputDelta = layerOutputDelta.multiply(1.0 / input.countColumns());

		var layerHiddenDelta = (weights.get(1).transpose()).multiply(layerOutputDelta);
		layerHiddenDelta = multiplyElements(layerHiddenDelta, ActivationFunction.applyDerivative(layerHidden, activation));

		var layerOutputWeightDelta = layerOutputDelta.multiply(layerHidden.transpose());
		var layerHiddenWeightDelta = layerHiddenDelta.multiply(input.transpose());

		weights.set(0, weights.get(0).subtract(layerHiddenWeightDelta.multiply(alpha)));
		weights.set(1, weights.get(1).subtract(layerOutputWeightDelta.multiply(alpha)));
	}

	private static MatrixStore<Double> dropout(int n, double chance) {
		Primitive64Store res = Primitive64Store.FACTORY.make(n, 1);
		for (int i = 0; i < n; ++i) {
			res.set(i, 0, Math.random() < chance ? 1 : 0);
		}
		return res;
	}

	public void loadWeights(String filename) {
		List<Vector<Vector<Double>>> tempWeights = new LinkedList<>();
		tempWeights.add(new Vector<>());
		int matrixIdx = 0;
		int rowIdx = 0;
		try {
			BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
			do {
				String line = bufferedReader.readLine();
				if (line == null) break;

				if (line.isEmpty()) {
					tempWeights.add(new Vector<>());
					++matrixIdx;
					rowIdx = 0;
					continue;
				}

				tempWeights.get(matrixIdx).add(new Vector<>());
				String[] values = line.trim().split(",");
				for (String value : values) {
					tempWeights.get(matrixIdx).get(rowIdx).add(Double.parseDouble(value));
				}
				++rowIdx;

			} while (true);

			List<MatrixStore<Double>> weights = new LinkedList<>();

			for (int i = 0; i < matrixIdx + 1; ++i) {
				weights.add(Primitive64Store.FACTORY.make(tempWeights.get(i).size(), tempWeights.get(i).get(0).size()));
				for (int k = 0; k < tempWeights.get(i).size(); ++k) {
					for (int o = 0; o < tempWeights.get(i).get(0).size(); ++o) {
						((Primitive64Store) (weights.get(i))).set(k, o, tempWeights.get(i).get(k).get(o));
					}
				}
			}
			bufferedReader.close();
			this.weights = weights;
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void saveWeights(String filename) {
		try {
			BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(filename));
			int weightsSize = weights.size();
			for (int i = 0; i < weightsSize; i++) {
				MatrixStore<Double> weight = weights.get(i);
				for (int k = 0; k < weight.countRows(); ++k) {
					for (int o = 0; o < weight.countColumns(); ++o) {
						bufferedWriter.write(weight.get(k, o) + ", ");
					}
					if (i != weightsSize - 1 || k != weight.countRows() - 1) bufferedWriter.newLine();
				}
				if (i != weightsSize - 1) bufferedWriter.newLine();
			}
			bufferedWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	void print() {
		for (MatrixStore<Double> weight : weights) {
			for (int i = 0; i < weight.countRows(); i++) {
				for (int k = 0; k < weight.countColumns(); k++) {
					System.out.printf("%.2f ", weight.get(i, k));
				}
				System.out.println();
			}
		}
	}

}
