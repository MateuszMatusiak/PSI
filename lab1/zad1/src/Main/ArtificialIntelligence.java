package Main;


import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Vector;

public class ArtificialIntelligence {

	Vector<Matrix> weights;
	int w = 0;

	public ArtificialIntelligence() {
		weights = new Vector<>();
	}

	public ArtificialIntelligence(int rows, int columns) {
		weights = new Vector<>();
		Matrix temp = Matrix.random(rows, columns, new Random());
		weights.add(temp);
		w++;
	}

	public void addLayer(int n, Double weightMinVal, Double weightMaxVal) {
		addLayerProc(n, weightMinVal, weightMaxVal);
	}

	//macierz razy liczba neurownów w poprzedniej
	public void addLayer(int n) {
		addLayerProc(n, (double) -1, 1.0);
	}

	private void addLayerProc(int n, Double min, Double max) {
		Random r = new Random();
		Matrix prev = weights.get(w - 1);
		Matrix temp = new Basic2DMatrix(n, prev.rows());

		for (int i = 0; i < n; ++i) {
			for (int k = 0; k < prev.rows(); ++k) {
				temp.set(i, k, r.nextDouble(min, max));
			}
		}
		weights.add(temp.multiply(prev.rows()));
		w++;
	}

	public Vector<Double> predict(Vector<Double> input) {
		Vector<Double> res;
		res = deepNeuralNetwork(input, weights);
		return res;
	}

	public void loadWeights(String filename) {
		Vector<Vector<Vector<Double>>> tempWeights = new Vector<>();
		tempWeights.add(new Vector<>());
		int matrixIdx = 0;
		int rowIdx = 0;
		try {
			BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

			do {
				String line = bufferedReader.readLine();
				if (line == null)
					break;

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

			for (int i = 0; i < matrixIdx + 1; ++i) {
				weights.add(new Basic2DMatrix(tempWeights.get(i).size(), tempWeights.get(i).get(0).size()));
				for (int k = 0; k < tempWeights.get(i).size(); ++k) {
					for (int o = 0; o < tempWeights.get(i).get(0).size(); ++o) {
						weights.get(i).set(k, o, tempWeights.get(i).get(k).get(o));
					}
				}
			}
			bufferedReader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static Double neuron(Vector<Double> input, org.la4j.Vector weights, Double bias) {
		int n = input.size();
		double res = 0;
		for (int i = 0; i < n; ++i) {
			res += input.get(i) * weights.get(i) + bias;
		}
		return res;
	}

	public static Vector<Double> neuralNetwork(Vector<Double> input, Matrix weight) {
		Vector<Double> res = new Vector<>();
		for (int i = 0; i < weight.rows(); ++i) {
			res.add(neuron(input, weight.getRow(i), (double) 0));
		}
		return res;
	}

	public static Vector<Double> deepNeuralNetwork(Vector<Double> input, Vector<Matrix> weights) {
		Vector<Double> res = input;
		for (Matrix weight : weights) {
			res = neuralNetwork(res, weight);
		}
		return res;
	}
}
