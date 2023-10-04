package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;

import static com.psi.U.setMatrixVal;

enum ActivationFunctionType {
	SIGMOID,
	TANH,
	RELU,
	SOFTMAX,
	NONE
}

public class ActivationFunction {
	private ActivationFunction() {
	}

	public static MatrixStore<Double> apply(MatrixStore<Double> input, ActivationFunctionType type) {
		switch (type) {
			case SIGMOID -> {
				return sigmoid(input);
			}
			case TANH -> {
				return tanh(input);
			}
			case RELU -> {
				return relu(input);
			}
			case SOFTMAX -> {
				return softmax(input);
			}
			default -> {
				return input;
			}
		}
	}

	public static MatrixStore<Double> applyDerivative(MatrixStore<Double> input, ActivationFunctionType type) {
		switch (type) {
			case SIGMOID -> {
				return sigmoidDerivative(input);
			}
			case TANH -> {
				return tanhDerivative(input);
			}
			case RELU -> {
				return reluDerivative(input);
			}
			default -> {
				return input;
			}
		}
	}

	public static MatrixStore<Double> relu(MatrixStore<Double> input) {
		Primitive64Store res = Primitive64Store.FACTORY.make(input.countRows(), input.countColumns());
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				if (input.get(i, k) > 0) {
					res.set(i, k, input.get(i, k));
				}
			}
		}
		return res;
	}

	public static MatrixStore<Double> reluDerivative(MatrixStore<Double> input) {
		Primitive64Store res = Primitive64Store.FACTORY.make(input.countRows(), input.countColumns());
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				res.set(i, k, input.get(i, k) > 0 ? 1 : 0);
			}
		}
		return res;
	}

	public static MatrixStore<Double> sigmoid(MatrixStore<Double> input) {
		Primitive64Store res = Primitive64Store.FACTORY.make(input.countRows(), input.countColumns());
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				res.set(i, k, 1.0 / (1.0 + Math.exp(-input.get(i, k))));
			}
		}
		return res;
	}

	public static MatrixStore<Double> sigmoidDerivative(MatrixStore<Double> input) {
		Primitive64Store res = Primitive64Store.FACTORY.make(input.countRows(), input.countColumns());
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				res.set(i, k, input.get(i, k) * (1 - input.get(i, k)));
			}
		}
		return res;
	}

	public static MatrixStore<Double> tanh(MatrixStore<Double> input) {
		Primitive64Store res = Primitive64Store.FACTORY.make(input.countRows(), input.countColumns());
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				res.set(i, k, Math.tanh(input.get(i, k)));
			}
		}
		return res;
	}

	public static MatrixStore<Double> tanhDerivative(MatrixStore<Double> input) {
		Primitive64Store res = Primitive64Store.FACTORY.make(input.countRows(), input.countColumns());
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				res.set(i, k, 1 - Math.pow(Math.tanh(input.get(i, k)), 2));
			}
		}
		return res;
	}

	public static MatrixStore<Double> softmax(MatrixStore<Double> input) {
		Primitive64Store res = Primitive64Store.FACTORY.make(input.countRows(), input.countColumns());
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				double sum = 0;
				for (int o = 0; o < input.countColumns(); ++o) {
					sum += Math.exp(input.get(i, o));
				}
				res.set(i, k, Math.exp(input.get(i, k)) / sum);
			}
		}
		return res;
	}

}
