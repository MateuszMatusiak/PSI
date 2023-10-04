package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.matrix.store.TransposedStore;

import java.util.List;

public class U {
	private U() {
	}

	public static MatrixStore<Double> multiplyElements(MatrixStore<Double> a, MatrixStore<Double> b) {
		if (a.countRows() != b.countRows() || a.countColumns() != b.countColumns())
			throw new IllegalArgumentException("Matrices must have the same dimensions");
		Primitive64Store result = Primitive64Store.FACTORY.make(a.countRows(), a.countColumns());
		for (int i = 0; i < a.countRows(); ++i) {
			for (int k = 0; k < a.countColumns(); ++k) {
				result.set(i, k, a.get(i, k) * b.get(i, k));
			}
		}
		return result;
	}

	public static MatrixStore<Double> createBatchInput(List<MatrixStore<Double>> matrixList, int start, int end) {
		int rows = (int) matrixList.get(0).countRows();
		int columns = end - start;
		Primitive64Store result = Primitive64Store.FACTORY.make(rows, columns);
		for (int i = start; i < end; ++i) {
			for (int k = 0; k < rows; ++k) {
				result.set(k, i - start, matrixList.get(i).get(k, 0));
			}
		}
		return result;
	}

	public static MatrixStore<Double> createBatchExpected(List<Integer> input, int start, int end, int range) {
		int height = end - start;
		Primitive64Store result = Primitive64Store.FACTORY.make(range, height);
		result.fillAll(0.0);

		for (int i = 0; i < height; ++i) {
			result.set(input.get(i + start).intValue(), i, 1);
		}
		return result;
	}

	public static void setMatrixVal(MatrixStore<Double> input, int i, int k, double value) {
		if (input instanceof Primitive64Store) {
			((Primitive64Store) input).set(i, k, value);
		} else if (input instanceof TransposedStore) {
			((Primitive64Store) (((TransposedStore<Double>) input).getOriginal())).set(k, i, value);
		} else {
			throw new RuntimeException(input.getClass().getSimpleName() + " is not supported instance");
		}
	}

	public static int findRowWithMaxValue(MatrixStore<Double> matrix) {
		int maxRowIndex = 0;
		double maxValue = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < matrix.countRows(); ++i) {
			MatrixStore<Double> row = matrix.row(i);
			double rowMaxValue = row.norm();

			if (rowMaxValue > maxValue) {
				maxValue = rowMaxValue;
				maxRowIndex = i;
			}
		}
		return maxRowIndex;
	}

	public static MatrixStore<Double> flatten(MatrixStore<Double> input) {
		long rows = input.countColumns() * input.countRows();
		var result = Primitive64Store.FACTORY.make(rows, 1);
		int p = 0;
		for (int i = 0; i < input.countRows(); ++i) {
			for (int k = 0; k < input.countColumns(); ++k) {
				result.set(p, 0, input.get(i, k));
				p++;
			}
		}
		return result;
	}

	public static MatrixStore<Double> reshape(MatrixStore<Double> input, long rows, long columns) {
		var result = Primitive64Store.FACTORY.make(rows, columns);
		int p = 0;
		for (int i = 0; i < rows; ++i) {
			for (int k = 0; k < columns; ++k) {
				result.set(i, k, input.get(p, 0));
				p++;
			}
		}
		return result;
	}

	public static int result(MatrixStore<Double> input) {
		var value = input.get(0, 0);
		var index = 0;
		for (int i = 1; i < input.countRows(); ++i) {
			if (value < input.get(i, 0)) {
				value = input.get(i, 0);
				index = i;
			}
		}
		return index;
	}

}

