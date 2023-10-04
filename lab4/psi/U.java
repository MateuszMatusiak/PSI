package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.matrix.store.TransposedStore;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

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
		int cols = end - start;
		Primitive64Store result = Primitive64Store.FACTORY.make(rows, cols);
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

		for (int i = 0; i < matrix.countRows(); i++) {
			MatrixStore<Double> row = matrix.row(i);
			double rowMaxValue = row.norm();

			if (rowMaxValue > maxValue) {
				maxValue = rowMaxValue;
				maxRowIndex = i;
			}
		}
		return maxRowIndex;
	}
}

