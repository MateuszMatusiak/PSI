package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.random.Uniform;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static com.psi.ConvolutionalNeuralNetworks.*;

public class Solution {

	public static void main(String[] args) {
		long start = System.currentTimeMillis();
		test();
		System.out.println("Time: " + (System.currentTimeMillis() - start) / 1000.0);
		System.out.println(new Date());
	}

	public static void test() {
		Primitive64Store input = Primitive64Store.FACTORY.rows(new double[][]{
				{8.5, 0.65, 1.2},
				{9.5, 0.8, 1.3},
				{9.9, 0.8, 0.5},
				{9.0, 0.9, 1.0}
		});
		var expected = Primitive64Store.FACTORY.rows(new double[][]{
				{0.0},
				{1.0}
		});

		var k1w = Primitive64Store.FACTORY.rows(new double[][]{
				{0.1, 0.2, -0.1, -0.1, 0.1, 0.9, 0.1, 0.4, 0.1}
		});
		var k2w = Primitive64Store.FACTORY.rows(new double[][]{
				{0.3, 1.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1}
		});

		var wy = Primitive64Store.FACTORY.rows(new double[][]{
				{0.1, -0.2, 0.1, 0.3},
				{0.2, 0.1, 0.5, -0.3}
		});


		var fragmented = ConvolutionalNeuralNetworks.split(input, 3, 3, 1);
		System.out.println(fragmented.countColumns() + " " + fragmented.countRows());
		var kernels = connectMatricesVertically(k1w, k2w);

		var CNN = new ConvolutionalNeuralNetworks(kernels, wy);
		var output = CNN.calculate(fragmented, 0.01, expected, false);
		System.out.println("\ncalculate");
		System.out.println(output);
		System.out.println(CNN.wy);
		System.out.println(CNN.kernels);
	}

	public static void zad1() {
		var input = Primitive64Store.FACTORY.rows(new double[][]{
				{1, 1, 1, 0, 0},
				{0, 1, 1, 1, 0},
				{0, 0, 1, 1, 1},
				{0, 0, 1, 1, 0},
				{0, 1, 1, 0, 0}
		});
		var filter = Primitive64Store.FACTORY.rows(new double[][]{
				{1, 0, 1},
				{0, 1, 0},
				{1, 0, 1}
		});
		var result = ConvolutionalNeuralNetworks.conv(input, filter);
		System.out.println(result);
	}

	public static void zad2() throws IOException {
		int iterations = 50;
		int trainingSize = 1000;
		int testingSize = 10000;
		double alpha = 0.01;
		int neurons = 10;

		System.out.println("Loading");
		List<MatrixStore<Double>> trainingImages = MNIST.getTrainingImages();
		List<Integer> trainingLabels = MNIST.getTrainingLabels();
		List<MatrixStore<Double>> testImages = MNIST.getTestImages();
		List<Integer> testLabels = MNIST.getTestLabels();
		System.out.println("Loaded");
		var res = new ArrayList<MatrixStore<Double>>();
		for (int i = 0; i < neurons; ++i) {
			var temp = Primitive64Store.FACTORY.make(neurons, 1);
			temp.fillAll(0.0);
			temp.set(i, 0, 1);
			res.add(temp);
		}

		var filters = ConvolutionalNeuralNetworks.randomizeMatrices(3, 3, 16, -0.01, 0.02);
		var kernels = connectMatricesVertically(filters);
		kernels = ActivationFunction.relu(kernels);

		var wy = Primitive64Store.FACTORY.makeFilled(10, 10816, Uniform.of(-0.1, 0.2));
		var CNN = new ConvolutionalNeuralNetworks(kernels, wy);

		for (int p = 0; p < iterations; p++) {
			for (int i = 0; i < Math.min(trainingSize, trainingImages.size()); ++i) {
				var reshaped = U.reshape(trainingImages.get(i), 28, 28);
				var fragmented = ConvolutionalNeuralNetworks.split(reshaped, 3, 3, 1);
				CNN.calculate(fragmented, alpha, res.get(trainingLabels.get(i)), false);
			}
			System.out.println((p + 1) + ": Learned");

			var correct = 0;
			for (int i = 0; i < Math.min(testImages.size(), testingSize); i++) {
				var testImage = testImages.get(i);
				var reshaped = U.reshape(testImage, 28, 28);
				var fragmented = ConvolutionalNeuralNetworks.split(reshaped, 3, 3, 1);
				var output = CNN.calculate(fragmented, alpha, null, false);
				if (U.result(output) == testLabels.get(i)) {
					correct++;
				}
			}
			System.out.println((p + 1) + ": " + correct + " / " + Math.min(testImages.size(), testingSize));
		}

	}

	public static void zad3() throws IOException {
		int iterations = 50;
		int trainingSize = 1000;
		int testingSize = 10000;
		double alpha = 0.001;
		int neurons = 10;

		System.out.println("Loading");
		List<MatrixStore<Double>> trainingImages = MNIST.getTrainingImages();
		List<Integer> trainingLabels = MNIST.getTrainingLabels();
		List<MatrixStore<Double>> testImages = MNIST.getTestImages();
		List<Integer> testLabels = MNIST.getTestLabels();
		System.out.println("Loaded");
		var res = new ArrayList<MatrixStore<Double>>();
		for (int i = 0; i < neurons; ++i) {
			var temp = Primitive64Store.FACTORY.make(neurons, 1);
			temp.fillAll(0.0);
			temp.set(i, 0, 1);
			res.add(temp);
		}

		var filters = ConvolutionalNeuralNetworks.randomizeMatrices(3, 3, 16, -0.1, 0.2);
		var kernels = connectMatricesVertically(filters);
		var wy = Primitive64Store.FACTORY.makeFilled(10, 676, Uniform.of(-0.01, 0.02));
		var CNN = new ConvolutionalNeuralNetworks(kernels, wy);

		for (int p = 0; p < iterations; p++) {
			for (int i = 0; i < Math.min(trainingSize, trainingImages.size()); ++i) {
				var reshaped = U.reshape(trainingImages.get(i), 28, 28);
				var fragmented = ConvolutionalNeuralNetworks.split(reshaped, 3, 3, 1);
				CNN.calculate(fragmented, alpha, res.get(trainingLabels.get(i)), true);
			}
			System.out.println((p + 1) + ": Learned");

			var correct = 0;
			for (int i = 0; i < Math.min(testImages.size(), testingSize); i++) {
				var testImage = testImages.get(i);
				var reshaped = U.reshape(testImage, 28, 28);
				var fragmented = ConvolutionalNeuralNetworks.split(reshaped, 3, 3, 1);
				var output = CNN.calculate(fragmented, alpha, null, true);
				if (U.result(output) == testLabels.get(i)) {
					correct++;
				}
			}
			System.out.println((p + 1) + ": " + correct + " / " + Math.min(testImages.size(), testingSize));
		}
	}
}
