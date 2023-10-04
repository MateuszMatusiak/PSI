package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static com.psi.U.*;

public class Main {

	public static int iterations = 350;
	public static int trainingSize = 1000;
	public static int testingSize = 10000;
	public static int batch = 100;
	public static double alpha = 0.005;
	public static int neurons = 40;

	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		try {
			zad1();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Time: " + (System.currentTimeMillis() - startTime) / 1000.0);
		System.out.println(new Date());

	}

	public static void test() {
		ArtificialIntelligence AI = new ArtificialIntelligence();
		AI.loadWeights("data.txt");

		double[][] xArray = {{0.5, 0.1, 0.2, 0.8},
				{0.75, 0.3, 0.1, 0.9},
				{0.1, 0.7, 0.6, 0.2}};

		double[][] yArray = {{0.1, 0.5, 0.1, 0.7},
				{1.0, 0.2, 0.3, 0.6},
				{0.1, -0.5, 0.2, 0.2}};

		Primitive64Store input = Primitive64Store.FACTORY.make(3, 4);
		Primitive64Store expected = Primitive64Store.FACTORY.make(3, 4);

		for (int i = 0; i < 3; ++i) {
			for (int k = 0; k < 4; ++k) {
				input.set(i, k, xArray[i][k]);
				expected.set(i, k, yArray[i][k]);
			}
		}

		AI.miniBatchGradientDescent(input, expected, 0.01, false, ActivationFunctionType.RELU);

	}

	public static void zad1() throws IOException {
		System.out.println("Loading");
		List<MatrixStore<Double>> trainingImages = MNIST.getTrainingImages();
		List<Integer> trainingLabels = MNIST.getTrainingLabels();
		List<MatrixStore<Double>> testImages = MNIST.getTestImages();
		List<Integer> testLabels = MNIST.getTestLabels();
		System.out.println("Loaded");
		ArtificialIntelligence AI = new ArtificialIntelligence(neurons, 784, -0.1, 0.2);

		AI.addLayer(10, -0.1, 0.2);

		var res = new ArrayList<MatrixStore<Double>>(10);
		for (int i = 0; i < 10; ++i) {
			var temp = Primitive64Store.FACTORY.make(10, 1);
			temp.fillAll(0.0);
			temp.set(i, 0, 1);
			res.add(temp);
		}

		for (int p = 0; p < iterations; p++) {
			for (int i = 0; i < (Math.min(trainingImages.size(), trainingSize)); i++) {
				AI.fit(trainingImages.get(i), res.get(trainingLabels.get(i)), alpha, true);
			}
			System.out.println((p + 1) + ": Learned");

			var correct = 0;
			for (int i = 0; i < (Math.min(testImages.size(), testingSize)); i++) {
				var result = AI.predict(testImages.get(i));
				if (findRowWithMaxValue(result) == testLabels.get(i)) {
					correct++;
				}
			}
			System.out.println(correct + " / " + testImages.size());
		}
	}

	public static void zad2() throws IOException {
		int iterations = 350;
		int trainingSize = 1000;
		int testingSize = 10000;
		int batch = 100;
		double alpha = 0.1;
		int neurons = 40;

		System.out.println("Loading");
		List<MatrixStore<Double>> trainingImages = MNIST.getTrainingImages();
		List<Integer> trainingLabels = MNIST.getTrainingLabels();
		List<MatrixStore<Double>> testImages = MNIST.getTestImages();
		List<Integer> testLabels = MNIST.getTestLabels();
		System.out.println("Loaded");
		ArtificialIntelligence AI = new ArtificialIntelligence(neurons, 784, -0.1, 0.2);

		AI.addLayer(10, -0.1, 0.2);

		for (int p = 0; p < iterations; p++) {
			for (int i = 0; i < (Math.min(trainingImages.size(), trainingSize)); i += batch) {
				var input = createBatchInput(trainingImages, i, i + batch);
				var expected = createBatchExpected(trainingLabels, i, i + batch, 10);
				AI.miniBatchGradientDescent(input, expected, alpha, false, ActivationFunctionType.RELU);
			}
			System.out.println((p + 1) + ": Learned");

			var correct = 0;
			for (int i = 0; i < (Math.min(testImages.size(), testingSize)); i++) {
				var result = AI.predict(testImages.get(i));
				if (findRowWithMaxValue(result) == testLabels.get(i)) {
					correct++;
				}
			}
			System.out.println(correct + " / " + testImages.size());
		}
	}

	public static void zad3() throws IOException {
		int iterations = 350;
		int trainingSize = 1000;
		int testingSize = 10000;
		int batch = 100;
		double alpha = 0.02;
		int neurons = 100;

		System.out.println("Loading");
		List<MatrixStore<Double>> trainingImages = MNIST.getTrainingImages();
		List<Integer> trainingLabels = MNIST.getTrainingLabels();
		List<MatrixStore<Double>> testImages = MNIST.getTestImages();
		List<Integer> testLabels = MNIST.getTestLabels();
		System.out.println("Loaded");
		ArtificialIntelligence AI = new ArtificialIntelligence(neurons, 784, -0.01, 0.02);

		AI.addLayer(10, -0.01, 0.02);

		for (int p = 0; p < iterations; p++) {
			for (int i = 0; i < (Math.min(trainingImages.size(), trainingSize)); i += batch) {
				var input = createBatchInput(trainingImages, i, i + batch);
				var expected = createBatchExpected(trainingLabels, i, i + batch, 10);
				AI.miniBatchGradientDescent(input, expected, alpha, true, ActivationFunctionType.TANH);
			}
			System.out.println((p + 1) + ": Learned");

			var correct = 0;
			for (int i = 0; i < (Math.min(testImages.size(), testingSize)); i++) {
				var result = AI.predict(testImages.get(i));
				if (findRowWithMaxValue(result) == testLabels.get(i)) {
					correct++;
				}
			}
			System.out.println(correct + " / " + testImages.size());
		}
	}
}
