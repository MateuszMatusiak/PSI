package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MNIST {
	private static final String TRAIN_IMAGES_PATH = "./train-images.idx3-ubyte";
	private static final String TRAIN_LABELS_PATH = "./train-labels.idx1-ubyte";
	private static final String TEST_IMAGES_PATH = "./t10k-images.idx3-ubyte";
	private static final String TEST_LABELS_PATH = "./t10k-labels.idx1-ubyte";

	public static List<MatrixStore<Double>> getTrainingImages() throws IOException {
		return readImages(TRAIN_IMAGES_PATH);
	}

	public static List<Integer> getTrainingLabels() throws IOException {
		return readLabels(TRAIN_LABELS_PATH);
	}

	public static List<MatrixStore<Double>> getTestImages() throws IOException {
		return readImages(TEST_IMAGES_PATH);
	}

	public static List<Integer> getTestLabels() throws IOException {
		return readLabels(TEST_LABELS_PATH);
	}

	public static List<MatrixStore<Double>> readImages(String filePath) throws IOException {
		DataInputStream dataInputStream = new DataInputStream(new FileInputStream(filePath));

		int magicNumber = dataInputStream.readInt();
		if (magicNumber != 2051) {
			throw new IOException("Invalid magic number in image file: " + magicNumber);
		}

		int imagesCount = dataInputStream.readInt();
		int rows = dataInputStream.readInt();
		int cols = dataInputStream.readInt();

		List<MatrixStore<Double>> res = new ArrayList<>(imagesCount);
		for (int i = 0; i < imagesCount; i++) {
			Primitive64Store temp = Primitive64Store.FACTORY.make(rows * cols, 1);
			for (int k = 0; k < rows * cols; k++) {
				temp.set(k, 0, (dataInputStream.readUnsignedByte() / 255.0));
			}
			res.add(temp);
		}

		dataInputStream.close();

		return res;
	}

	public static List<Integer> readLabels(String filePath) throws IOException {
		DataInputStream dataInputStream = new DataInputStream(new FileInputStream(filePath));

		int magicNumber = dataInputStream.readInt();
		if (magicNumber != 2049) {
			throw new IOException("Invalid magic number in label file: " + magicNumber);
		}

		int labelsCount = dataInputStream.readInt();
		List<Integer> labels = new ArrayList<>(labelsCount);

		for (int i = 0; i < labelsCount; i++) {
			labels.add(dataInputStream.readUnsignedByte());
		}

		dataInputStream.close();
		return labels;
	}

}
