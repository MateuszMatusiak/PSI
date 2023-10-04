package com.psi;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.Primitive64Store;
import org.ojalgo.random.Uniform;

import java.util.ArrayList;
import java.util.List;

public class ConvolutionalNeuralNetworks {
    MatrixStore<Double> kernels;
    MatrixStore<Double> wy;
    List<Point> points;

    public ConvolutionalNeuralNetworks(MatrixStore<Double> kernels, MatrixStore<Double> wy) {
        this.kernels = kernels;
        this.wy = wy;
        this.points = new ArrayList<>();
    }

    public MatrixStore<Double> calculate(MatrixStore<Double> imageSections, double alpha, MatrixStore<Double> expected, boolean pool) {
        MatrixStore<Double> layerOutput;
        var kernelLayer = imageSections.multiply(kernels.transpose());
        if (pool) {
            var filter = Primitive64Store.FACTORY.rows(new double[][]{{0, 1}, {1, 0}});
            var conv = ConvolutionalNeuralNetworks.conv(kernelLayer, filter);
            conv = ActivationFunction.relu(conv);
            var pooled = pool(ActivationFunction.relu(conv), 2, new ArrayList<>());
            layerOutput = wy.multiply(pooled);
        } else {
            layerOutput = wy.multiply(U.flatten(kernelLayer));
        }
        if (expected == null) {
            return layerOutput;
        }
        var layerOutputDelta = layerOutput.subtract(expected).multiply(2.0 / layerOutput.countRows());
        var kernelLayerDelta = (wy.transpose()).multiply(layerOutputDelta);
        var layerOutputWeightDelta = layerOutputDelta.multiply(U.flatten(kernelLayer).transpose());
        var reshaped = U.reshape(kernelLayerDelta, kernelLayer.countRows(), 1).transpose();
        var kernelLayerWeightDelta = reshaped.multiply(imageSections);
        wy = wy.subtract(layerOutputWeightDelta.multiply(alpha));
        kernels = kernels.subtract(kernelLayerWeightDelta.multiply(alpha));
        return layerOutput;
    }

    public static MatrixStore<Double> pool(MatrixStore<Double> input, int stride, List<Point> maxPoints) {
        int columns = ((int) Math.ceil((double) input.countColumns() / stride));
        int rows = ((int) Math.ceil((double) input.countRows() / stride));
        var result = Primitive64Store.FACTORY.make(rows, columns);

        for (int resRow = 0, i = 0; i < input.countRows(); i += stride, resRow++) {
            for (int resCol = 0, k = 0; k < input.countColumns(); k += stride, resCol++) {
                double max = Double.NEGATIVE_INFINITY;
                Point maxPoint = null;

                for (int m = 0; m < stride; ++m) {
                    for (int n = 0; n < stride; ++n) {
                        double value = 0;
                        if (i + m < input.countRows() && k + n < input.countColumns()) {
                            value = input.get(i + m, k + n);
                        }
                        if (value > max) {
                            max = value;
                            maxPoint = new Point(resRow, resCol, i + m, k + n);
                        }
                    }
                }
                result.set(resRow, resCol, max);
                maxPoints.add(maxPoint);
            }
        }
        return result;
    }

    public static MatrixStore<Double> conv(MatrixStore<Double> input, MatrixStore<Double> filter) {
        int columns = ((int) (input.countColumns() - filter.countColumns()) + 1);
        int rows = ((int) (input.countRows() - filter.countRows()) + 1);
        var result = Primitive64Store.FACTORY.make(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < columns; k++) {
                double sum = 0;
                for (int o = 0; o < filter.countRows(); o++) {
                    for (int p = 0; p < filter.countColumns(); p++) {
                        sum += filter.get(o, p) * input.get(i + o, k + p);
                    }
                }
                result.set(i, k, sum);
            }
        }
        return result;
    }

    public static MatrixStore<Double> split(MatrixStore<Double> input, int columns, int rows, int padding) {
        int h = (int) (Math.floor((double) (input.countColumns() - columns) / padding) + 1);
        int v = (int) (Math.floor((double) (input.countRows() - rows) / padding) + 1);
        var splitNumber = h * v;
        var result = Primitive64Store.FACTORY.make(splitNumber, rows * columns);

        h = 0;
        v = 0;
        for (var k = 0; k < splitNumber; k++) {
            var p = 0;
            for (var i = 0; i < columns; i++)
                for (var j = 0; j < rows; j++) {
                    result.set(k, p, input.get(j + v, i + h));
                    p++;
                }
            if (h + columns >= input.countColumns()) {
                h = 0;
                v += padding;
            } else {
                h += padding;
            }
        }
        return result;
    }

    public static MatrixStore<Double> connectMatricesVertically(List<MatrixStore<Double>> matrices) {
        var res = U.flatten(matrices.get(0)).transpose();
        for (int i = 1; i < matrices.size(); i++) {
            res = connectMatricesVertically(res, U.flatten(matrices.get(i)).transpose());
        }
        return res;
    }

    public static MatrixStore<Double> connectMatricesVertically(MatrixStore<Double> a, MatrixStore<Double> b) {
        var result = Primitive64Store.FACTORY.make(a.countRows() + b.countRows(), a.countColumns());
        for (var i = 0; i < a.countRows() + b.countRows(); i++) {
            if (i < a.countRows()) {
                for (var k = 0; k < a.countColumns(); k++) {
                    result.set(i, k, a.get(i, k));
                }
            } else {
                for (var k = 0; k < b.countColumns(); k++) {
                    result.set(i, k, b.get(i - a.countRows(), k));
                }
            }
        }
        return result;
    }

    public static List<MatrixStore<Double>> randomizeMatrices(int columns, int rows, int count, double min, double range) {
        var res = new ArrayList<MatrixStore<Double>>();
        for (var i = 0; i < count; i++) {
            MatrixStore<Double> temp = Primitive64Store.FACTORY.makeFilled(rows, columns, Uniform.of(min, range));
            res.add(temp);
        }
        return res;
    }
}
