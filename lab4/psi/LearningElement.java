package com.psi;

import org.ojalgo.matrix.store.MatrixStore;

public record LearningElement(MatrixStore<Double> input,
							  Integer expected) {
}
