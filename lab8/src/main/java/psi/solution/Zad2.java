package psi.solution;

import psi.Chromosome;

import java.util.Arrays;
import java.util.Comparator;

public class Zad2 {
	public static void main(String[] args) {
		Chromosome[] population = new Chromosome[10];
		for (int i = 0; i < population.length; i++) {
			population[i] = new Chromosome(8);
		}
		int a = 0;
		int b = 0;
		int iteration = 0;
		while (true) {
			Arrays.sort(population, Comparator.comparingDouble(Zad2::fitness));
			var chromosome = population[population.length - 1];
			int[] numbers = splitIntArray(chromosome.genes);
			a = numbers[0];
			b = numbers[1];
			System.out.println(iteration + " generation:");
			System.out.println("a = " + a + ", b = " + b + ", fitness = " + fitness(chromosome));
			if (equation(a, b))
				break;
			population = rouletteWheelSelection(population);
			for (int i = 0; i < population.length / 2; i++) {
				int pos2 = (int) (Math.random() * (population.length / 2));
				population[pos2] = population[i].crossOnePoint(population[pos2]).first();

				if (Math.random() < 0.1)
					population[i].replacement();
			}
			iteration++;
		}
	}

	public static Chromosome[] rouletteWheelSelection(Chromosome[] population) {
		double[] fitness = new double[population.length];
		double sum = 0;
		for (int i = 0; i < population.length; i++) {
			fitness[i] = fitness(population[i]);
			sum += fitness[i];
		}
		for (int i = 0; i < population.length; i++) {
			fitness[i] = fitness[i] / sum;
		}
		Chromosome[] res = new Chromosome[population.length];
		for (int i = 0; i < population.length; i++) {
			res[i] = getRandomElement(population, fitness);
		}
		return res;
	}

	public static Chromosome getRandomElement(Chromosome[] elements, double[] probabilities) {
		double randomValue = Math.random();
		double cumulativeProbability = 0.0;

		for (int i = 0; i < elements.length; i++) {
			cumulativeProbability += probabilities[i];
			if (randomValue <= cumulativeProbability) {
				return elements[i];
			}
		}
		return elements[0];
	}

	public static double fitness(Chromosome chromosome) {
		int[] numbers = splitIntArray(chromosome.genes);
		int a = numbers[0];
		int b = numbers[1];
		double y = (2 * a * a) + b;

		return 33.0 - Math.abs((33.0 - y) / 33.0);
	}

	public static int[] splitIntArray(int[] intArray) {
		int[] result = new int[2];
		for (int i = 0; i < 4; i++) {
			result[0] |= (intArray[i]) << (3 - i);
		}

		for (int i = 4; i < 8; i++) {
			result[1] |= (intArray[i]) << (7 - i);
		}
		return result;
	}

	public static boolean equation(int a, int b) {
		return (2 * a * a) + b == 33;
	}
}
