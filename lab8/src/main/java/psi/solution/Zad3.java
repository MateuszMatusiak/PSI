package psi.solution;

import psi.Chromosome;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Zad3 {
	public static Item[] items = new Item[]{
			new Item(3, 266),
			new Item(13, 442),
			new Item(10, 671),
			new Item(9, 526),
			new Item(7, 388),
			new Item(1, 245),
			new Item(8, 210),
			new Item(8, 145),
			new Item(2, 126),
			new Item(9, 322)
	};

	public static void main(String[] args) {

		Chromosome[] population = new Chromosome[8];
		for (int i = 0; i < population.length; i++) {
			population[i] = new Chromosome(items.length);
		}
		int iteration = 0;
		while (true) {
			Arrays.sort(population, (a, b) -> fitness(b) - fitness(a));
			var fitness = fitness(population[0]);
			if (fitness > 2200) {
				System.out.println(iteration + " generation:");
				System.out.println("Found solution with fitness " + fitness);
				printItems(population[0]);
				break;
			}

			List<Chromosome> newPopulation = new ArrayList<>(items.length);
			newPopulation.addAll(Arrays.asList(rouletteWheelSelection(population)));
			for (int i = 0; i < (population.length - (population.length * 0.25)) / 2; ++i) {
				int pos1 = (int) (Math.random() * population.length);
				int pos2 = (int) (Math.random() * population.length);
				while (pos2 == pos1) {
					pos2 = (int) (Math.random() * population.length);
				}
				var children = population[pos1].crossOnePoint(population[pos2]);
				if (Math.random() < 0.05)
					children.first().replacement();
				newPopulation.add(children.first());
				if (Math.random() < 0.05)
					children.second().replacement();
				newPopulation.add(children.second());
			}
			if ((population.length - (population.length * 0.25)) % 2 != 0) {
				int pos1 = (int) (Math.random() * population.length);
				int pos2 = (int) (Math.random() * population.length);
				while (pos2 == pos1) {
					pos2 = (int) (Math.random() * population.length);
				}
				var children = population[pos1].crossOnePoint(population[pos2]);
				if (Math.random() < 0.05)
					children.first().replacement();
				newPopulation.add(children.first());
			}

			population = newPopulation.toArray(new Chromosome[0]);

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
		int size = (int) (population.length * 0.25);
		Chromosome[] res = new Chromosome[size];
		for (int i = 0; i < size; i++) {
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

	public static int fitness(Chromosome chromosome) {
		int weight = 0;
		int value = 0;
		for (int i = 0; i < chromosome.genes.length; i++) {
			if (chromosome.genes[i] == 1) {
				weight += items[i].weight();
				value += items[i].value();
			}
		}
		if (weight > 35)
			return 0;
		return value;
	}

	public static void printItems(Chromosome chromosome) {
		for (int i = 0; i < chromosome.genes.length; i++) {
			if (chromosome.genes[i] == 1) {
				System.out.println(items[i]);
			}
		}
	}
}
