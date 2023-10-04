package psi.solution;

import psi.Chromosome;

import java.util.Arrays;

public class Zad1 {

	public static void main(String[] args) {
		Chromosome[] population = new Chromosome[10];
		for (int i = 0; i < population.length; i++) {
			population[i] = new Chromosome(10);
		}
		int iteration = 0;
		while (iteration < 100) {
			System.out.println(iteration + " generation:");
			System.out.println(population[0]);
			iteration(population);
			iteration++;
		}
	}

	private static void iteration(Chromosome[] population) {
		Arrays.sort(population, (a, b) -> fitness(b) - fitness(a));
		var children = population[0].crossOnePoint(population[1]);
		if (Math.random() < 0.6) {
			population[0].replacement();
		}
		if (Math.random() < 0.6) {
			population[1].replacement();
		}
		population[population.length - 1] = children.first();
		population[population.length - 2] = children.second();
	}

	public static int fitness(Chromosome chromosome) {
		int res = 0;
		for (int i = 0; i < chromosome.size(); i++) {
			res += chromosome.getGene(i);
		}
		return res;
	}
}
