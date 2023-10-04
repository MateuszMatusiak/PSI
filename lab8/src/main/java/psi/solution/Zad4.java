package psi.solution;

import psi.Chromosome;

import java.util.*;

public class Zad4 {
    public static Point[] points = new Point[]{
            new Point(0, 119, 38),
            new Point(1, 37, 38),
            new Point(2, 197, 55),
            new Point(3, 85, 165),
            new Point(4, 12, 50),
            new Point(5, 100, 53),
            new Point(6, 81, 142),
            new Point(7, 121, 137),
            new Point(8, 85, 145),
            new Point(9, 80, 197),
            new Point(10, 91, 176),
            new Point(11, 106, 55),
            new Point(12, 123, 57),
            new Point(13, 40, 81),
            new Point(14, 78, 125),
            new Point(15, 190, 46),
            new Point(16, 187, 40),
            new Point(17, 37, 107),
            new Point(18, 17, 11),
            new Point(19, 67, 56),
            new Point(20, 78, 133),
            new Point(21, 87, 23),
            new Point(22, 184, 197),
            new Point(23, 111, 12),
            new Point(24, 66, 178)
    };

    public static void main1(String[] args) {
        var chr = new Chromosome(new int[]{20, 14, 17, 13, 4, 18, 1, 19, 21, 23, 0, 16, 15, 2, 12, 11, 5, 7, 22, 9, 24, 10, 3, 8, 6});
        System.out.println(fitness(chr));
    }


    public static void main(String[] args) {
        Random r = new Random();

        Chromosome[] population = new Chromosome[100];
        for (int i = 0; i < population.length; i++) {
            population[i] = new Chromosome(new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, true);
        }

        int iteration = 0;
        while (true) {
            Arrays.sort(population, Comparator.comparingDouble(Zad4::fitness));
            var fitness = fitness(population[0]);
            if (iteration % 1000 == 0)
                System.out.println(iteration + " generation:" + fitness);
            if (fitness < 1400) {
                System.out.println(iteration + " generation:");
                System.out.println("Found solution with fitness " + fitness);
                printItems(population[0]);
                System.out.println();
                if (fitness < 1200)
                    break;
            }

            List<Chromosome> newPopulation = new ArrayList<>(points.length);
            newPopulation.addAll(Arrays.asList(rouletteWheelSelection(population)));
            for (int i = 0; i < (population.length - (int) (population.length * 0.20)); ++i) {
                int pos1 = r.nextInt(population.length);
                int pos2 = r.nextInt(population.length);
                while (pos2 == pos1) {
                    pos2 = r.nextInt(population.length);
                }
                var child = population[pos1].crossOrder(population[pos2]);

                if (r.nextDouble() < 0.01) {
                    child.randomSwap();
                }
                newPopulation.add(child);
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

        int size = (int) (population.length * 0.20);
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

    public static double fitness(Chromosome chromosome) {
        double distance = 0;
        for (int i = 0; i < chromosome.genes.length; i++) {
            Point p1 = points[chromosome.genes[i]];
            Point p2 = points[chromosome.genes[(i + 1) % chromosome.genes.length]];
            distance += Math.sqrt(Math.pow(p2.x() - p1.x(), 2) + Math.pow(p2.y() - p1.y(), 2));
        }
        return distance;
    }

    public static void printItems(Chromosome chromosome) {
        for (int i = 0; i < chromosome.genes.length; i++) {
            System.out.println(points[chromosome.genes[i]]);
        }
    }

}
