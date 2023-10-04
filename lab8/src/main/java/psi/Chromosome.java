package psi;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Chromosome {
    public int[] genes;
    private Random r = new Random();

    public Chromosome(int size) {
        genes = new int[size];
        for (int i = 0; i < genes.length; i++) {
            genes[i] = Math.random() < 0.5 ? 0 : 1;
        }
    }

    public Chromosome(int[] genes) {
        this.genes = genes;
    }

    public Chromosome(int[] genes, boolean shuffle) {
        this.genes = genes;
        if (shuffle) {
            shuffleArray(this.genes);
        }
    }

    public Chromosome(Chromosome chromosome) {
        this.genes = chromosome.genes.clone();
    }

    public int size() {
        return genes.length;
    }

    public int getGene(int index) {
        return genes[index];
    }

    public void setGene(int index, int value) {
        genes[index] = value;
    }

    public static void shuffleArray(int[] array) {
        Random rand = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            int index = rand.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[index];
            array[index] = temp;
        }
    }

    public void replacement() {
        int pos = r.nextInt(genes.length);
        this.replacement(pos);
    }

    public void replacement(int pos) {
        genes[pos] = genes[pos] == 0 ? 1 : 0;
    }

    public void randomSwap() {
        int pos1 = r.nextInt(genes.length);
        int pos2 = r.nextInt(genes.length);
        int temp = genes[pos1];
        genes[pos1] = genes[pos2];
        genes[pos2] = temp;
    }

    public void adjacentSwap() {
        int pos = r.nextInt(genes.length - 1);
        int temp = genes[pos];
        genes[pos] = genes[pos + 1];
        genes[pos + 1] = temp;
    }

    public void inversion() {
        int pos1 = r.nextInt(genes.length);
        int pos2 = r.nextInt(genes.length);
        if (pos1 > pos2) {
            int temp = pos1;
            pos1 = pos2;
            pos2 = temp;
        }
        Chromosome temp = new Chromosome(this);
        for (int i = pos1; i <= pos2; i++) {
            genes[i] = temp.genes[pos2 - i];
        }
    }

    public Pair crossOnePoint(Chromosome other) {
		int pos = r.nextInt(genes.length);
        int[] genes1 = new int[genes.length];
        int[] genes2 = new int[genes.length];
        for (int i = 0; i < genes.length; i++) {
            if (i < pos) {
                genes1[i] = genes[i];
                genes2[i] = other.genes[i];
            } else {
                genes1[i] = other.genes[i];
                genes2[i] = genes[i];
            }
        }
        return new Pair(new Chromosome(genes1), new Chromosome(genes2));
    }

    public Pair crossTwoPoints(Chromosome other) {
		int pos1 = r.nextInt(genes.length);
		int pos2 = r.nextInt(genes.length);
        if (pos1 > pos2) {
            int temp = pos1;
            pos1 = pos2;
            pos2 = temp;
        }
        int[] genes1 = new int[genes.length];
        int[] genes2 = new int[genes.length];
        for (int i = 0; i < genes.length; i++) {
            if (i < pos1 || i > pos2) {
                genes1[i] = genes[i];
                genes2[i] = other.genes[i];
            } else {
                genes1[i] = other.genes[i];
                genes2[i] = genes[i];
            }
        }
        return new Pair(new Chromosome(genes1), new Chromosome(genes2));
    }

    public Chromosome crossOrder(Chromosome other) {
		int pos1 = r.nextInt(genes.length);
		int pos2 = r.nextInt(genes.length);

        if (pos1 > pos2) {
            int temp = pos1;
            pos1 = pos2;
            pos2 = temp;
        }
        int[] child = new int[genes.length];
        List<Integer> used = new ArrayList<>(genes.length);
        for (int i = pos1; i <= pos2; i++) {
            child[i] = genes[i];
            used.add(genes[i]);
        }

        int index = 0;
        for (int i = 0; i < genes.length; i++) {
            if (index == pos1) {
                index = pos2 + 1;
            }

            if (index == genes.length) {
                index = 0;
            }

            if (!used.contains(other.genes[i])) {
                child[index] = other.genes[i];
                used.add(other.genes[i]);
                index++;
            }
        }

        return new Chromosome(child);
    }

    @Override
    public String toString() {
        StringBuilder res = new StringBuilder("| ");
        for (int gene : genes) {
            res.append(gene);
            res.append(" ");
        }
        res.append("|");
        return res.toString();
    }
}
