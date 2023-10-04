package Main;

import java.util.Vector;

public class Main {


	public static void main(String[] args) {
		ArtificialIntelligence AI = new ArtificialIntelligence();
		AI.loadWeights("data.txt");

		Vector<Double> input = new Vector<>();
		input.add(0.5);
		input.add(0.75);
		input.add(0.1);

		System.out.println(AI.predict(input));

		ArtificialIntelligence AI2 = new ArtificialIntelligence(3, 5);

		AI2.addLayer(5);
		AI2.addLayer(4);

		System.out.println(AI2.predict(input));
	}
}
