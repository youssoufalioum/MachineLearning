package cm.ia.main;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Prediction {

	public static void main(String[] args) throws IOException {
	
		MultiLayerNetwork model=ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));

		String[] labelsName={"Iris-setosa","Iris-versicolor","Iris-virginica"};
		
		System.out.println("Prediction");	
		INDArray inputData=Nd4j.create(new double[][] {
			{5.1,3.5,1.4,0.2},
			{3.1,3.5,2.4,4.2},
			{4.1,2.5,1.5,1.2},
			{2.1,3.5,3.4,0.5},
			{3.1,3.5,1.6,2.2},
			{1.3,4.8,1.4,1.2}
		});
		
		INDArray output=model.output(inputData);
		
		int[] classe=output.argMax(1).toIntVector();
		
		System.out.println(output);
		
		for (int i = 0; i < classe.length; i++) {
			System.out.println("Classe: "+labelsName[classe[i]]);
		}

	}

}
