package cm.ia.main;



import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;

public class TestDl4j {

	public static void main(String[] args) throws IOException, InterruptedException {
		
		double learningRate=0.001;
		int numInputs=4;
		int numHiddenNodes=10;
		int numOutput=3;
		int batchSize=1;
		int classIndex=4;
		int numEpochs=100;
		
		
		//Configuration du Modele:
		MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
				.updater(new Adam(learningRate))
				.list()
					.layer(0,new DenseLayer.Builder()
							.nIn(numInputs)
							.nOut(numHiddenNodes)
							.activation(Activation.SIGMOID).build())
					
					.layer(1,new OutputLayer.Builder()
							.nIn(numHiddenNodes)
							.nOut(numOutput)
							.activation(Activation.SOFTMAX)
							.lossFunction(LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
							.build())
				.build();
		//System.out.println(configuration.toJson());
		
		//Creation du Modele:
		MultiLayerNetwork model = new MultiLayerNetwork(configuration);
		model.init();
		
		//Entrainement du Modele:
		File fileEntrainement= new ClassPathResource("entrainement.csv").getFile();
		RecordReader recordReaderTrain = new CSVRecordReader();
		recordReaderTrain.initialize(new FileSplit(fileEntrainement));
		DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,numOutput);
		
		//Démarage du serveur de Monitoring du processus d'apprentissage
		UIServer uiServer = UIServer.getInstance();
		InMemoryStatsStorage statsStorage = new InMemoryStatsStorage(); 
		uiServer.attach(statsStorage);
		model.setListeners(new StatsListener(statsStorage));
		
		for (int i = 0; i < numEpochs; i++) {
			model.fit(dataSetIteratorTrain);
		}
		
		
		System.out.println("Evaluation");
		
		//Evaluation du Modele:
		File fileTest= new ClassPathResource("test.csv").getFile();
		RecordReader recordReaderTest = new CSVRecordReader();
		recordReaderTest.initialize(new FileSplit(fileTest));
		DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,numOutput);
		
		Evaluation evaluation = new Evaluation(numOutput);
		while(dataSetIteratorTest.hasNext()) {
			DataSet dataset = dataSetIteratorTest.next();
			INDArray features = dataset.getFeatures();
			INDArray labels = dataset.getLabels();
			INDArray predicted = model.output(features);
			evaluation.eval(labels, predicted);
		}
		
		System.out.println(evaluation.stats());
		
		ModelSerializer.writeModel(model,"irisModel.zip",true);
		
		
	}

}
