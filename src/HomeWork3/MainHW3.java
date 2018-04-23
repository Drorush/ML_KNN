package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;

public class MainHW3 {
	private static int[] bestHyperParameteres;
	private static double bestError;

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		int num_of_folds = 10;
		Instances trainData = loadData("auto_price.txt");
		trainData.randomize(new Random()); // shuffle data
		FeatureScaler fs = new FeatureScaler();
		double currentCombinationAvgError;
		Instances[] arr = null;

		// m = 0 for regular training data , m = 1 for scaled training data
		for (int m = 0; m < 2; m++) {
			bestHyperParameteres = new int[3];
			bestError = Double.MAX_VALUE;
			if (m == 1) {
				trainData = fs.scaleData(trainData);
			}
			arr = divideTrainData(trainData, num_of_folds);
			// iterate over all possible hyperparameters
			for (int k = 1; k <= 20; k++) {
				for (int p = 1; p <= 4; p++) {
					for (int weightingScheme = 0; weightingScheme < 2; weightingScheme++) {
						// checking current combination
						Knn knn = new Knn();
						knn.setK(k);
						knn.setP(p);
						knn.setWeightingScheme(weightingScheme);
						// train X times each time with X-1 subsets
						currentCombinationAvgError = getCurrentCombinationError(arr, knn, num_of_folds, null);
						if (currentCombinationAvgError < bestError) {
							bestError = currentCombinationAvgError;
							bestHyperParameteres[0] = k;
							bestHyperParameteres[1] = p;
							bestHyperParameteres[2] = weightingScheme;
						}
					}
				}
			}
			System.out.println("--------------------------------");
			String dataSet = m == 0 ? "original" : "scaled";
			String majorityFunction = bestHyperParameteres[2] == 0 ? "uniform" : "weighted";
			System.out.println("Results for " + dataSet + " dataset: ");
			System.out.println("--------------------------------");
			System.out.println("Cross validation error with K = " + bestHyperParameteres[0]
					+ ", lp = " + bestHyperParameteres[1] + ", majority function = "
					+ majorityFunction + " for auto_price data is: " + bestError + "\n");
		}

		printMessages(trainData, bestHyperParameteres);

	}

	private static void printMessages(Instances trainData, int[] bestHyperParameteres) throws Exception
	{
		Instances[] arr = null;
		int[] numOfFolds = {trainData.size(), 50, 10, 5, 3};
		int num_of_folds;
		double currentCombinationAvgError;
		Knn knn;
		long[] avgFold = new long[2];

		for(int i = 0; i < numOfFolds.length; i++)
		{
			num_of_folds = numOfFolds[i];
			arr = divideTrainData(trainData,num_of_folds);
			knn = new Knn();
			knn.setK(bestHyperParameteres[0]);
			knn.setP(bestHyperParameteres[1]);
			knn.setWeightingScheme(bestHyperParameteres[2]);
			currentCombinationAvgError = getCurrentCombinationError(arr, knn, num_of_folds, avgFold);
			System.out.println("--------------------------------");
			System.out.println("Results for " + num_of_folds + " folds:");
			System.out.println("--------------------------------");
			System.out.println("Cross validation error of regular knn on auto_price dataset is " + currentCombinationAvgError
					+ " and the average elapsed time is " + avgFold[0]);
			System.out.println("The total elapsed time is: " + avgFold[1] + "\n");
			knn = new Knn();
			knn.setK(bestHyperParameteres[0]);
			knn.setP(bestHyperParameteres[1]);
			knn.setWeightingScheme(bestHyperParameteres[2]);
			knn.distEffCheck = true;
			currentCombinationAvgError = getCurrentCombinationError(arr, knn, num_of_folds, avgFold);
			System.out.println("Cross validation error of efficient knn on auto_price dataset is " + currentCombinationAvgError
					+ " and the average elapsed time is " + avgFold[0]);
			System.out.println("The total elapsed time is: " + avgFold[1] + "\n");
		}

	}

	private static Instances createTrainingData(Instances[] arr, int j)
	{
		Instances data = new Instances(arr[0],0, 0);
		for (int i = 0; i < arr.length; i++)
		{
			if (i != j)
			{
				for (int k = 0; k < arr[i].size(); k++)
				{
					data.add(arr[i].instance(k));
				}
			}
		}

		return data;
	}

	private static Instances[] divideTrainData(Instances data, int foldNumber)
	{
		Instances[] arr = new Instances[foldNumber];
		int partitionSize = data.size() / foldNumber;
		int j = 0;
		for (int i = 0; i < foldNumber; i++)
		{
			arr[i] = new Instances(data, j, partitionSize);
			j += partitionSize;
		}

		// if there is leftovers
		int leftovers = data.size()-j;
		for (int k = 0; k < leftovers; k++)
		{
			arr[k].add(data.instance(j++));
		}

		return arr;
	}

	private static double getCurrentCombinationError(Instances[] arr, Knn knn, int num_of_folds, long[] avgFold) throws Exception
	{

		double currentCombinationError = 0;
		long time;
		long timeOfSingleFold;
		long timeOfAvgFold = 0;
		for (int i = 0; i < num_of_folds; i++)
		{
			Instances validationData = arr[i];
			Instances trainingData = createTrainingData(arr, i);
			knn.buildClassifier(trainingData);
			// get the error of current train
			time = System.nanoTime();
			currentCombinationError += knn.crossValidationError(arr,num_of_folds,i);
			timeOfSingleFold = System.nanoTime() - time;
			timeOfAvgFold += timeOfSingleFold;
		}
		if (avgFold != null)
		{
			avgFold[0] = timeOfAvgFold / num_of_folds;
			avgFold[1] = timeOfAvgFold;
		}

		return currentCombinationError/num_of_folds;
	}

}
