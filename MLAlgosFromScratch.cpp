//============================================================================
// Name        : MLAlgosFromScratch.cpp
// Author      : Dinesh Angadipeta DXA@190032
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

//Passenger struct
struct Passenger {
	double pclass;
	double surv;
	double sex;
	double age;
};

//sigmoid function
double sigmoid(double z) {
	return (1 / (1 + exp(-z)));
}

//this is the function that is used to transpose the 2d vector
vector<vector<double>> transpose(const vector<vector<double>> &matrix) {
	vector<vector<double>> result(matrix[0].size(),
			vector<double>(matrix.size()));
	for (size_t i = 0; i < matrix.size(); ++i) {
		for (size_t j = 0; j < matrix[0].size(); ++j) {
			result[j][i] = matrix[i][j];
		}
	}
	return result;
}

int main() {
	//default length
	const int MAX_LEN = 10000;

	//vector for all the passengers
	vector<Passenger> data(MAX_LEN);

	ifstream inFS;
	string line;
	//used to read in data
	string rowno_in, pclass_in, surv_in, sex_in, age_in;

	//will be used to transfer data to main vector
	Passenger holder;

	inFS.open("titanic_project.csv");
	if (!inFS.is_open()) {

		return 1;
	}

	//Used to skip first line
	getline(inFS, line);

	int numObservations = 0;
	while (inFS.good()) {
		//used to get first value which will be ignored.
		getline(inFS, rowno_in, ',');

		getline(inFS, pclass_in, ',');
		holder.pclass = stod(pclass_in);
		getline(inFS, surv_in, ',');
		holder.surv = stod(surv_in);
		getline(inFS, sex_in, ',');
		holder.sex = stod(sex_in);
		getline(inFS, age_in, '\n');
		holder.age = stod(age_in);

		data[numObservations] = holder;

		numObservations++;
	}

	data.resize(numObservations);

	inFS.close();

	//this is where the training and test data is created.
	vector<Passenger> train(data.begin(), data.begin() + 800);
	vector<Passenger> test(data.begin() + 800, data.end());
	//learning rate
	double learning_rate = 0.001;

	vector<double> weights(2);

	vector<vector<double>> data_matrix(train.size());
	vector<double> labels(train.size());

	//this is where the initial weights are set.
	//this is used to measure the time elapsed in the program.
	auto start = high_resolution_clock::now();
	weights[0] = 1;
	weights[1] = 1;
	//this is where the data matrix and labels are defined with the survived and sex data.
	for (unsigned int i = 0; i < train.size(); i++) {

		data_matrix[i] = vector<double>(2);
		data_matrix[i][0] = 1;
		data_matrix[i][1] = train[i].sex;
		labels[i] = train[i].surv;

	}
	//this holds the transposed data matrix
	vector<vector<double>> t_data_matrix = transpose(data_matrix);

	vector<double> prob(train.size());
	vector<double> error(train.size());
	vector<double> wAdder(weights.size());

	/*for(i in 1:500000){
	 prob_vector <- sigmoid(data_matrix %*% weights)
	 error <- labels- prob_vector
	 weights <- weights + learning_rate * t(data_matrix) %*% error
	 }*/

	for (int j = 0; j < 500000; j++) {

		for (unsigned int k = 0; k < train.size(); k++) {
			double res = (data_matrix[k][0] * weights[0])
					+ (data_matrix[k][1] * weights[1]);

			prob[k] = (sigmoid(res));

		}

		for (unsigned int l = 0; l < train.size(); l++) {
			error[l] = (labels[l] - prob[l]);

		}

		for (unsigned int m = 0; m < weights.size(); m++) {
			double wAdderHolder = 0.0;
			for (unsigned int n = 0; n < train.size(); n++) {
				wAdderHolder += t_data_matrix[m][n] * error[n];
			}
			wAdder[m] = wAdderHolder;
			wAdderHolder = 0.0;
		}

		for (unsigned int o = 0; o < weights.size(); o++) {
			weights[o] = (weights[o] + (learning_rate * wAdder[o]));

		}

	}

	cout
			<< "coefficients for logistic regression for predicting survived based on sex:"
			<< endl;
	cout << "w0: " << weights[0] << endl;
	cout << "w1: " << weights[1] << endl;
	//This prints out the time used in training.
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	cout << "Training time of the algorithm: " << duration.count()
			<< " seconds" << endl;

	//This is there the testing data is predicted and analyzed to created the metrics to print out.
	vector<vector<double>> test_matrix(test.size());
	vector<double> test_labels(test.size());
	vector<double> predicted(test.size());
	vector<double> probabilities(test.size());
	vector<double> predictions(test.size());
	vector<double> meanHolder(test.size());
	double TP = 0;
	double FP = 0;
	double TN = 0;
	double FN = 0;

	for (unsigned int i = 0; i < test.size(); i++) {

		test_matrix[i] = vector<double>(2);
		test_matrix[i][0] = 1;
		test_matrix[i][1] = test[i].sex;
		test_labels[i] = test[i].surv;

	}

	for (unsigned int k = 0; k < test.size(); k++) {
		double resPred = (test_matrix[k][0] * weights[0])
				+ (test_matrix[k][1] * weights[1]);
		predicted[k] = resPred;

	}

	for (unsigned int r = 0; r < test.size(); r++) {
		probabilities[r] = (exp(predicted[r])) / (1 + exp(predicted[r]));
	}

	for (unsigned int e = 0; e < test.size(); e++) {
		if (probabilities[e] > 0.5) {
			predictions[e] = 1;
		} else {
			predictions[e] = 0;
		}
	}

	for (unsigned int q = 0; q < test.size(); q++) {
		if (predictions[q] == test_labels[q]) {
			meanHolder[q] = 1;
		} else {
			meanHolder[q] = 0;
		}
	}
	//This is where the predictions are compared with the labels to find TP, TN, FN, and FP.
	for (unsigned int w = 0; w < test.size(); w++) {
		if (predictions[w] == 0 && test_labels[w] == 0) {
			TP++;
		} else if (predictions[w] == 1 && test_labels[w] == 1) {
			TN++;
		} else if (predictions[w] == 0 && test_labels[w] == 1) {
			FN++;
		} else {
			FP++;
		}
	}

	//This accuracy, sensitivity, and specificity are all calculated here.
	cout << "accuracy: " << ((TP + TN) / (TP + TN + FP + FN)) << endl;
	cout << "sensitivity: " << ((TP) / (TP + FN)) << endl;
	cout << "specificity: " << ((TN) / (TN + FP)) << endl;
	cout << endl;

	//Everything is repeated from here on out for the different predictors.

	start = high_resolution_clock::now();
	//this is where the initial weights are set.
	weights[0] = 1;
	weights[1] = 1;
	//this is where the data matrix and labels are defined with the survived and age data.
	for (unsigned int i = 0; i < train.size(); i++) {

		data_matrix[i] = vector<double>(2);
		data_matrix[i][0] = 1;
		data_matrix[i][1] = train[i].age;
		labels[i] = train[i].surv;

	}
	//this holds the transposed data matrix
	t_data_matrix = transpose(data_matrix);

	prob.clear();
	error.clear();
	wAdder.clear();

	/*for(i in 1:500000){
	 prob_vector <- sigmoid(data_matrix %*% weights)
	 error <- labels- prob_vector
	 weights <- weights + learning_rate * t(data_matrix) %*% error
	 }*/

	for (int j = 0; j < 500000; j++) {

		for (unsigned int k = 0; k < train.size(); k++) {
			double res = (data_matrix[k][0] * weights[0])
					+ (data_matrix[k][1] * weights[1]);

			prob[k] = (sigmoid(res));

		}

		for (unsigned int l = 0; l < train.size(); l++) {
			error[l] = (labels[l] - prob[l]);

		}

		for (unsigned int m = 0; m < weights.size(); m++) {
			double wAdderHolder = 0.0;
			for (unsigned int n = 0; n < train.size(); n++) {
				wAdderHolder += t_data_matrix[m][n] * error[n];
			}
			wAdder[m] = wAdderHolder;
			wAdderHolder = 0.0;
		}

		for (unsigned int o = 0; o < weights.size(); o++) {
			weights[o] = (weights[o] + (learning_rate * wAdder[o]));

		}

	}

	cout
			<< "coefficients for logistic regression for predicting survived based on age:"
			<< endl;
	cout << "w0: " << weights[0] << endl;
	cout << "w1: " << weights[1] << endl;

	stop = high_resolution_clock::now();
	duration = duration_cast<seconds>(stop - start);
	cout << "Training time of the algorithm: " << duration.count()
			<< " seconds" << endl;

	test_matrix.clear();
	test_labels.clear();
	predicted.clear();
	probabilities.clear();
	predictions.clear();
	meanHolder.clear();
	TP = 0;
	FP = 0;
	TN = 0;
	FN = 0;

	for (unsigned int i = 0; i < test.size(); i++) {

		test_matrix[i] = vector<double>(2);
		test_matrix[i][0] = 1;
		test_matrix[i][1] = test[i].age;
		test_labels[i] = test[i].surv;

	}

	for (unsigned int k = 0; k < test.size(); k++) {
		double resPred = (test_matrix[k][0] * weights[0])
				+ (test_matrix[k][1] * weights[1]);
		predicted[k] = resPred;

	}

	for (unsigned int r = 0; r < test.size(); r++) {
		probabilities[r] = (exp(predicted[r])) / (1 + exp(predicted[r]));
	}

	for (unsigned int e = 0; e < test.size(); e++) {
		if (probabilities[e] > 0.5) {
			predictions[e] = 1;
		} else {
			predictions[e] = 0;
		}
	}

	for (unsigned int q = 0; q < test.size(); q++) {
		if (predictions[q] == test_labels[q]) {
			meanHolder[q] = 1;
		} else {
			meanHolder[q] = 0;
		}
	}
	for (unsigned int w = 0; w < test.size(); w++) {
		if (predictions[w] == 0 && test_labels[w] == 0) {
			TP++;
		} else if (predictions[w] == 1 && test_labels[w] == 1) {
			TN++;
		} else if (predictions[w] == 0 && test_labels[w] == 1) {
			FN++;
		} else {
			FP++;
		}
	}

	cout << "accuracy: " << ((TP + TN) / (TP + TN + FP + FN)) << endl;
	cout << "sensitivity: " << ((TP) / (TP + FN)) << endl;
	cout << "specificity: " << ((TN) / (TN + FP)) << endl;
	cout << endl;

	start = high_resolution_clock::now();
	//this is where the initial weights are set.
	weights[0] = 1;
	weights[1] = 1;
	//this is where the data matrix and labels are defined with the survived and passenger class data.
	for (unsigned int i = 0; i < train.size(); i++) {

		data_matrix[i] = vector<double>(2);
		data_matrix[i][0] = 1;
		data_matrix[i][1] = train[i].pclass;
		labels[i] = train[i].surv;

	}
	//this holds the transposed data matrix
	t_data_matrix = transpose(data_matrix);

	prob.clear();
	error.clear();
	wAdder.clear();

	/*for(i in 1:500000){
	 prob_vector <- sigmoid(data_matrix %*% weights)
	 error <- labels- prob_vector
	 weights <- weights + learning_rate * t(data_matrix) %*% error
	 }*/

	for (int j = 0; j < 500000; j++) {

		for (unsigned int k = 0; k < train.size(); k++) {
			double res = (data_matrix[k][0] * weights[0])
					+ (data_matrix[k][1] * weights[1]);

			prob[k] = (sigmoid(res));

		}

		for (unsigned int l = 0; l < train.size(); l++) {
			error[l] = (labels[l] - prob[l]);

		}

		for (unsigned int m = 0; m < weights.size(); m++) {
			double wAdderHolder = 0.0;
			for (unsigned int n = 0; n < train.size(); n++) {
				wAdderHolder += t_data_matrix[m][n] * error[n];
			}
			wAdder[m] = wAdderHolder;
			wAdderHolder = 0.0;
		}

		for (unsigned int o = 0; o < weights.size(); o++) {
			weights[o] = (weights[o] + (learning_rate * wAdder[o]));

		}

	}

	cout
			<< "coefficients for logistic regression for predicting survived based on the passenger class:"
			<< endl;
	cout << "w0: " << weights[0] << endl;
	cout << "w1: " << weights[1] << endl;

	stop = high_resolution_clock::now();
	duration = duration_cast<seconds>(stop - start);
	cout << "Training time of the algorithm: " << duration.count()
			<< " seconds" << endl;

	test_matrix.clear();
	test_labels.clear();
	predicted.clear();
	probabilities.clear();
	predictions.clear();
	meanHolder.clear();
	TP = 0;
	FP = 0;
	TN = 0;
	FN = 0;

	for (unsigned int i = 0; i < test.size(); i++) {

		test_matrix[i] = vector<double>(2);
		test_matrix[i][0] = 1;
		test_matrix[i][1] = test[i].pclass;
		test_labels[i] = test[i].surv;

	}

	for (unsigned int k = 0; k < test.size(); k++) {
		double resPred = (test_matrix[k][0] * weights[0])
				+ (test_matrix[k][1] * weights[1]);
		predicted[k] = resPred;

	}

	for (unsigned int r = 0; r < test.size(); r++) {
		probabilities[r] = (exp(predicted[r])) / (1 + exp(predicted[r]));
	}

	for (unsigned int e = 0; e < test.size(); e++) {
		if (probabilities[e] > 0.5) {
			predictions[e] = 1;
		} else {
			predictions[e] = 0;
		}
	}

	for (unsigned int q = 0; q < test.size(); q++) {
		if (predictions[q] == test_labels[q]) {
			meanHolder[q] = 1;
		} else {
			meanHolder[q] = 0;
		}
	}
	for (unsigned int w = 0; w < test.size(); w++) {
		if (predictions[w] == 0 && test_labels[w] == 0) {
			TP++;
		} else if (predictions[w] == 1 && test_labels[w] == 1) {
			TN++;
		} else if (predictions[w] == 0 && test_labels[w] == 1) {
			FN++;
		} else {
			FP++;
		}
	}

	cout << "accuracy: " << ((TP + TN) / (TP + TN + FP + FN)) << endl;
	cout << "sensitivity: " << ((TP) / (TP + FN)) << endl;
	cout << "specificity: " << ((TN) / (TN + FP)) << endl;
	cout << endl;

	return 0;
}
