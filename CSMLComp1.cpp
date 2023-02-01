//============================================================================
// Name        : CSMLComp1.cpp
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

using namespace std;



double vectSum (vector<double> v){
	double sum = 0.0;

	for (auto& i : v)
		sum += i;

	return sum;

}

double vectAverage(vector<double> v) {
    if (v.empty()) {
        return 0;
    }

    double sum = 0.0;
    for (auto &i: v) {
        sum += (double)i;
    }
    return sum / v.size();
}

double vectMedian(vector<double> v){

	nth_element(v.begin(), v.begin() + v.size()/2, v.end());
	return v[v.size()/2];
}

double vectRange(vector<double> v){
	double min = *min_element(v.begin(), v.end());
	double max = *max_element(v.begin(), v.end());
	return max - min;
}

double vectCovar(vector<double> v, vector<double> m){

	int n = v.size();
	double avgV = vectAverage(v);
	double avgM = vectAverage(m);
	double covar = 0.0;

	for(int i = 0; i < n; i++){
		covar += ((v[i] - avgV)*(m[i] - avgM));
	}
	return (covar / (n-1));
}

double vectCorrel(vector<double> v, vector<double> m){


	    int n = v.size();
	    double avgV = vectAverage(v);
	    double avgM = vectAverage(m);
	    double stddV = 0;
	    double stddM = 0;

	    for (int i = 0; i < n; i++) {
	        stddV += (v[i] - avgV) * (v[i] - avgV);
	        stddM += (m[i] - avgM) * (m[i] - avgM);
	    }
	    stddV = sqrt(stddV / n);
	    stddM = sqrt(stddM / n);



	    return (((vectCovar(v, m)* (n-1))/n) / (stddV*stddM));
}

int main() {
	ifstream inFS;
	string line;
	string rm_in, medv_in;
	const int MAX_LEN = 1000;
	vector<double> rm(MAX_LEN);
	vector<double> medv(MAX_LEN);

	cout << "Opening file Boston.csv."<< endl;

	inFS.open("Boston.csv");
	if(!inFS.is_open()){
		cout << "Could not open file Boston.csv" <<endl;
		return 1;
	}

	cout << "Reading line " << endl;
	getline(inFS, line);

	cout << "heading: " << line << endl;

	int numObservations = 0;
	while (inFS.good()){
		getline(inFS, rm_in, ',');
		getline(inFS, medv_in, '\n');

		rm.at(numObservations)= stof(rm_in);
		medv.at(numObservations) = stof(medv_in);

		numObservations++;
	}

	rm.resize(numObservations);
	medv.resize(numObservations);

	cout << "new length " << rm.size() << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "rm sum: " << vectSum(rm) << endl;
	cout << "rm avg: " << vectAverage(rm) << endl;
	cout << "rm median: " << vectMedian(rm) << endl;
	cout << "rm range: " << vectRange(rm) << endl;
	cout << "-------------------------------------------------------------------------------" << endl;
	cout << "medv sum: " << vectSum(medv) << endl;
    cout << "medv avg: " << vectAverage(medv) << endl;
    cout << "medv median: " << vectMedian(medv) << endl;
    cout << "medv range: " << vectRange(medv) << endl;
    cout << "-------------------------------------------------------------------------------" << endl;
    cout << "covariance between rm and medv: "<<vectCovar(rm, medv)<<endl;
    cout << "correlation between rm and medv: "<<vectCorrel(rm, medv)<<endl;

    cout << "-------------------------------------------------------------------------------" << endl;

	cout << "Closing file: Boston.csv" << endl;
	inFS.close();


}
