// LogisticRegression.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Utils.h"
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/io.hpp>
//
//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_int.hpp>
//
//// refer to matrix row
//#include <boost/numeric/ublas/matrix_proxy.hpp>
//
//#include "util.hpp"
//#include "data_loader.hpp"


using namespace std;
using namespace Utils;


bool debug = true;
const double e = 2.718281828;

//
double sigmoid(double x) {
	return 1.0 / (1.0 + pow(e, -x));
}


// target: max the gradient of the log-likehood with respect to the kth Teta:
// gra = sum{y(i)-f(teta(k) * x(i)(k)}, where f(x) = 1/1+e**(-x),
// where i denotes the ith training row and k denotes the kth feature.
// Then we know how to update the teta in each iteration:
// teta(k)(t+1) = teta(k)(t) + alpha * gra
void lr_without_regularization(const matrix& x,	const vector<double>& y) {

	// the convergence rate
	double epsilon = 0.00001;
	// the learning rate
	double alpha = 0.00005;
	int max_iters = 2000;
	int iter = 0;

	// init
	vector<double> Teta_k(x[0].size());
	fill(begin(Teta_k), end(Teta_k), 1);

	cout << "old Teta: " << Teta_k << endl;

	vector<double> Teta_k_plus_1(x[0].size());
	fill(begin(Teta_k_plus_1), end(Teta_k_plus_1), 1);

	cout << "new Teta: " << Teta_k_plus_1 << endl;

	while (true) {
		// update each Teta
		double cost = 0;
		for (size_t k = 0; k<Teta_k_plus_1.size(); ++k) {
			double gradient = 0;
			for (size_t i = 0; i<x.size(); ++i) {
				double z_i = 0;
				for (size_t j = 0; j<Teta_k.size(); ++j) {
#if 0
					cout << "x(i,j):" << x(i, j) << endl;
					cout << "weight_old(j):" << weight_old(j) << endl;
#endif
					z_i += Teta_k[j] * x[i][j];
				}
#if 0
				cout << "z_i:" << z_i << endl;
				cout << "y(i):" << y(i) << endl;
				cout << "x(i,k)" << x(i, k) << endl;
				cout << "sigmoid(-y(i) * z_i)" << sigmoid(-y(i) * z_i) << endl;
#endif
				auto sig = sigmoid(z_i);
				gradient = y[i] - x[i][k] * sig;
				cost += pow(sig - y[i], 2);
			}
			Teta_k_plus_1[k] = Teta_k[k] + alpha * gradient;
		}

		double dist = norm(Teta_k_plus_1, Teta_k);
		if (dist < epsilon) {
			cout << "the best weight: " << Teta_k_plus_1 << endl;
			break;
		}
		else {
			Teta_k.swap(Teta_k_plus_1);
			// weight_old = weight_new;
		}

		iter += 1;
		if (iter >= max_iters) {
			cout << "Reach max_iters=" << max_iters << endl;
			break;
		}

		cout << "================================================" << endl;
		cout << "The " << iter << " th iteration, weight:" << endl;
		cout << Teta_k_plus_1 << endl << endl;
		cout << "the diff between the old weight and the new weight: " << dist << endl;
		cout << "the cost of the new weight: " << 0.5*cost << endl ;
	}

	cout << "The best weight:" << endl;
	cout << Teta_k_plus_1 << endl;
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " data_file" << endl;
		return -1;
	}

	const int record_num = 270;
	const int dim_num = 13 + 1;

	vector<double> y;// (record_num);
	matrix x;// (record_num, vector<double>(dim_num));
	load_file(argv[1], y, x);

	// lr_method
	lr_without_regularization(x, y);

	return 0;
}
