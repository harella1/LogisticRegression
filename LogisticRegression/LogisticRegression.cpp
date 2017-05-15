// LogisticRegression.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Utils.h"


using namespace std;
using namespace Utils;


bool debug = true;
const double e = 2.718281828;
// the convergence rate
const double epsilon = 0.00001;
// the learning rate
const double alpha = 0.5;

const int training_size = 300;

//
double sigmoid(double x) {
	return 1.0 / (1.0 + pow(e, -x));
}


vector<double> h_teta_x(const matrix& x, const vector<double>& teta)
{
	vector<double> res(teta.size());
	for (size_t i = 0; i < teta.size(); i++)
		res[i] = sigmoid(inner_product(cbegin(teta), cend(teta), cbegin(x[i]), 0.0));
	return res;
}
//cost function
// J(teta) = 1/m * sum( y(i)*log(sig(sum(teta(j)*x(i,j)) + (1-y(i)*log(1-sig(sum(teta(j)*x(i,j)) )
double cost(const vector<double>& Teta, const vector<double>& y, const matrix& x)
{
	vector<double> diffs (y.size());
	for (size_t i = 0; i < y.size(); i++)
	{
		auto z_i = inner_product(cbegin(Teta), cend(Teta), cbegin(x[i]), 0.0);
		double z_i2 = 0.0;
		//for (size_t j = 0; j<Teta.size(); ++j) 
		//	z_i2 += Teta[j] * x[i][j];
		auto sig = sigmoid(z_i);
		auto first = -y[i] * log(sig+epsilon);
		auto second = (1-y[i]) * log(1- sig + epsilon);
		diffs[i] = first - second;
	}
	double sum = 0.0;
	for (auto x : diffs) sum += x;
	auto res = sum / y.size();
	return res;
}





// target: max the gradient of the log-likehood with respect to the kth Teta:
// gra = sum{y(i)-sig(teta(k) * x(i)(k)}, where sig(x) = 1/1+e**(-x),
// where i denotes the ith training row and k denotes the kth feature.
// Then we know how to update the teta in each iteration:
// teta(k)(t+1) = teta(k)(t) + alpha * gra
void lr_without_regularization(const matrix& x,	const vector<double>& y) {

	int max_iters = 2000;
	int iter = 0;

	// init
	vector<double> Teta_k(x[0].size());
	fill(begin(Teta_k), end(Teta_k), 0);

	//cout << "new Teta: " << Teta_k_plus_1 << endl;
	auto step_cost = cost(Teta_k, y, x);
	cout << "the cost of the first step: " << step_cost << endl;

	while (true) {
		// update each Teta
		for (size_t k = 0; k<Teta_k.size(); ++k) {
			double gradient = 0;
			for (size_t i = 0; i<training_size; ++i) {
				//auto z_i = inner_product(cbegin(Teta_k), cend(Teta_k), cbegin(x[i]), 0.0);
				double z_i = 0;
				for (size_t j = 0; j<Teta_k.size(); ++j) {
					z_i += Teta_k[j] * x[i][j];
				}

				auto sig = sigmoid(z_i);
				gradient += (sig - y[i])*x[i][k];
			}
			gradient /= x.size();
			Teta_k[k] = Teta_k[k] + alpha * gradient;
		}

		iter += 1;
		if (iter >= max_iters) {
			cout << "Reach max_iters=" << max_iters << endl;
			break;
		}
		auto step_cost = cost(Teta_k, y, x);
		cout << "================================================" << endl;
		cout << "The " << iter << " th iteration, weight:" << endl;
		cout << Teta_k << endl << endl;
		//cout << "the diff between the old weight and the new weight: " << dist << endl;
		cout << "the cost of the new step: " << step_cost << endl ;
	}

	cout << "The best weight:" << endl;
	cout << Teta_k << endl;
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " data_file" << endl;
		return -1;
	}

	vector<double> y;// (record_num);
	matrix x;// (record_num, vector<double>(dim_num));
	load_file(argv[1], y, x);

	// lr_method
	lr_without_regularization(x, y);

	return 0;
}
