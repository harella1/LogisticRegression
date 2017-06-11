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

const int training_size = 300;
const int max_iters = 500;
matrix results;

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
	double sum = 0.0;
	for (size_t i = 0; i < y.size(); i++)
	{
		double z_i2 = 0.0;
		for (size_t j = 0; j<Teta.size(); ++j) 
			z_i2 += Teta[j] * x[i][j];
		auto sig = sigmoid(z_i2);
		sum += y[i] == 1 ? log(sig+epsilon) : log(1- sig + epsilon);
	}
	auto res = -sum / y.size();
	return res;
}

double predict(const vector<double>& Teta, const matrix& x, const vector<double>& y, size_t start, size_t end) {
	if (end < start) return 0;
	auto matches = 0.0;
	for (size_t i = start; i < end; i++)
	{
		double z_i2 = 0.0;
		for (size_t j = 0; j < Teta.size(); ++j)
			z_i2 += Teta[j] * x[i][j];
		auto sig = sigmoid(z_i2);
		if ((sig >= 0.5) && y[i] == 1 || sig < 0.5 && y[i] == 0)
			matches++;
	}
	return matches / (end - start);

}

// target: max the gradient of the log-likehood with respect to the kth Teta:
// gra = sum{y(i)-sig(teta(k) * x(i)(k)}, where sig(x) = 1/1+e**(-x),
// where i denotes the ith training row and k denotes the kth feature.
// Then we know how to update the teta in each iteration:
// teta(k)(t+1) = teta(k)(t) + alpha * gra
const vector<double> lr_without_regularization(const matrix& x,	const vector<double>& y, double alpha) {
	vector<double> costs(max_iters+3);

	int iter = 0;

	// init
	vector<double> Teta_k(x[0].size());
	fill(begin(Teta_k), end(Teta_k), 0);

	//cout << "new Teta: " << Teta_k_plus_1 << endl;
	auto step_cost = cost(Teta_k, y, x);
	//cout << "the cost of the first step: " << step_cost << endl;

	while (true) {
		// update each Teta
		for (size_t k = 0; k<Teta_k.size(); ++k) {
			double gradient = 0;
			for (size_t i = 0; i<training_size; ++i) {
				double z_i = 0;
				for (size_t j = 0; j<Teta_k.size(); ++j) {
					z_i += Teta_k[j] * x[i][j];
				}

				auto sig = sigmoid(z_i);
				gradient += (y[i] -sig)*x[i][k];
			}
			gradient /= x.size();
			Teta_k[k] = Teta_k[k] + alpha * gradient;
		}
		auto cur_cost = cost(Teta_k, y, x);
		auto dist = step_cost - cur_cost;
		step_cost = cur_cost;
		costs[iter+3] = step_cost;
		iter += 1;
		if (iter >= max_iters) {
			//cout << "Reach max_iters=" << max_iters << endl;
			break;
		}

		//cout << "================================================" << endl;
		//cout << "The " << iter << " th iteration, cost:" << step_cost << endl;
		////cout << Teta_k << endl << endl;
		//cout << "the diff : " << std::fixed << std::setw(11)
		//	<< std::setprecision(6) <<dist << endl;
		//cout << "the cost of the new step: "  endl ;
	}
	auto error1 = predict(Teta_k, x, y, training_size, y.size());
	auto error2 = predict(Teta_k, x, y, 0, training_size);
	costs[0] = alpha;
	costs[1] = error1;
	costs[2] = error2;

	//cout << "Error on test input:" << error1 << endl;
	//cout << "Error on train input:" << error2 << endl;
	//cout << Teta_k << endl;
	return costs;
}


void select_significant_features(const matrix& x, const vector<double>& y, int num_of_features) {

	auto x_T = transpose(x);

	matrix f;
	f.push_back(x_T[0]);
	for (size_t i = 1; i <= num_of_features; i++)
	{
		f.resize(i + 1);
		auto t1 = chrono::steady_clock::now();
		for (size_t j = 1; j < x_T.size(); j++)
		{
			f[i] = x_T[j];
			auto f_vec = transpose(f);
			results.push_back(lr_without_regularization(f_vec, y, 0.8));
		}
		cout << "duration: " << std::chrono::duration<double>(chrono::steady_clock::now() - t1).count() << " sec\n";

		auto score = col(results, 1);
		auto best_feature = std::max_element(cbegin(score),cend(score)) - cbegin(score);

		std::swap(x_T[best_feature], x_T.back());
		f[i] = x_T.back();
		x_T.pop_back();
		if(i < num_of_features) results._Pop_back_n(x_T.size());
		cout << "selected feature " << best_feature+i  << "\n";
	}
}

int main(int argc, char* argv[]) {
	system("wmic cpu get name");
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " data_file" << endl;
		return -1;
	}

	vector<double> y;
	matrix x;
	load_file(argv[1], y, x);

	results.push_back(vector<double>(max_iters+3));
	int i = -2;
	generate(begin(results[0]), end(results[0]), [&] {return i++; });

	// the learning rate
	//double alpha = 2.5;
	//// lr_method
	//cout << "Iterations for each test: " << max_iters << "\n";
	//for (; alpha > 0.5; alpha-=0.3)
	//{
	//	cout << "alpha: " << alpha << endl;
	//	auto t1 = chrono::steady_clock::now();
	//	results.push_back( lr_without_regularization(x, y, alpha));
	//	auto t2 = chrono::steady_clock::now();
	//	cout << "duration: " << std::chrono::duration<double>( t2 - t1).count() << " sec\n";
	//}

	select_significant_features(x, y, 5);
	ofstream output("output.csv");
	output << results << "\n";

	return 0;
}
