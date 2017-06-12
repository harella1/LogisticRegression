#pragma once

using std::vector;
using std::ostream;
using std::cbegin;
using std::cend;
using std::fstream;
using std::string;

namespace Utils {
	template <typename T>
	inline ostream& operator <<(ostream& out, const vector<T>& input)
	{
		std::for_each(cbegin(input), cend(input), [&](T elem) { out << elem << ",";});
		return out;
	}

	template <>
	inline ostream& operator <<(ostream& out, const matrix& input)
	{
		for (auto& vec : input)
			out << vec << "\n";
		return out;
	}


	template <typename T>
	T norm(const vector<T>& v1, const vector<T>& v2)
	{
		T sum = std::inner_product(cbegin(v1), cend(v1), cbegin(v2), 0.0, std::plus<T>(), [&](auto elem1, auto elem2)
		{
			double minus = elem1 - elem2;
			return minus * minus;
		});
		return sqrt(sum);
	}

	inline vector<double> col(const matrix& x, int column)
	{
		vector<double> res(x.size());
		if (column <= x.size())
			for (size_t i = 0; i < x.size(); i++)
				res[i] = x[i][column];
		return res;
	}

	inline matrix transpose(const matrix& x)
	{
		matrix x_T(x[0].size());
		std::fill(begin(x_T), end(x_T), vector<double>(x.size()));
		for (size_t i = 0; i < x.size(); i++)
			for (size_t j = 0; j < x[i].size(); j++)
				x_T[j][i] = x[i][j];
		return x_T;

	}

	inline void normalize_data(matrix& x)
	{
		auto x_T = transpose(x);

		vector<std::pair<double, double>> min_max(x[0].size());
//		std::fill(begin(min_max), end(min_max), std::make_pair(0, 0));
		for (size_t i = 1; i < x_T.size(); i++)
		{
			auto x = minmax_element(cbegin(x_T[i]), cend(x_T[i]));
			min_max[i] = std::make_pair(*x.first, *x.second);
		}

		for (size_t i = 0; i < x.size(); i++)
			for (size_t j = 1; j < x[i].size(); j++)
				x[i][j] = (x[i][j]-min_max[j].first)/(min_max[j].second - min_max[j].first);

	}

	template<typename T>
	inline bool load_file(string csv_file, vector<T>& y, matrix& x)
	{
		auto csv = fstream(csv_file);
		char c;
		int index = 1;
		for (std::string line; std::getline(csv, line); )
		{
			std::replace(line.begin(), line.end(), ',', ' ');
			std::stringstream in(line);
			in >> c;
			if (c == 'M')
				y.push_back(0);
			else
				y.push_back(1);
			std::stringstream in2;
			in2 << "1 " << in.rdbuf();
			x.push_back(
				std::vector<double>(std::istream_iterator<double>(in2),
					std::istream_iterator<double>()));
		}
		normalize_data(x);

		//csv.clear();
		//csv.seekg(0, ios::beg);

		//vector<char> buff(1024*1024);
		//long long read_count = 1;
		//while (read_count>0)
		//{
		//	csv.read(&buff[0], buff.size());
		//	read_count = csv.gcount();


		//}
		return true;
	}

}

