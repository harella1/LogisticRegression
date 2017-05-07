#pragma once

using std::vector;
using std::ostream;
using std::cbegin;
using std::cend;
using std::fstream;
using std::string;

namespace Utils {
	template <typename T>
	ostream& operator <<(ostream& out, const vector<T>& input)
	{
		std::for_each(cbegin(input), cend(input), [&](T elem) { out << elem << " ";});
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
		//assert(v1.size() == v2.size());
		//double sum = 0;
		//for (size_t i = 0; i<v1.size(); ++i)
		//{
		//	double minus = v1[i] - v2[i];
		//	double r = minus * minus;
		//	sum += r;
		//}

		//return sqrt(sum);
	}
	template<typename T>
	inline bool load_file(string csv_file, vector<T>& y, matrix& x)
	{
		auto csv = fstream(csv_file);
		char c;
		for (std::string line; std::getline(csv, line); )
		{
			std::replace(line.begin(), line.end(), ',', ' ');
			std::istringstream in(line);
			in >> c;
			if (c == 'M')
				y.push_back(0);
			else
				y.push_back(1);
			x.push_back(
				std::vector<double>(std::istream_iterator<double>(in),
					std::istream_iterator<double>()));
		}

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

