#ifndef FUNC_HPP
#define FUNC_HPP

#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

//Global min: f(0) = 0
//x_i within [-5.12, 5.12]
double rastriginFunction(const vector<double>& x) {
    double A = 10.0;
    double sum = A * x.size();
    for (double xi : x) {
        xi = std::clamp(xi, -5.12, 5.12);
        sum += (xi * xi - A * cos(2 * M_PI * xi));
    }
    return sum;
}

//Commonly use a = 1, b = 100
//Global min: f(a, a^2, a^3, ..., a^n) = 0
double rosenbrockFunctionGeneralized(const std::vector<double>& x, double a, double b) {
    double total = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double term1 = std::pow(a - x[i], 2);
        double term2 = b * std::pow(x[i+1] - x[i]*x[i], 2);
        total += term1 + term2;
    }
    return total;
}

//Global min: f(1, 1, ..., 1) = 0
//N within [4,7] has another local min at (-1, 1, ..., 1)
double rosenbrockFunction(const std::vector<double>& x) {
    return rosenbrockFunctionGeneralized(x, 1.0, 100.0);
}

//Global min: f(0, 0, ..., 0) = 0
double sphereFunction(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

/**
 * Calculates the Michalewicz function.
 * @param x The input vector (coordinates). x_i within [0, PI]
 * @param m The steepness parameter (default is 10).
 * @return The value of the function at point x.
 * Global min With dimension d: 
 * d = 2: f(2.20, 1.57) ≈ -1.8013
 * d = 5: min = -4.687658
 * d = 10: min = -9.66015
 * d = 20: min = -19.6370
 * d = 30: min = -29.6309
 * d = 50: min = -49.6248
 * 
 * 
 * https://www.sfu.ca/~ssurjano/michal.html
 * https://doi.org/10.48550/arXiv.2001.11465
 */
double michalewiczFunction(const std::vector<double>& x, int m = 10) {
    double sum = 0.0;
    int d = x.size();

    for (int i = 0; i < d; ++i) {
        double xi = std::clamp(x[i], 0.0, M_PI);
        double term = std::sin(xi) * std::pow(std::sin(((i+1) * xi * xi) / M_PI), 2 * m);
        sum += term;
    }

    return -sum;
}

//Global Minimum with d dimension: f(-2.903534, -2.903534, ..., -2.903534) ≈ -39.16599*d
//x_i within [-5, 5]
double styblinskiTangFunction(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        val = std::clamp(val, -5.0, 5.0);
        //(x^4 - 16x^2 + 5x)
        sum += (std::pow(val, 4) - 16.0 * std::pow(val, 2) + 5.0 * val);
    }
    return 0.5 * sum;
}

//Global Minimum with d dimension: f(0, 0, ..., 0) = 0
//x_i within [-600, 600]
double griewankFunction(const std::vector<double>& x) {
    double sum = 0.0;
    double product = 1.0;
    int d = x.size();

    for (int i = 0; i < d; ++i) {
        sum += (x[i] * x[i]) / 4000.0;
        product *= std::cos(x[i] / std::sqrt(i + 1));
    }

    return sum - product + 1.0;
}

//x_i within [-32.768, 32.768]
//Global min: f(0, 0, ..., 0) = 0
double ackleyFunctionGeneralized(const std::vector<double>& x, double a = 20.0, double b = 0.2, double c = 2.0 * M_PI) {
    int d = x.size();
    if (d == 0) return 0.0;

    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < d; ++i) {
        double x_i = std::clamp(x[i], -32.768, 32.768);
        sum1 += x_i * x_i;
        sum2 += std::cos(c * x_i);
    }

    double term1 = -a * std::exp(-b * std::sqrt(sum1 / d));
    double term2 = -std::exp(sum2 / d);

    return term1 + term2 + a + std::exp(1.0);
}

//Commonly use a = 20, b = 0.2, c = 2 * PI
double ackleyFunction(const std::vector<double>& x) {
    return ackleyFunctionGeneralized(x, 20.0, 0.2, 2.0 * M_PI);
}

double targetFunction(const vector<double>& x) {
    return rastriginFunction(x);
}

#endif // FUNC_HPP