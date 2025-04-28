#include <vector>
#include <random>
#include <iostream>
#include <array>
#include <algorithm>
#include <ctime>
#include <tuple>
#include <iomanip>

#include "helper.hpp"

void get_random_weights(std::vector<double> &weights, size_t seed)
{
    std::mt19937 e(seed);
    std::uniform_real_distribution<double> u;
    for (double &val : weights)
    {
        val = u(e);
    }
}

static inline double dot_product(const std::vector<double> &x, const std::vector<double> &w)
{
    if (x.size() != w.size())
    {
        std::cerr << "\033[32minputs and weights size do not match\n";
        return NAN;
    }

    double h = 0.0;
    for (size_t i = 0; i != x.size(); ++i)
    {
        h += x[i] * w[i];
    }

    return h;
}

std::vector<double> predict(const std::vector<std::vector<double>> &X, const std::vector<double> &w, double bias = 0)
{
    std::vector<double> y_pred;
    y_pred.reserve(X.size());
    for (const std::vector<double> &x : X)
    {
        if (x.size() != w.size())
        {
            std::cerr << "Predict: vector x and w have differnt sizes\n";
            return std::vector<double>{};
        }

        double h = std::inner_product(x.begin(), x.end(), w.begin(), bias);
        y_pred.push_back(h);
    }

    return y_pred;
}

double cost_function(
    const std::vector<std::vector<double>> &X,
    const std::vector<double> &y,
    const std::vector<double> &w,
    double b)
{
    std::vector<double> y_pred = predict(X, w, b);
    if (y_pred.size() != y.size())
    {
        std::cerr << "cost_function: y and y_pred sizes do not match\n";
        return NAN;
    }
    double cost = 0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        try
        {
            cost += pow(y_pred.at(i) - y.at(i), 2);
        }
        catch (const std::out_of_range &e)
        {
            std::cerr << "index: " << i
                      << " y_pred size: " << y_pred.size()
                      << " y size: " << y.size() << "\n";
            std::cerr << e.what() << '\n';
            return NAN;
        }
    }

    return cost / (2.0 * y.size());
}

void linear_regression(
    const std::vector<std::vector<double>> &X,
    const std::vector<double> &y,
    std::vector<double> &weights,
    double &bias,
    size_t n_iter = 1000,
    double learning_rate = 0.01,
    size_t seed = 42)
{
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    weights.resize(n_features);
    if (y.size() != n_samples)
    {
        std::cerr << "Targets and inputs size must match\n";
        return;
    }

    for (const std::vector<double> &x_vec : X)
    {
        if (x_vec.size() != n_features)
        {
            std::cerr << "X must have same column sizes\n";
            return;
        }
    }

    std::mt19937 e(time(0));
    std::uniform_real_distribution<double> u;
    for (double &w : weights)
    {
        w = u(e);
    }
    bias = u(e);

    std::vector<double> w_gradients(n_features, 0);
    double b_gradient = 0.0;
    for (size_t i = 0; i < n_iter; ++i)
    {
        for (size_t j = 0; j < n_samples; ++j)
        {
            double h = dot_product(X[j], weights) + bias;
            b_gradient += h - y[j];
            for (size_t k = 0; k < w_gradients.size(); ++k)
            {
                w_gradients[k] += (h - y[j]) * X[j][k];
            }
        }

        for (size_t k = 0; k < weights.size(); ++k)
        {
            double step = -learning_rate * w_gradients.at(k) / n_samples;
            weights.at(k) += step;
            bias = bias - learning_rate * b_gradient / n_samples;
            w_gradients.at(k) = 0;
            b_gradient = 0;
        }

        if ((i + 1) % 100 == 0)
        {
            printf("Iteration: %zu, Cost: %.8g\n", i + 1, cost_function(X, y, weights, bias));
        }
    }
}

using std::vector;

void gradient_descent(
    const std::vector<std::vector<double>> &X,
    const std::vector<double> &y,
    std::vector<double> &weights,
    double &bias,
    double learning_rate,
    size_t n_iterations)
{
    size_t n_samples = y.size();
    size_t n_features = weights.size();

    for (size_t iteration = 0; iteration < n_iterations; ++iteration)
    {
        vector<double> predictions = predict(X, weights, bias);

        // Calculate gradients
        vector<double> dw(n_features, 0.0);
        double db = 0.0;

        for (size_t i = 0; i < n_samples; ++i)
        {
            double error = predictions[i] - y[i];
            for (size_t j = 0; j < n_features; ++j)
            {
                dw[j] += error * X[i][j];
            }
            db += error;
        }

        for (size_t j = 0; j < n_features; ++j)
        {
            weights[j] -= learning_rate * dw[j] / n_samples;
        }
        bias -= learning_rate * db / n_samples;

        // Print cost for monitoring (optional)
        if ((iteration + 1) % 100 == 0)
        {
            std::cout << "Iteration " << (iteration + 1) << ", Cost: " << std::fixed << std::setprecision(8) << cost_function(X, y, weights, bias) << std::endl;
        }
    }
}

void print_vec(const char *msg, const std::vector<double> &v)
{
    std::cout << msg << v << "\n";
}

int main()
{
    constexpr size_t n_samples = 1000;
    constexpr size_t n_features = 8;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::vector<double> w;
    double b = 10;

    std::mt19937 e;
    std::uniform_real_distribution<double> u;
    std::uniform_int_distribution<int> ui(2, 20);

    for (size_t i = 0; i < n_features; ++i)
    {
        w.push_back(static_cast<double>(ui(e)));
    }

    for (size_t i = 0; i < n_samples; ++i)
    {
        std::vector<double> v;
        for (size_t j = 0; j < n_features; ++j)
        {
            v.push_back(u(e));
        }
        X.push_back(v);
        y.push_back(dot_product(X.at(i), w) + b);
    }

    std::vector<double> lw;
    double lb{};

    std::vector<double> gw(n_features);
    double gb{};

    double learning_rate = 0.01;
    size_t n_iter = 1000;

    linear_regression(X, y, lw, lb, n_iter, learning_rate);
    std::cout << "\n\nPerforming gradient descent\n";
    gradient_descent(X, y, gw, gb, learning_rate, n_iter);

    print_vec("Original Weights: ", w);
    print_vec("Gradient Descent: ", gw);
    print_vec("Linear Regression: ", lw);

    return 0;
}