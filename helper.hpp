#pragma once
#include <vector>
#include <iostream>
#include <array>

// std::ostream &operator<<(std::ostream &out, const std::vector<double> &v);

[[maybe_unused]]
static std::ostream &operator<<(std::ostream &out, const std::vector<double> &v)
{
    std::printf("Vector(size=%zu, [", v.size());
    for (size_t i = 0; i < v.size(); ++i)
    {
        i == v.size() - 1 ? out << v[i] : out << v[i] << ", ";
    }
    out << "])";
    return out;
}

template <std::size_t M, std::size_t N>
std::ostream &operator<<(std::ostream &out, const std::array<std::array<double, N>, M> &X)
{
    out << "[";
    for (size_t i = 0; i < M; ++i)
    {
        i ? out << " [" : out << "[";
        for (size_t j = 0; j < N; ++j)
        {
            out << X[i][j];
            if (j != N - 1)
                out << ", ";
        }
        i != M - 1 ? out << "],\n" : out << "]]\n";
    }

    return out;
}

template <std::size_t N>
double predict(const std::array<double, N> &x, const std::array<double, N> &w, double bias = 0)
{
    return std::inner_product(x.begin(), x.end(), w.begin(), bias);
}

template <std::size_t M, std::size_t N>
std::vector<double>
predict(
    const std::array<std::array<double, N>, M> &X,
    const std::array<double, N> &w,
    double bias = 0) noexcept
{
    std::vector<double> result(w.size(), 0.0);
    for (const auto &x : X)
    {
        for (size_t j = 0; j < x.size(); ++j)
        {
            result.at(j) = std::inner_product(x.begin(), x.end(), w.begin(), bias);
        }
    }

    return result;
}