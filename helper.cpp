#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include "helper.hpp"
#include <algorithm>
#include <cfenv>
#include <array>
#include <cassert>

void log_error(const char *msg, int line, const char *func)
{
    std::cout << "\033[31m"
              << "ERROR: "
              << line << " " << func << ": " << msg << "\033[m" << "\n";
}

static inline double predict(const std::vector<double> &x, const std::vector<double> &w, double bias = 0)
{
    if (x.size() != w.size())
    {
        log_error("vector size mismatch", __LINE__, __func__);
        return std::numeric_limits<double>::signaling_NaN();
    }

    return std::inner_product(x.begin(), x.end(), w.begin(), bias);
}


