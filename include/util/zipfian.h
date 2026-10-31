#ifndef COMMON_ZIPFIAN_H
#define COMMON_ZIPFIAN_H

#include <cassert>
#include <cstdint>
#include <cmath>

class Zipfian {
public:
    Zipfian(double theta, size_t n) : n_(n),
        alpha_(1 / (1 - theta)), zeta2_(1 + pow(2.0, -theta)) {
        assert(n > 2);
        zetan_ = zeta2_;
        for (size_t i = 3; i <= n; ++i) {
            zetan_ += pow((double)i, -theta);
        }
        eta_ = (1 - pow(2.0 / n, 1 - theta)) / (1 - zeta2_ / zetan_);
    }

    size_t GetIdx(double val) {
        double uz = val * zetan_;
        if (uz < 1) return 0;
        if (uz < zeta2_) return 1;
        return (size_t)(n_ * pow(eta_ * val - eta_ + 1, alpha_));
    }

private:
    size_t n_;
    double alpha_;
    double zeta2_;
    double zetan_;
    double eta_;
};

#endif