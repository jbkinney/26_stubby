// Adapted from TFMPvalue by Jean-Stéphane Varré (2007).
// Generalized from fixed 4-letter DNA alphabet to arbitrary alphabet size C.
// Stripped of R/Rcpp dependencies; designed for pybind11 wrapping.

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cstdint>
#include <cmath>
#include <chrono>
#include <map>
#include <vector>
#include <algorithm>

typedef int64_t qlonglong;

#define ROUND_TO_INT(n) ((qlonglong)floor(n))

class Matrix {
public:
    int numChars;   // alphabet size C
    int length;     // number of positions L

    std::vector<std::vector<double>> mat;      // [char][pos], the raw score matrix
    std::vector<std::vector<qlonglong>> matInt; // [char][pos], discretized integer matrix
    std::vector<double> background;             // uniform: 1.0/numChars each

    double granularity;
    double errorMax;
    std::vector<qlonglong> offsets;
    qlonglong offset;
    std::vector<qlonglong> minScoreColumn;
    std::vector<qlonglong> maxScoreColumn;
    qlonglong minScore;
    qlonglong maxScore;
    qlonglong scoreRange;
    std::vector<qlonglong> bestScore;
    std::vector<qlonglong> worstScore;

    double maxTime;         // wall-clock limit in seconds (0 = no limit)
    size_t maxDictSize;     // max entries per DP map (0 = no limit)
    int abortReason;        // 0=ok, 1=max_dict_size, 2=max_time
    std::chrono::steady_clock::time_point deadline;

    Matrix() : numChars(0), length(0), granularity(1.0), offset(0),
               errorMax(0), minScore(0), maxScore(0), scoreRange(0),
               maxTime(0), maxDictSize(0), abortReason(0) {}

    // Initialize from an L x C matrix (theta[l][c] = fitness contribution).
    // Internally stored as [char][pos] to match TFMPvalue convention.
    void initFromTheta(const std::vector<std::vector<double>>& theta);

    void computesIntegerMatrix(double granularity, bool sortColumns = true);

    // Returns distribution as probability-weighted values (background = 1/C).
    // Multiply by C^L to convert to sequence counts.
    std::vector<std::map<qlonglong, double>>
    calcDistribWithMapMinMax(qlonglong min, qlonglong max);

    void lookForPvalue(qlonglong requestedScore, qlonglong min, qlonglong max,
                       double* pmin, double* pmax);

    qlonglong lookForScore(qlonglong min, qlonglong max,
                           double requestedPvalue, double* rpv, double* rppv);
};

#endif
