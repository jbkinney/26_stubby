// Adapted from TFMPvalue by Jean-Stéphane Varré (2007).
// Core DP and iterative refinement, generalized to arbitrary alphabet size.

#include "Matrix.h"

void Matrix::initFromTheta(const std::vector<std::vector<double>>& theta) {
    length = (int)theta.size();
    if (length == 0) return;
    numChars = (int)theta[0].size();

    // Store as [char][pos] to match TFMPvalue convention
    mat.resize(numChars, std::vector<double>(length));
    for (int l = 0; l < length; l++) {
        for (int k = 0; k < numChars; k++) {
            mat[k][l] = theta[l][k];
        }
    }

    // Uniform background: each character has probability 1/C
    background.assign(numChars, 1.0 / numChars);
}

void Matrix::computesIntegerMatrix(double gran, bool sortColumns) {
    double minS = 0, maxS = 0;

    for (int i = 0; i < length; i++) {
        double cmin = mat[0][i];
        double cmax = cmin;
        for (int k = 1; k < numChars; k++) {
            if (mat[k][i] < cmin) cmin = mat[k][i];
            if (mat[k][i] > cmax) cmax = mat[k][i];
        }
        minS += cmin;
        maxS += cmax;
    }

    double sRange = maxS - minS + 1;

    if (gran > 1.0) {
        this->granularity = gran / sRange;
    } else if (gran < 1.0) {
        this->granularity = 1.0 / gran;
    } else {
        this->granularity = 1.0;
    }

    matInt.assign(numChars, std::vector<qlonglong>(length));
    for (int k = 0; k < numChars; k++) {
        for (int p = 0; p < length; p++) {
            matInt[k][p] = ROUND_TO_INT(mat[k][p] * this->granularity);
        }
    }

    this->errorMax = 0.0;
    for (int i = 0; i < length; i++) {
        double maxE = mat[0][i] * this->granularity - matInt[0][i];
        for (int k = 1; k < numChars; k++) {
            double e = mat[k][i] * this->granularity - matInt[k][i];
            if (e > maxE) maxE = e;
        }
        this->errorMax += maxE;
    }

    if (sortColumns) {
        qlonglong globalMin = 0;
        for (int i = 0; i < length; i++) {
            for (int k = 0; k < numChars; k++) {
                if (matInt[k][i] < globalMin) globalMin = matInt[k][i];
            }
        }
        globalMin--;

        std::vector<qlonglong> maxs(length);
        for (int i = 0; i < length; i++) {
            maxs[i] = matInt[0][i];
            for (int k = 1; k < numChars; k++) {
                if (matInt[k][i] > maxs[i]) maxs[i] = matInt[k][i];
            }
        }

        std::vector<std::vector<qlonglong>> mattemp(numChars, std::vector<qlonglong>(length));
        for (int i = 0; i < length; i++) {
            qlonglong mx = maxs[0];
            int p = 0;
            for (int j = 1; j < length; j++) {
                if (maxs[j] > mx) {
                    mx = maxs[j];
                    p = j;
                }
            }
            maxs[p] = globalMin;
            for (int k = 0; k < numChars; k++) {
                mattemp[k][i] = matInt[k][p];
            }
        }

        for (int k = 0; k < numChars; k++) {
            for (int i = 0; i < length; i++) {
                matInt[k][i] = mattemp[k][i];
            }
        }
    }

    // Offsets to make all integer scores non-negative
    this->offset = 0;
    offsets.resize(length);
    for (int i = 0; i < length; i++) {
        qlonglong cmin = matInt[0][i];
        for (int k = 1; k < numChars; k++) {
            if (matInt[k][i] < cmin) cmin = matInt[k][i];
        }
        offsets[i] = -cmin;
        for (int k = 0; k < numChars; k++) {
            matInt[k][i] += offsets[i];
        }
        this->offset += offsets[i];
    }

    // Per-column and total score ranges
    minScoreColumn.resize(length);
    maxScoreColumn.resize(length);
    minScore = 0;
    maxScore = 0;
    for (int i = 0; i < length; i++) {
        minScoreColumn[i] = matInt[0][i];
        maxScoreColumn[i] = matInt[0][i];
        for (int k = 1; k < numChars; k++) {
            if (matInt[k][i] < minScoreColumn[i]) minScoreColumn[i] = matInt[k][i];
            if (matInt[k][i] > maxScoreColumn[i]) maxScoreColumn[i] = matInt[k][i];
        }
        minScore += minScoreColumn[i];
        maxScore += maxScoreColumn[i];
    }
    this->scoreRange = maxScore - minScore + 1;

    bestScore.resize(length);
    worstScore.resize(length);
    bestScore[length - 1] = maxScore;
    worstScore[length - 1] = minScore;
    for (int i = length - 2; i >= 0; i--) {
        bestScore[i] = bestScore[i + 1] - maxScoreColumn[i + 1];
        worstScore[i] = worstScore[i + 1] - minScoreColumn[i + 1];
    }
}

std::vector<std::map<qlonglong, double>>
Matrix::calcDistribWithMapMinMax(qlonglong min, qlonglong max) {
    std::vector<std::map<qlonglong, double>> nbocc(length + 1);
    abortReason = 0;

    std::vector<qlonglong> maxs(length + 1);
    maxs[length] = 0;
    for (int i = length - 1; i >= 0; i--) {
        maxs[i] = maxs[i + 1] + maxScoreColumn[i];
    }

    for (int k = 0; k < numChars; k++) {
        if (matInt[k][0] + maxs[1] >= min) {
            nbocc[0][matInt[k][0]] += background[k];
        }
    }

    nbocc[length - 1][max + 1] = 0.0;
    for (int pos = 1; pos < length; pos++) {
        auto iter = nbocc[pos - 1].begin();
        while (iter != nbocc[pos - 1].end()) {
            for (int k = 0; k < numChars; k++) {
                qlonglong sc = iter->first + matInt[k][pos];
                if (sc + maxs[pos + 1] >= min) {
                    if (sc > max) {
                        nbocc[length - 1][max + 1] += nbocc[pos - 1][iter->first] * background[k];
                    } else {
                        nbocc[pos][sc] += nbocc[pos - 1][iter->first] * background[k];
                    }
                }
            }
            iter++;

            if (maxDictSize > 0 && nbocc[pos].size() > maxDictSize) {
                abortReason = 1;
                return nbocc;
            }
            if (maxTime > 0 && std::chrono::steady_clock::now() > deadline) {
                abortReason = 2;
                return nbocc;
            }
        }
    }

    return nbocc;
}

void Matrix::lookForPvalue(qlonglong requestedScore, qlonglong min, qlonglong max,
                           double* pmin, double* pmax) {
    auto nbocc = calcDistribWithMapMinMax(min, max);

    // Cumulative tail sum from high to low, stored in nbocc[length]
    double sum = 0.0;
    qlonglong s = max + 1;
    auto riter = nbocc[length - 1].rbegin();
    while (riter != nbocc[length - 1].rend()) {
        sum += riter->second;
        if (riter->first >= requestedScore) s = riter->first;
        nbocc[length][riter->first] = sum;
        riter++;
    }

    auto iter = nbocc[length].find(s);
    while (iter != nbocc[length].begin()) {
        auto prev = iter;
        --prev;
        if (prev->first >= s - (qlonglong)ceil(errorMax)) {
            iter = prev;
        } else {
            break;
        }
    }

    *pmax = nbocc[length][s];
    *pmin = iter->second;
}

qlonglong Matrix::lookForScore(qlonglong min, qlonglong max,
                               double requestedPvalue, double* rpv, double* rppv) {
    auto nbocc = calcDistribWithMapMinMax(min, max);

    double sum = 0.0;
    auto riter = nbocc[length - 1].rbegin();
    qlonglong alpha = riter->first + 1;
    qlonglong alpha_E = alpha;
    nbocc[length][alpha] = 0.0;

    while (riter != nbocc[length - 1].rend()) {
        sum += riter->second;
        nbocc[length][riter->first] = sum;
        if (sum >= requestedPvalue) {
            break;
        }
        riter++;
    }

    if (sum > requestedPvalue) {
        alpha_E = riter->first;
        --riter;
        alpha = riter->first;
    } else {
        if (riter == nbocc[length - 1].rend()) {
            --riter;
            alpha = alpha_E = riter->first;
        } else {
            alpha = riter->first;
            ++riter;
            sum += riter->second;
            alpha_E = riter->first;
        }
        nbocc[length][alpha_E] = sum;
    }

    if (alpha - alpha_E > (qlonglong)ceil(errorMax)) alpha_E = alpha;

    *rpv = nbocc[length][alpha];
    *rppv = nbocc[length][alpha_E];

    return alpha;
}
