#include "BuyingPowerTransformer.hpp"
#include <algorithm> // For sorting and removing duplicates
#include <unordered_set> // For fast duplicate detection

using namespace std;
// Sort rows by date (ascending)
    vector<DataPoint> BuyingPowerTransformer::sortedByDate(const std::vector<Row> &rows) {
        vector<DataPoint> points;

    // Convert Row objects into DataPoint objects
    for (const auto &row : rows) {
        if (row.cells.size() >= 2) {
            DataPoint dp;
            dp.date = row.cells[0];
            dp.value = std::stod(row.cells[1]); // Convert value from string to double
            points.push_back(dp);
        }
    }

    // Sort by date (string compare assuming YYYY-MM-DD format)
        sort(points.begin(), points.end(),
              [](const DataPoint &a, const DataPoint &b) {
                  return a.date < b.date;
              });

    return points;
}

// Remove duplicate entries based on date, keeping first occurrence
   vector<DataPoint> BuyingPowerTransformer::dedupeByDate(const std::vector<DataPoint> &points) {
     vector<DataPoint> unique;
       unordered_set<std::string> seenDates;

    for (const auto &p : points) {
        if (seenDates.insert(p.date).second) { // If date not already in set
            unique.push_back(p);
        }
    }

    return unique;
}

// Normalize values relative to a base (e.g., base=100 means first value becomes 100)
    vector<DataPoint> BuyingPowerTransformer::normalize(const std::vector<DataPoint> &points, double base) {
        vector<DataPoint> norm = points;

    if (!points.empty()) {
        double firstValue = points.front().value;
        for (auto &p : norm) {
            p.value = (p.value / firstValue) * base;
        }
    }

    return norm;
}
