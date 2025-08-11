#pragma once
#include "CsvTable.hpp"
#include <vector>

using namespace std;
// Class for transforming buying power data
class BuyingPowerTransformer {
    public: //Sorted a vector of CSV rows by date (ascending order)
            // Converts Row objects to DataPoint objects
            vector<CsvTable::Row>sortedByDate(vector<CsvTable::Row> x)const;
            //Remove duplicate data points based on date.
            //Keeps the first occurrence of each date.
            vector<CsvTable::Row>dedupeByDate(const vector<CsvTable::Row>& x)const;
            // Normalize values so that the first data point equals 'base.
            //Example: if base=100, the first value becomes 100 and others are
            //scaled accordingly.
            vector<CsvTable::Row>normalize(const vector<CsvTable::Row>& x,
                 double newBase = 100.0)const;
};