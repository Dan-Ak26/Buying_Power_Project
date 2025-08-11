#include "CsvTable.hpp"
#include <fstream>   // For reading CSV files
#include <sstream>   // For splitting lines by commas
#include <iostream>  // For debug/error output

using namespace std;
// Load a CSV file into the table
// Returns true if successful, false if file not found
bool CsvTable::load(const std::string &path) {
        ifstream file(path);
    if (!file.is_open()) {
        cerr << "Error: Cannot open CSV file: " << path << "\n";
        return false;
    }

    rows_.clear(); // Clear any existing data
        string line;

    // Read CSV line-by-line
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        Row row;

        // Extract each cell, split by comma
        while (getline(ss, cell, ',')) {
            row.cells.push_back(cell);
        }

        rows_.push_back(row);
    }

    return true;
}

// Return all rows as a vector
    vector<Row> CsvTable::rows() const {
        return rows_;
}
