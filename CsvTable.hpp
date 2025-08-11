#pragma once
#include <string>
#include <vector>

using namespace std;
// Class for loading and storing CSV data
class CsvTable {
    public:// Represents a single row in a CSV file  
        // Each cell is stored as a string
         struct Row {string date; double value; };
        //Load CSV file from given path
        // Returns true if file was successfully
        //loaded, false otherwise
         bool load(const string&path);
         // Retrieve all loaded rows as a vector
         const vector<Row>& rows() const
         {return rows_;}
    private:
        // Internal storage for CSV rows
        vector<Row> rows_;      
};