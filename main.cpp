#include "CsvTable.hpp"                // Custom class for reading and storing CSV data
#include "BuyingPowerTransformer.hpp"  // Custom class for processing & transforming buying power data

#include <iostream>   // For console input/output
#include <fstream>    // For file output (e.g., saving BMP)
#include <vector>     // For dynamic arrays (used in image data, CSV rows, etc.)
#include <cstdint>    // For fixed-width integer types like uint32_t
#include <algorithm>  // For sorting and transformation operations
#include <filesystem> // For file path handling

// Function to save image data in BMP format
// Parameters:
//   path - location to save the BMP file
//   W    - image width in pixels
//   H    - image height in pixels
//   bgr  - pixel data in BGR format (bottom-up order will be handled inside)
static bool saveBMP(const std::string& path, int W, int H, const std::vector<uint8_t>& bgr) {
    // Row size must be padded to multiple of 4 bytes
    uint32_t rowSize = ((24 * W + 31) / 32) * 4;
    uint32_t imgSize = rowSize * H;
    uint32_t fileSize = 14 + 40 + imgSize; // BMP header (14 bytes) + DIB header (40 bytes) + image data

    // Open file in binary mode
    std::ofstream f(path, std::ios::binary);
    if (!f) return false; // If file couldn't be opened, return false

    // BMP Header (14 bytes)
    f.put('B').put('M');               // Signature
    f.write(reinterpret_cast<char*>(&fileSize), 4); // File size
    uint32_t reserved = 0;
    f.write(reinterpret_cast<char*>(&reserved), 4); // Reserved bytes
    uint32_t offset = 14 + 40; // Offset to image data
    f.write(reinterpret_cast<char*>(&offset), 4);

    // DIB Header (40 bytes)
    uint32_t headerSize = 40;
    f.write(reinterpret_cast<char*>(&headerSize), 4); // DIB header size
    f.write(reinterpret_cast<char*>(&W), 4);          // Image width
    f.write(reinterpret_cast<char*>(&H), 4);          // Image height
    uint16_t planes = 1;
    f.write(reinterpret_cast<char*>(&planes), 2);     // Number of color planes
    uint16_t bpp = 24;
    f.write(reinterpret_cast<char*>(&bpp), 2);        // Bits per pixel
    uint32_t compression = 0;
    f.write(reinterpret_cast<char*>(&compression), 4);// Compression method
    f.write(reinterpret_cast<char*>(&imgSize), 4);    // Image size
    uint32_t ppm = 2835; // 72 DPI
    f.write(reinterpret_cast<char*>(&ppm), 4);        // Horizontal resolution
    f.write(reinterpret_cast<char*>(&ppm), 4);        // Vertical resolution
    uint32_t colors = 0;
    f.write(reinterpret_cast<char*>(&colors), 4);     // Number of colors
    uint32_t importantColors = 0;
    f.write(reinterpret_cast<char*>(&importantColors), 4); // Important colors

    // Write image data
    for (int y = 0; y < H; ++y) {
        f.write(reinterpret_cast<const char*>(&bgr[y * W * 3]), W * 3);
        // Add row padding
        for (uint32_t pad = 0; pad < rowSize - W * 3; ++pad) {
            f.put(0);
        }
    }

    return true; // File saved successfully
}

int main() {
    CsvTable t; // Create CSV table object

    // Load CSV file (returns false if missing)
    if (!t.load("data/CURR0000SA0R.csv")) {
        std::cerr << "CSV Missing\n";
        return 1;
    }

    BuyingPowerTransformer tf; // Transformer object for cleaning/normalizing data

    // Step 1: Remove duplicate entries & sort by date
    auto clean = tf.dedupeByDate(tf.sortedByDate(t.rows()));

    // Step 2: Normalize values so that 100.0 is the base
    auto norm = tf.normalize(clean, 100.0);

    // Output the number of rows
    std::cout << "Rows: " << norm.size() << "\n";

    // Step 3: Print each date and its normalized value
    for (int i = 0; i < (int)norm.size(); ++i) {
        std::cout << norm[i].date << " -> " << norm[i].value << "\n";
    }

    return 0; // Exit successfully
}
