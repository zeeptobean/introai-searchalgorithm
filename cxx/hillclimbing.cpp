#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

double objectiveFunction(double x) {
    double term1 = std::pow(x, 4);
    double term2 = -3.0 * std::pow(x, 2);
    double term3 = std::sqrt(2) * x;
    
    return -(term1 + term2 + term3) / (2.0/3.0);
}

//find max of a function
void continuousHillClimbing() {
    double current_x = 0.0; // Starting point
    double step_size = 0.0005;
    double current_val = objectiveFunction(current_x);

    while (true) {
        double next_x_plus = current_x + step_size;
        double next_x_minus = current_x - step_size;

        if (objectiveFunction(next_x_plus) > current_val) {
            current_x = next_x_plus;
            current_val = objectiveFunction(next_x_plus);
        } else if (objectiveFunction(next_x_minus) > current_val) {
            current_x = next_x_minus;
            current_val = objectiveFunction(next_x_minus);
        } else {
            // No better neighbors found
            break;
        }
    }
    std::cout << "Continuous Max found at x = " << current_x << " (Value: " << current_val << ")\n";
}

// --- DISCRETE EXAMPLE ---
// Objective: Find the highest value in a fixed set of integers
void discreteHillClimbing() {
    std::vector<int> data = {1, 3, 7, 12, 15, 14, 10, 5};
    int current_index = 0; // Start at index 0 (value 1)

    while (true) {
        int next_index = -1;
        int best_neighbor_val = data[current_index];

        // Check left neighbor
        if (current_index > 0 && data[current_index - 1] > best_neighbor_val) {
            next_index = current_index - 1;
            best_neighbor_val = data[next_index];
        }
        // Check right neighbor
        if (current_index < data.size() - 1 && data[current_index + 1] > best_neighbor_val) {
            next_index = current_index + 1;
            best_neighbor_val = data[next_index];
        }

        if (next_index != -1) {
            current_index = next_index;
        } else {
            break; // Local maximum reached
        }
    }
    std::cout << "Discrete Max found at index " << current_index << " (Value: " << data[current_index] << ")\n";
}

int main() {
    continuousHillClimbing();
    // discreteHillClimbing();
    return 0;
}