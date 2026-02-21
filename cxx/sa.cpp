#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <format>

using namespace std;

// Random number setup
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

double objectiveFunction(double x) {
    double term1 = std::pow(x, 4);
    double term2 = -3.0 * std::pow(x, 2);
    double term3 = std::sqrt(2) * x;
    
    return -(term1 + term2 + term3) / (2.0/3.0);
}

void continuousSA() {
    double current_x = 0.0;
    double current_energy = objectiveFunction(current_x);
    
    double temp = 10.0;
    const double cooling_rate = 0.99;      
    const double min_temp = 0.001;

    std::uniform_real_distribution<> move_dis(-1.0, 1.0);

    uint32_t iteration = 1;
    const uint32_t max_iterations = 10000;
    while (temp > min_temp && iteration < max_iterations) {
        // 1. Propose a random neighbor
        double next_x = current_x + move_dis(gen);
        double next_energy = objectiveFunction(next_x);

        cout << format("Temp: {:.2f}, Current: f({:.2f}) = {:.4f}, Next: f({:.2f}) = {:.4f}\n",
               temp, current_x, current_energy, next_x, next_energy);

        // 2. Calculate change in "energy" (we want to` maximize, so energy = value)
        double delta = next_energy - current_energy;

        // 3. Acceptance probability
        // If delta > 0, exp(delta/temp) will be > 1, so we always accept.
        if (delta > 0 || dis(gen) < std::exp(delta / temp)) {
            current_x = next_x;
            current_energy = next_energy;
        }

        // 4. Cool down
        temp *= cooling_rate;
        iteration++;
    }
    if(iteration >= max_iterations) {
        cout << "Reached max iterations\n";
    } else {
        cout << format("Iterations: {}, ", iteration);
    }
    std::cout << "Continuous SA Max: f(" << current_x << ") = " << current_energy << "\n";
}

void continuousSALinear() {
    double current_x = 0.0;
    double current_energy = objectiveFunction(current_x);
    const double max_temp = 10.0;
    std::uniform_real_distribution<> move_dis(-1.0, 1.0);

    uint32_t iteration = 0;
    const uint32_t max_iterations = 10000;
    while (iteration < max_iterations) {
        double temp = max_temp * (1.0 - (double) iteration / max_iterations);
        double next_x = current_x + move_dis(gen);
        double next_energy = objectiveFunction(next_x);

        cerr << format("Temp: {:.2f}, Current: f({:.2f}) = {:.4f}, Next: f({:.2f}) = {:.4f}\n",
               temp, current_x, current_energy, next_x, next_energy);

        // 2. Calculate change in "energy" (we want to` maximize, so energy = value)
        double delta = next_energy - current_energy;

        // 3. Acceptance probability
        // If delta > 0, exp(delta/temp) will be > 1, so we always accept.
        if (delta > 0 || dis(gen) < std::exp(delta / temp)) {
            current_x = next_x;
            current_energy = next_energy;
        }
        iteration++;
    }
    std::cerr << "Continuous SA Linear Max: f(" << current_x << ") = " << current_energy << "\n";
}

// --- DISCRETE SA ---
// Goal: Find the max value in a "rugged" vector
void discreteSA() {
    std::vector<int> landscape = {1, 10, 2, 3, 25, 2, 8, 30, 5, 1}; 
    int current_idx = 0;
    int current_val = landscape[current_idx];

    double temp = 100.0;
    double cooling_rate = 0.99;

    std::uniform_int_distribution<> step_dis(-1, 1);

    while (temp > 0.1) {
        // 1. Pick a neighbor (left or right)
        int direction = (dis(gen) > 0.5) ? 1 : -1;
        int next_idx = current_idx + direction;

        // Boundary check
        if (next_idx >= 0 && next_idx < landscape.size()) {
            int next_val = landscape[next_idx];
            double delta = (double)next_val - current_val;

            // 2. Acceptance logic
            if (delta > 0 || dis(gen) < std::exp(delta / temp + (double) 1e-9)) {
                current_idx = next_idx;
                current_val = next_val;
            }
        }
        temp *= cooling_rate;
    }
    std::cout << "Discrete SA Max: Index " << current_idx << " (Value: " << current_val << ")\n";
}

int main() {
    freopen("dump.txt", "w", stdout);
    freopen("dump2.txt", "w", stderr);
    continuousSA();
    continuousSALinear();
    // discreteSA();
    return 0;
}