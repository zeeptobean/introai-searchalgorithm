#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <format>
#include <functional>

using namespace std;

struct ContinuousProblem {
    function<double(const vector<double>&)> objective_function;
    uint32_t dimension;
    double upper_bound;
    double lower_bound;
};

// Random number setup
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dist(0.0, 1.0);

//geometric cooling, with optional max_iteration to avoid infinite loop
void SA(const ContinuousProblem& problem, double temp = 98.0, double cooling_rate = 0.96, double min_temp = 0.0001, double step_bound = 0.01, uint32_t max_iterations = 10000) {
    uniform_real_distribution<double> bounded_dist(problem.lower_bound, problem.upper_bound);
    uniform_real_distribution<double> step_dist(-step_bound, step_bound);

    vector<double> current_x(problem.dimension);
    for (auto& x : current_x) {
        x = bounded_dist(gen);
    }
    double current_energy = problem.objective_function(current_x);

    uint32_t iteration = 1;
    while (temp > min_temp && iteration <= max_iterations) {
        // 1. Propose a random neighbor
        vector<double> next_x = current_x;
        for(auto& x : next_x) {
            x += step_dist(gen);
            x = std::clamp(x, problem.lower_bound, problem.upper_bound);
        }
        double next_energy = problem.objective_function(next_x);

        //print sth

        // 2. Calculate change in "energy" (we want to` maximize, so energy = value)
        double delta = next_energy - current_energy;

        // 3. Acceptance probability
        // If delta > 0, exp(delta/temp) will be > 1, so we always accept.
        if (delta > 0 || dist(gen) < std::exp(delta / temp)) {
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
    //print 
}

void SALinear(const ContinuousProblem& problem, double max_temp = 98.0, double step_bound = 0.01, uint32_t iterations = 10000) {
    uniform_real_distribution<double> bounded_dist(problem.lower_bound, problem.upper_bound);
    uniform_real_distribution<double> step_dist(-step_bound, step_bound);

    vector<double> current_x(problem.dimension);
    for (auto& x : current_x) {
        x = bounded_dist(gen);
    }
    double current_energy = problem.objective_function(current_x);

    uint32_t iteration = 0;
    while (iteration < iterations) {
        double temp = max_temp * (1.0 - (double) iteration / iterations);
        vector<double> next_x = current_x;
        for(auto& x : next_x) {
            x += step_dist(gen);
            x = std::clamp(x, problem.lower_bound, problem.upper_bound);
        }
        double next_energy = problem.objective_function(next_x);

        //print sth

        double delta = next_energy - current_energy;

        // 3. Acceptance probability
        // If delta > 0, exp(delta/temp) will be > 1, so we always accept.
        if (delta > 0 || dist(gen) < std::exp(delta / temp)) {
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