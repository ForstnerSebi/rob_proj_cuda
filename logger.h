#include <stdio.h>
#include <chrono>

#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

namespace logger
{
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;

    void start_time()
    {
        t_start = high_resolution_clock::now();
    }

    void log(int layer, string module_name, string message)
    {
        duration<double> timestamp = high_resolution_clock::now() - t_start;
        cout << "\033[1;34m[" << timestamp.count() << "]\033[0m" << "\033[1;31m\t" << layer << "\t\033[0m" << "\033[1;32m(" << module_name << ")\033[0m" << "\t" << message << "\n";
    }
}