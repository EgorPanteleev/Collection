#include "iostream"
#include <Kokkos_Core.hpp>
//#define KOKKOS_ENABLE_HIP

void computation(Kokkos::View<int*> array, int size) {
    Kokkos::parallel_for("compute", size, KOKKOS_LAMBDA(const int i) {
        array(i) = i * i;
    });
}
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    const int size = 100000000;
    Kokkos::View<int*> array("array", size);

    // Measure sequential time
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i) {
        array(i) = i * i;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seq_time = end - start;
    std::cout << "Sequential time: " << seq_time.count() << " seconds" << std::endl;

    // Measure parallel time
    start = std::chrono::high_resolution_clock::now();
    computation(array, size);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> par_time = end - start;
    std::cout << "Parallel time: " << par_time.count() << " seconds" << std::endl;

    std::cout << "With parallel we " << seq_time.count() / par_time.count() * 100 << "% faster" << std::endl;
    Kokkos::finalize();
    return 0;

}

