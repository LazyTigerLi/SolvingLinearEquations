/**
 * Equations_solve solves large linear equations both the CPU and the offload device,
 * then compares results. If the code executes on both CPU and the offload
 * device, the name of the offload device and a success message are displayed.
 */

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include <random>

using namespace sycl;

// The number of linear equations
constexpr int N = 10;


/**
 * Generate linear equations randomly
 */
void generate(float(*matrix)[N + 1])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-10 * N, 10 * N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (j != i)
                matrix[i][j] = 1 + i * j;
        }
        matrix[i][i] = (1 + (N - 1) * i) * N / 2.0 - (i - 1) * i + 1;
        matrix[i][N] = dis(gen);
    }
}

/**
 * Perform matrix multiplication on host to verify results from device.
 */
bool verify(float(*matrix)[N + 1], float* result)
{
    for (int i = 0; i < N; ++i)
    {
        float sum = 0.0;
        for (int j = 0; j < N; ++j)
            sum += matrix[i][j] * result[j];
        if (std::fabs(sum - matrix[i][N]) > std::numeric_limits<float>::epsilon())
            return false;
    }
    return true;
}

int main() {
    // Initialize the linear equations, eps is the error
    float matrix[N][N + 1];
    float result[N];
    float eps = 1 + std::numeric_limits<float>::epsilon();
    for (int i = 0; i < N; ++i)
        result[i] = 0;
    generate(matrix);

    // The exception handler
    auto ehandler = [](cl::sycl::exception_list exceptionList) {
        for (std::exception_ptr const& e : exceptionList) 
        {
            try 
            {
                std::rethrow_exception(e);
            }
            catch (cl::sycl::exception const& e) 
            {
                std::cout << "Caught an asynchronous DPC++ exception, terminating the program.\n";
                std::terminate();
            }
        }
    };

    // Initialize the device queue with the default selector. The device queue is
    // used to enqueue kernels. It encapsulates all states needed for execution.
    default_selector selector;
    queue q(selector, ehandler);

    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    // Create buffers for the coefficient matrix and the result vector
    buffer<float, 2> matrix_buf(reinterpret_cast<float*>(matrix), range<2>(N, N + 1));
    buffer<float, 1> result_buf(result, range<1>(N));
    buffer<float, 1> eps_buf(&eps, range<1>(1));

    std::cout << "The number of equations: N " << N << "\n";

    while (eps > std::numeric_limits<float>::epsilon())
    {
        q.submit([&](auto& h)
            {
                accessor m(matrix_buf, h, read_only);
                accessor r(result_buf, h);
                accessor e(eps_buf, h);
                // Execute kernel.
                h.parallel_for(range(N), [=](auto index)
                    {
                        float sum = 0.0;
                        for (int j = 0; j < N; ++j)
                        {
                            if (j != index)
                                sum += m[index][j] * r[j];
                        }
                        float old = r[index];
                        r[index] = (m[index][N] - sum) / m[index][index];
                        if (e[0] > std::fabs(r[index] - old))
                            e[0] = std::fabs(r[index] - old);
                    });
            }).wait();
    }
    
    std::cout << "The linear equations:\n";
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N + 1; ++j)
            std::cout << matrix[i][j] << '\t';
        std::cout << '\n';
    }
    std::cout << "The solution of the equations using DPC++:\n";
    for (int i = 0; i < N; ++i)
        std::cout << result[i] << '\t';
    std::cout << '\n';
    
    if (!verify(matrix, result))
        std::cout << "Success!\n";
    else
        std::cout << "Failure\n";

    return 0;
}