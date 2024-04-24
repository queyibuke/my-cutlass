#include <iostream>
#include <fstream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"
#include <cstring>

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Put all the created template variables to create GemmSplitKParallel template variable
using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                                       LayoutInputA,
                                                       ElementInputB,
                                                       LayoutInputB,
                                                       ElementOutput,
                                                       LayoutOutput,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

int run(cutlass::HostTensor<ElementInputA, LayoutInputA>& tensor_a, 
        cutlass::HostTensor<ElementInputB, LayoutInputB>& tensor_b,
        cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_c,
        cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_d,
        ElementComputeEpilogue alpha,
        ElementComputeEpilogue beta,
        cutlass::gemm::GemmCoord& problem_size,
        int split_k_slices) {

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS to host for comparison
  tensor_d.sync_host();
  if(status !=cutlass::Status::kSuccess) {
    return -1;
  } else {
    return 0;
  }
}

int main() {

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }
  if (props.major != 8) {//7
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests pass if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;
    return 0;
  }
  else {
    
  // Define problem size
  const int length_m = 5120;
  const int length_n = 4096;
  const int length_k = 4096;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());  // <- Create matrix D with dimensions M x N 
                                                                                 // used to store output from CUTLASS kernel
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  // Split K dimension into 16 partitions
  int split_k_slices = 16;

  // // Fill input and output matrices on host using CUTLASS helper functions
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_a.host_view(),
  //     1,
  //     ElementInputA(4),
  //     ElementInputA(-4),
  //     0);  // <- Fill matrix A on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_b.host_view(),
  //     1,
  //     ElementInputB(4),
  //     ElementInputB(-4),
  //     0);  // <- Fill matrix B on host with uniform-distribution random data
  // cutlass::reference::host::TensorFillRandomUniform(
  //     tensor_c.host_view(),
  //     1,
  //     ElementOutput(4),
  //     ElementOutput(-4),
  //     0);  // <- Fill matrix C on host with uniform-distribution random data
  // cutlass::reference::host::TensorFill(
  //     tensor_d.host_view());  // <- fill matrix D on host with zeros
  //ElementAccumulator matrix_a[problem_size.m()][problem_size.k()];
  //ElementAccumulator matrix_b[problem_size.k()][problem_size.n()];
  //ElementAccumulator matrix_c[problem_size.m()][problem_size.n()];
  //ElementAccumulator matrix_d[problem_size.m()][problem_size.n()];
  
  //memset(matrix_a, 0x49, problem_size.m() * problem_size.k() * sizeof(ElementAccumulator));
  //memset(matrix_b, 0x50, problem_size.k() * problem_size.n() * sizeof(ElementAccumulator));
  //memset(matrix_c, 0x51, problem_size.m() * problem_size.n() * sizeof(ElementAccumulator));
  //memset(matrix_d, 0x48, problem_size.m() * problem_size.n() * sizeof(ElementAccumulator));
  std::cout << "1" << std::endl; 
  cutlass::reference::host::TensorFill(tensor_a.host_view(), ElementInputA(1));
  cutlass::reference::host::TensorFill(tensor_b.host_view(), ElementInputB(2));
  cutlass::reference::host::TensorFill(tensor_c.host_view(), ElementOutput(3));
  cutlass::reference::host::TensorFill(tensor_d.host_view(), ElementOutput(0));
  std::cout << "2" << std::endl;

  int result = run(tensor_a, tensor_b, tensor_c, tensor_d, alpha, beta, problem_size, split_k_slices);
  std::cout << "3" << std::endl;

  for(unsigned int i = 0; i < tensor_d.size(); i++) {
	  if(i != 0 && i % 5120 == 0) {
	  	std::cout << "\nthe " << i / 5120 << "line is :" << std::endl;
	  }
	  std::cout <<tensor_d.host_data(i) << " ";
  }
  //std::ofstream outfile("out.txt");
  //if(!outfile.is_open()) {
  //  std::cerr << "not open file!" << std::endl;
  //  return 1;
  //}
  //outfile << "tensor_a.size is " << tensor_a.size() << "\ntensor_b.size is " << tensor_b.size() << "\ntensor_c.size is " << tensor_c.size() << "\ntensor_d.size is " << tensor_d.size() << std::endl;
  //outfile.close();
  if(result != 0) { 
    std::cerr << "Error occured during matrix multipilication." << std::endl;
    return -1;
  }
  }
}

