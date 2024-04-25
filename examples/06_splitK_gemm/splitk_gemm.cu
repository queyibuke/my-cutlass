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
#include <fstream>

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputG = cutlass::half_t;              // <- data type of elements in input matrix G
using ElementInputW1 = cutlass::half_t;              // <- data type of elements in input matrix W1
using ElementInputX = cutlass::half_t;              // <- data type of elements in input matrix X
using ElementOutput1 = float;                        // <- data type of elements in output matrix O1
using ElementOutput2 = float;                        // <- data type of elements in output matrix O2

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputG = cutlass::layout::ColumnMajor;
using LayoutInputW1 = cutlass::layout::ColumnMajor;
using LayoutInputX = cutlass::layout::RowMajor;
using LayoutOutput1 = cutlass::layout::RowMajor;
using LayoutOutput2 = cutlass::layout::RowMajor;

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
using EpilogueOp = cutlass::epilogue::thread::LinearCombination< ElementOutput1,// <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput1>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

 using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput2,// <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput2>::value,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function   

// Put all the created template variables to create GemmSplitKParallel template variable
using GemmO1 = cutlass::gemm::device::GemmSplitKParallel<ElementInputG,
                                                       LayoutInputG,
                                                       ElementInputX,
                                                       LayoutInputX,
                                                       ElementOutput1,
                                                       LayoutOutput1,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

using GemmO2 = cutlass::gemm::device::GemmSplitKParallel<ElementInputW1,
                                                       LayoutInputW1,
                                                       ElementInputX,
                                                       LayoutInputX,
                                                       ElementOutput2,
                                                       LayoutOutput2,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

int run(cutlass::HostTensor<ElementInputG, LayoutInputG>& tensor_g, 
        cutlass::HostTensor<ElementInputW1, LayoutInputW1>& tensor_w1,
        cutlass::HostTensor<ElementInputX, LayoutInputX>& tensor_x,
        cutlass::HostTensor<ElementOutput1, LayoutOutput1>& tensor_c,
        cutlass::HostTensor<ElementOutput2, LayoutOutput2>& tensor_d,
        cutlass::HostTensor<ElementOutput1, LayoutOutput1>& tensor_o1,
        cutlass::HostTensor<ElementOutput2, LayoutOutput2>& tensor_o2,
        ElementComputeEpilogue alpha1,
        ElementComputeEpilogue beta1,
        ElementComputeEpilogue alpha2,
        ElementComputeEpilogue beta2,        
        cutlass::gemm::GemmCoord& problem_size,
        int split_k_slices) {

  // Copy data from host to GPU
  tensor_g.sync_device();
  tensor_w1.sync_device();
  tensor_x.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_o1.sync_device();
  tensor_o2.sync_device();

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename GemmO1::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_g.device_ref(),  // <- reference to matrix G on device
                                     tensor_x.device_ref(),  // <- reference to matrix X on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_o1.device_ref(),  // <- reference to matrix O1 on device
                                     {alpha1, beta1},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
  
  typename GemmO2::Arguments arguments2{problem_size,  // <- problem size of matrix multiplication
                                     tensor_w1.device_ref(),  // <- reference to matrix W1 on device
                                     tensor_x.device_ref(),  // <- reference to matrix X on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     tensor_o2.device_ref(),  // <- reference to matrix O2 on device
                                     {alpha2, beta2},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size1 = GemmO1::get_workspace_size(arguments);
  size_t workspace_size2 = GemmO2::get_workspace_size(arguments2);
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size1);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size2);

  // Instantiate CUTLASS kernel depending on templates
  GemmO1 gemm_op1;
  GemmO2 gemm_op2;

  // Initialize CUTLASS kernel with arguments and workspace pointer
  cutlass::Status status = gemm_op1.initialize(arguments, workspace1.get());
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(arguments2, workspace2.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  status = gemm_op1();
  CUTLASS_CHECK(status);
  status = gemm_op2();
  CUTLASS_CHECK(status);

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS to host for comparison
  tensor_o1.sync_host();
  tensor_o2.sync_host();
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
  const int length_m = 8192;
  const int length_n = 4096;
  const int length_k = 11008;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputG, LayoutInputG> tensor_g(problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputW1, LayoutInputW1> tensor_w1(problem_size.mk());  // <- Create matrix B with dimensions M x K
  cutlass::HostTensor<ElementInputX, LayoutInputX> tensor_x(problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput1, LayoutOutput1> tensor_c(problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput2, LayoutOutput2> tensor_d(problem_size.mn());  // <- Create matrix D with dimensions M x N 
  cutlass::HostTensor<ElementOutput1, LayoutOutput1> tensor_o1(problem_size.mn());  // <- Create matrix D with dimensions M x N
  cutlass::HostTensor<ElementOutput2, LayoutOutput2> tensor_o2(problem_size.mn());  // <- Create matrix D with dimensions M x N
                                                                                 // used to store output from CUTLASS kernel
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha1 = ElementComputeEpilogue(1);//需要激活函数赋值
  ElementComputeEpilogue beta1 = ElementComputeEpilogue(0);
  ElementComputeEpilogue alpha2 = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta2 = ElementComputeEpilogue(0); 

  // Split K dimension into 16 partitions
  int split_k_slices = 16;

  cutlass::reference::host::TensorFill(tensor_g.host_view(), ElementInputG(1));
  cutlass::reference::host::TensorFill(tensor_w1.host_view(), ElementInputW1(2));
  cutlass::reference::host::TensorFill(tensor_x.host_view(), ElementInputX(3));
  cutlass::reference::host::TensorFill(tensor_c.host_view(), ElementOutput1(0));
  cutlass::reference::host::TensorFill(tensor_d.host_view(), ElementOutput2(0));
  cutlass::reference::host::TensorFill(tensor_o1.host_view(), ElementOutput1(0));
  cutlass::reference::host::TensorFill(tensor_o2.host_view(), ElementOutput2(0));

  int result = run(tensor_g, tensor_w1, tensor_x, tensor_c, tensor_d, tensor_o1, tensor_o2, alpha1, beta1, alpha2, beta2, problem_size, split_k_slices);
  
  std::cout << "tensor_o1: " << std::endl;
  std::cout << "the first number is " << tensor_o1.host_data(0) << std::endl;
  std::cout << "the second number is  " << tensor_o1.host_data(1) <<std::endl;
  std::cout << "the lastd number is  " << tensor_o1.host_data(33554431) <<std::endl;
  //for(unsigned int i = 0; i < tensor_o1.size(); i++) {
	  //if((i + 1) / 8192 == 0) {
	  //	std::cout << "\nthe " << i / 8192 << "line is :" << std::endl;
       	  //      std::cout << tensor_o1.host_data(i) << " ";
	  //}
	  //if((i + 1) / 8192 == 4095) {
	  //	std::cout << tensor_o1.host_data(i) << " ";
	  //}
  //}
  std::cout << "\ntensor_o2: " << std::endl;
  //for(unsigned int i = 0; i < tensor_o1.size(); i++) {
  // if((i + 1) / 8192 == 0) {
      //std::cout << "\nthe " << i / 8192 << "line is :" << std::endl; 
  //    std::cout << tensor_o2.host_data(i) << " ";
  //  }
  //}
  //std::cout << std::endl;

  std::cout << "the first number is " << tensor_o2.host_data(0) << std::endl;
  std::cout << "the second number is  " << tensor_o2.host_data(1) <<std::endl;
  std::cout << "the lastd number is  " << tensor_o2.host_data(33554431) <<std::endl;


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

