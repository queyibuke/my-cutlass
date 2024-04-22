/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "../../common/cutlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// TT //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16t_f16n_align8_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

TEST(SM90_Device_Gemm_f16t_f16t_f16n_align4_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 4,
      cutlass::half_t, LayoutB, 4,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 4,
      cutlass::half_t, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}


TEST(SM90_Device_Gemm_f16t_f16t_f16n_align2_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 2,
      cutlass::half_t, LayoutB, 2,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 2,
      cutlass::half_t, LayoutC, 2,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// TN //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_align8_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_align4_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 4,
      cutlass::half_t, LayoutB, 4,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 4,
      cutlass::half_t, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16t_f16n_f16n_align2_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 2,
      cutlass::half_t, LayoutB, 2,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 2,
      cutlass::half_t, LayoutC, 2,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// NT //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f16n_align8_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f16n_align4_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 4,
      cutlass::half_t, LayoutB, 4,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 4,
      cutlass::half_t, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16t_f16n_align2_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 2,
      cutlass::half_t, LayoutB, 2,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 2,
      cutlass::half_t, LayoutC, 2,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// NN //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f16n_align8_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 8,
      cutlass::half_t, LayoutB, 8,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 8,
      cutlass::half_t, LayoutC, 8,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f16n_align4_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 4,
      cutlass::half_t, LayoutB, 4,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 4,
      cutlass::half_t, LayoutC, 4,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

TEST(SM90_Device_Gemm_f16n_f16n_f16n_align2_tensor_op_gmma_f32_warpspecialized_cooperative, 128x128x64) {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::half_t, LayoutA, 2,
      cutlass::half_t, LayoutB, 2,
      float,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      cutlass::half_t, LayoutC, 2,
      cutlass::half_t, LayoutC, 2,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveOp,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

///////////////////////////////////////////////////////////////////////////////

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
