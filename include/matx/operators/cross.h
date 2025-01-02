////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once


#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{

  /**
   * Returns the polynomial evaluated at each point
   */
  namespace detail {
    template <typename OpA, typename OpB>
    class CrossOp : public BaseOp<CrossOp<OpA, OpB>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        typename detail::base_type_t<OpB> b_;

        static constexpr int32_t out_rank = cuda::std::max(OpA::Rank(), OpB::Rank());
        static constexpr int32_t min_rank = cuda::std::min(OpA::Rank(), OpB::Rank());

        cuda::std::array<index_t, out_rank> out_dims_;

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;

        __MATX_INLINE__ std::string str() const { return "cross()"; }
        __MATX_INLINE__ CrossOp(const OpA &OpA, const OpB &OpB) : OpA_(OpA), OpB_(OpB) {
          MATX_STATIC_ASSERT_STR(OpA::Rank() >= 1 && OpB::Rank() >= 1, matxInvalidDim, "Operators to cross() must have rank of at least one.");

          for (int32_t i = 0; i < min_rank; i++) {
            MATX_ASSERT_STR(a_.Size(OpA::Rank() - 1 - i) == b_.Size(OpB::Rank() - 1 - i) , matxInvalidSize, "Operators to cross() must have equal sizes in all dimensions, beginning from the furthest right.");
          }

          MATX_ASSERT_STR(a_.Size(OpA::Rank() - 1) == 3 || a_.Size(OpA::Rank() - 1) == 2, matxInvalidSize, "Operator A to cross() must have size 2 or 3.")
          MATX_ASSERT_STR(b_.Size(OpB::Rank() - 1) == 3 || b_.Size(OpB::Rank() - 1) == 2, matxInvalidSize, "Operator A to cross() must have size 2 or 3.")
        };

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          cuda::std::array idxA{indices...};
          cuda::std::array idxB{indices...};

          
          idx2[Rank() - 2] = pp_get<Rank() - 2>(indices...) % op2_.Size(Rank() - 2);
          idx2[Rank() - 1] = pp_get<Rank() - 1>(indices...) % op2_.Size(Rank() - 1);

          idx1[Rank() - 2] = pp_get<Rank() - 2>(indices...) / op2_.Size(Rank() - 2);
          idx1[Rank() - 1] = pp_get<Rank() - 1>(indices...) / op2_.Size(Rank() - 1);

          return get_value(op2_, idx2) * get_value(op1_, idx1);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return out_rank;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          return cuda:;std::max(a_.Size(dim), b_.Size(dim));
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<OpB>()) {
            b_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }          
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }     

          if constexpr (is_matx_op<OpB>()) {
            b_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          } 
        }      
    };
  }


  /**
   * @brief Evaluate a cross product
   *  
   * @tparam OpA Type of input tensor 1
   * @tparam OpB Type of input tensor 2
   * @param OpA Input tensor 1
   * @param OpB Input tensor 2
   * @return cross operator 
   */
  template <typename OpA, typename OpB>
  __MATX_INLINE__ auto cross(const OpA &OpA, const OpB &OpB) {
    return detail::CrossOp(OpA, OpB);
  }
} // end namespace matx
