// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_GRAPH_CSE_H_
#define DALI_PIPELINE_GRAPH_CSE_H_

#include "dali/pipeline/graph/op_graph2.h"

namespace dali {
namespace graph {

/** Eliminate Common Subgraphs
 *
 * Runs a common subexpression (subgraph) analysis on the graph.
 * The graph is completely rewritten in the process.
 *
 * The algorithm works by traversing the original graph in topological order.
 * Each OpSpec is first updated by renaming the inputs to match the previously merged nodes.
 * If the update OpSpec was already seen, then the OpSpec is replaced and the output names
 * are added to the renaming map.
 *
 * op1(args1) --- out1_A -- op2(args2)
 *            \            /
 *             --- out1_B -
 *
 * op2(args1) --- out1_A -- op2(args2)
 *            \            /
 *             --- out1_B -
 *
 * To identify matchin operators, a key is computed which consists of the OpSpec's schema name,
 * arguments, inputs and output devices (but NOT output names!).
 *
 * If the key matches one previously seen, the operators are assumed equal.
 *
 * Some arguments are ignored - notably, the ones identifying the source location in Python
 * (that would make any kind of CSE pointless).
 *
 * There are some operators which are never merged:
 * - ExternalSource
 * - operators with explicitly given name
 * - operators with "preserve" argument set
 * - operators with NoPrune schema
 *
 */
void EliminateCommonSubgraphs(OpGraph &graph);

}  // namespace graph
}  // namespace dali


#endif  // DALI_PIPELINE_GRAPH_CSE_H_
