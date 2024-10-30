/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ADD_ABS_NODE_HPP_
#define ADD_ABS_NODE_HPP_

#include <iostream>
#include "register_custom_pass.h"
#include "all_ops.h"

using namespace std;
using namespace ge;

namespace pass {
constexpr const char *kOpTypeData = "Data";
constexpr const char *kOpTypFrameworkOp = "FrameworkOp";
int32_t kCount = 0;

// |o>-----------------------------------
// |o>      Data              Data
// |o>       |                 |
// |o>       |       ==>      Abs
// |o>       |                 |
// |o>   FrameworkOp       FrameworkOp
// |o>-----------------------------------
// pass修改子图说明：本例识别上图中左边的Data和FrameworkOp节点，并在中间插入Abs节点得到右图
graphStatus AddAbsNodeInSubgraph(GraphPtr &graph, CustomPassContext &custom_context) {
    // 1. 获取子图中的Data和FrameworkOp节点
    GNode src_node;
    GNode dst_node;
    vector<GNode> nodes = graph->GetAllNodes();
    graphStatus ret = GRAPH_FAILED;
    for (auto &node : nodes) {
        AscendString node_type;
        ret = node.GetType(node_type);
        if (ret != GRAPH_SUCCESS) {
            custom_context.SetErrorMessage("Get node type failed.");
            return -1;
        }
        AscendString node_name;
        ret = node.GetName(node_name);
        if (ret != GRAPH_SUCCESS) {
            custom_context.SetErrorMessage("Get node name failed.");
            return -1;
        }
        if (node_type == kOpTypeData) {
            src_node = node;
            cout << "Find src node: " << node_name.GetString() << "." << endl;
        } else if (node_type == kOpTypFrameworkOp) {
            AscendString node_name;
            dst_node = node;
            cout << "Find dst node: " << node_name.GetString() << "." << endl;
        }
    }
    // 2. 删除Data和FrameworkOp节点之间的边，如果没有找到目标节点或者目标节点间无连边，返回成功，无改图
    ret = graph->RemoveEdge(src_node, 0, dst_node, 0);
    if (ret != GRAPH_SUCCESS) {
        cout << "Do not find target nodes or there is no edge between src and dst nodes." << endl;
        return GRAPH_SUCCESS;
    }
    // 3. 在Data和FrameworkOp节点之间插入Abs节点
    string name = "abs_" + to_string(kCount++);
    auto abs = op::Abs(name.c_str());
    GNode node_abs = graph->AddNodeByOp(abs);
    ret = graph->AddDataEdge(src_node, 0, node_abs, 0);
    if (ret != GRAPH_SUCCESS) {
        custom_context.SetErrorMessage("Add data edge failed between const1_0 and abs.");
        return -1;
    }
    ret = graph->AddDataEdge(node_abs, 0, dst_node, 0);
    if (ret != GRAPH_SUCCESS) {
        custom_context.SetErrorMessage("Add data edge failed between abs and const1_RetVal.");
        return -1;
    }
    cout << "Add abs node success." << endl;
    return GRAPH_SUCCESS;
}
} // namespace pass

#endif