/**
 * @file cube_group_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/cube_group/cube_group_intf.h"

using namespace AscendC;
using namespace matmul;
const uint32_t BOLCKSTART = 6*2;
const uint32_t BOLCKSIZE= 6*2;
const uint32_t MSGQUEUESIZE = 48;
const uint32_t GROUPID = 1;

struct CubeMsgBody
{
    CubeGroupMsgHead head;
    uint8_t funcId;
    uint8_t reservedHead1;
    uint32_t reservedHead2;
    bool isTransA;
    bool isTransB;
    bool isAtomic;
    bool isLast;
    int32_t Time;
    int32_t tailN;
    int32_t tailK; 
    int64_t aivStartTime;
    uint64_t aAddr;
    uint64_t bAddr;
    uint64_t cAddr;
    uint64_t aGap;
    uint64_t bGap;
    uint64_t cGap;
    int64_t mOrgShape;
    int64_t nOrgShape;
    int64_t kaOrgShape;
    int64_t kbOrgShape;
    int64_t kcOrgShape;
    bool enPartialSum;
    uint16_t reserved;
};

template<class MatmulApiCfg, typename CubeMsgBody>
struct MyCallbackFunc
{
    template<int32_t funcId>
    __aicore__ inline static typename IsEqual<funcId, 0>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        GlobalTensor<int64_t> msgGlobal;
        msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
        using SrcAT = typename MatmulApiCfg::AType::T;
        auto skipNum = 0;
        for (int i = 0; i < skipNum + 1; ++i)
        {
            auto tmpId = handle.FreeMessage(rcvMsg + i);
        }
        handle.SetSkipMsg(skipNum);
    }
    template<int32_t funcId>
    __aicore__ inline static typename IsEqual<funcId, 1>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        GlobalTensor<int64_t> msgGlobal;
        msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
        using SrcAT = typename MatmulApiCfg::AType::T;
        LocalTensor<SrcAT> tensor_temp;
        auto skipNum = 3;
        auto tmpId = handle.FreeMessage(rcvMsg, CubeMsgState::VALID);
        for (int i = 1; i < skipNum + 1; ++i)
        {
            auto tmpId = handle.FreeMessage(rcvMsg + i, CubeMsgState::FAKE);
        }
        handle.SetSkipMsg(skipNum);
    }
    __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        if (rcvMsg->funcId == 0)
        {
            CubeGroupCallBack<0> (mm, rcvMsg, handle);
        }
        else if(rcvMsg->funcId == 1)
        {
            CubeGroupCallBack<1> (mm, rcvMsg, handle);
        }
    }
    __aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM)
    {
        auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
        auto tempTiling = (uint32_t*)&(foo.tiling);
        for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling)
        {
            *tempTiling = *tempTilingGM;
        }
        mm.SetSubBlockIdx(0);
        mm.Init(&foo.tiling, GetTPipePtr());
    }
    TCubeTiling tiling;
};


template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CubeGroupKernel {
public:
    __aicore__ inline CubeGroupKernel(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGm, GM_ADDR biasGM, 
                                GM_ADDR tilingGM, GM_ADDR workspaceGM, uint32_t isTransposeAIn, uint32_t isTransposeBIn)
    {
    KfcWorkspace desc(workspaceGM);
    using MatmulApiType = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, C_TYPE, CFG_NORM>;
    handle = CreateCubeResGroup<GROUPID, MatmulApiType, MyCallbackFunc, CubeMsgBody> (desc, BOLCKSTART, BOLCKSIZE, MSGQUEUESIZE, tilingGM);
    };

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC 
        {
            return;
        }
        auto queIdx = GetBlockIdx();
        handle.AssignQueue(queIdx);
        CubeGroupMsgHead head = {CubeMsgState::VALID, (uint8_t)queIdx};
        CubeMsgBody acubeMsgbody {head, 0, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, false, 0};
        CubeMsgBody bcubeMsgbody {head, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, false, 0};
        auto id = 0;
        if (GetBlockIdx() == 0)
        {
            auto msg = handle.template AllocMessage();
            bcubeMsgbody.aivStartTime = GetSystemCycle();
            id = handle.template PostMessage(msg, bcubeMsgbody);
            bool waitState = handle.template Wait<true> (id);
        }
        else if (GetBlockIdx() < 4)
        {
            auto msg = handle.AllocMessage();
            id = handle.PostFakeMsg(msg);
            bool waitState = handle.template Wait<true> (id);
        }
        else
        {
            auto msg = handle.template AllocMessage();
            id = handle.template PostMessage(msg, acubeMsgbody);
            bool waitState = handle.template Wait<true> (id);
        }
        auto msg1 = handle.AllocMessage();
        handle.SetQuit(msg1);
    };

private:
    TPipe pipe;
    CubeResGroupHandle<CubeMsgBody> handle;
};


extern "C" __global__ __aicore__ void cube_group_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    typename MatmulType<TPosition::GM, CubeFormat::ND, half> aType;
    typename MatmulType<TPosition::GM, CubeFormat::ND, half> bType;
    typename MatmulType<TPosition::LCM, CubeFormat::ND, float> cType;
    typename MatmulType<TPosition::GM, CubeFormat::ND, float> biasType;
    CubeGroupKernel<aType, bType, cType, biasType> op;
    op.Init(a, b, c, bias, tiling, workspace, 0, 0);
    op.Process(); 
}