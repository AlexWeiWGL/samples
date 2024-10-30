#include "kernel_operator.h"
constexpr int32_t BLOCK_BYTE_SIZE = 8; // equivalent to the definition of blockLen of DataCopyPad
constexpr int32_t BLOCK_GROUP_NUM = 4; // equivalent to the definition of blockCount of DataCopyPad
constexpr int32_t BLOCK_ELEMENT_NUM = BLOCK_BYTE_SIZE / sizeof(half);
constexpr int32_t BLOCKLEN_CEIL = 32 / sizeof(half); // since BLOCK_BYTE_SIZE<32
constexpr int32_t USE_CORE_NUM = 4;                  // num of core used
constexpr int32_t TILE_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TOTAL_LENGTH = USE_CORE_NUM * TILE_NUM * BUFFER_NUM * BLOCK_GROUP_NUM * BLOCK_ELEMENT_NUM;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // tensor num for each queue
class KernelReduceMin {
public:
    __aicore__ inline KernelReduceMin() {}
    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputGM)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        dstGlobal.SetGlobalBuffer((__gm__ half *)(outputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(zeroQueue, BUFFER_NUM, 32);
        pipe.InitBuffer(workQueue, BUFFER_NUM, 32);
    }
    __aicore__ inline void Process()
    {
        // loop count need to be doubled, due to double buffer
        const int32_t loopCount = TILE_NUM * BUFFER_NUM;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> zeroTensor = zeroQueue.AllocTensor<half>();
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy(inputLocal[i * BLOCKLEN_CEIL], srcGlobal[i * BLOCK_ELEMENT_NUM],
                              BLOCKLEN_CEIL); // each time copy 16 half elements to UB
        }
        inQueue.EnQue(inputLocal);
        zeroQueue.EnQue(zeroTensor);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> workLocal = workQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> inputLocal = inQueue.DeQue<half>();
        AscendC::LocalTensor<half> zeroTensor = zeroQueue.DeQue<half>();
        AscendC::Duplicate<half>(zeroTensor, 0, 32 / sizeof(half)); // set an all 0 tensor
        zeroQueue.EnQue(zeroTensor);
        zeroTensor = zeroQueue.DeQue<half>();
        // clear dstGM before doing calculations
        AscendC::DataCopy<half>(dstGlobal, zeroTensor, TILE_LENGTH);
        outQueue.EnQue<half>(outputLocal);
        outputLocal = outQueue.DeQue<half>();
        AscendC::Duplicate<half>(outputLocal, 0, BLOCK_GROUP_NUM * BLOCKLEN_CEIL);
        outQueue.EnQue<half>(outputLocal);
        outputLocal = outQueue.DeQue<half>();
        uint64_t Mask0 = ((uint64_t)1 << BLOCK_ELEMENT_NUM) -
                         1; // mask mode controls only the first 4 elements do ReduceMin calculation
        uint64_t Mask[2] = {Mask0, 0};
        // main calculation
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::ReduceMin<half>(outputLocal[i * BLOCKLEN_CEIL], inputLocal[i * BLOCKLEN_CEIL], workLocal, Mask, 1,
                                     8, false);
        }
        outQueue.EnQue<half>(outputLocal);
        inQueue.FreeTensor(inputLocal);
        workQueue.FreeTensor(workLocal);
        zeroQueue.FreeTensor(zeroTensor);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.DeQue<half>();
        AscendC::SetAtomicAdd<half>();
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy<half>(dstGlobal[i * BLOCK_ELEMENT_NUM], outputLocal[i * BLOCKLEN_CEIL], BLOCKLEN_CEIL);
        }
        AscendC::SetAtomicNone();
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> workQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zeroQueue;
};
extern "C" __global__ __aicore__ void reduce_min_custom(GM_ADDR inputGM, GM_ADDR outputGM)
{
    KernelReduceMin op;
    op.Init(inputGM, outputGM);
    op.Process();
}