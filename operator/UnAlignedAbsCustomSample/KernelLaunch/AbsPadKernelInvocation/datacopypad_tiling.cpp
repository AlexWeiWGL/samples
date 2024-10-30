#include "graph/tensor.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

uint8_t *GetTilingBuf(optiling::PadTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}

uint8_t *GenerateTiling(const std::vector<int64_t> shapePad, const std::vector<int64_t> shapeUsed)
{
    ge::Shape srcShape(shapePad);
    ge::Shape oriSrcShape(shapeUsed);
    uint32_t tmpMinSize, tmpMaxSize;
    AscendC::GetPadMaxMinTmpSize(srcShape, sizeof(int16_t), tmpMaxSize, tmpMinSize); 
    optiling::PadTiling tilingData;
    AscendC::PadTilingFunc(srcShape, oriSrcShape, tmpMaxSize, sizeof(int16_t), tilingData);
    return GetTilingBuf(&tilingData);
}
