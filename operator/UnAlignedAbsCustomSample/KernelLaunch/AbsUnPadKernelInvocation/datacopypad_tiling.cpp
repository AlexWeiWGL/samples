#include "graph/tensor.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

uint8_t *GetTilingBuf(optiling::UnPadTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}

uint8_t *GenerateTiling(const std::vector<int64_t> shape, const char *socVersion)
{
    platform_ascendc::PlatformAscendC *ascendcPlatform;
    if (socVersion != nullptr) {
        ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    }
    else{
        ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    }

    ge::Shape srcShape(shape);
    uint32_t tmpMinSize, tmpMaxSize;
    std::vector<int64_t> dstShape = srcShape.GetDims();

    AscendC::GetUnPadMaxMinTmpSize(*ascendcPlatform, srcShape, sizeof(int16_t), tmpMaxSize, tmpMinSize);
    optiling::UnPadTiling tilingData;
    AscendC::UnPadTilingFunc(srcShape, tmpMaxSize, sizeof(int16_t), tilingData);
    return GetTilingBuf(&tilingData);
}
