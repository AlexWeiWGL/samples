/**
 *  Copyright [2021] Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/time.h>
#include <fcntl.h>
#include <cerrno>
#include <pthread.h>
#include <cmath>
#include <unistd.h>
#include <csignal>
#include "common.h"
#include <vector>

using namespace std;

int32_t main(int32_t argc, char *argv[])
{
    int32_t s32Ret = HI_SUCCESS;
    pthread_t jpegdSendThreadTid[VDEC_MAX_CHN_NUM] = {0};
    pthread_t jpegdGetThreadTid[VDEC_MAX_CHN_NUM] = {0};

    signal(SIGINT, jpegd_handle_signal);
    signal(SIGTERM, jpegd_handle_signal);

    // Get input parameters
    s32Ret = set_param();
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("set_param failed!\n");
        return HI_FAILURE;
    }

    // Management Resource Request
    s32Ret = setup_acl_device();
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("Setup Device failed! ret code:%#x\n", s32Ret);
        return HI_FAILURE;
    }

    //initialize himpi
    s32Ret = hi_mpi_sys_init();
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("hi_mpi_sys_init failed!\n");
        return HI_FAILURE;
    }

    // Call the himpi interface to create a channel and 
    // use the decoder to start receiving user sent bitstreams before starting decoding
    s32Ret = jpegd_create();
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("start JPEGD fail for %#x!\n", s32Ret);
        int32_t exitRet = destory_resource_jpegd_mod();
        if (exitRet != HI_SUCCESS) {
            SAMPLE_PRT("exit JPEGD fail for %#x!\n", exitRet);
        }
        return s32Ret;
    }

    // Start sending code stream
    s32Ret = jpegd_start_send_stream(&jpegdSendThreadTid[0]);
    if (s32Ret != 0) {
        jpegd_stop_send_stream();
    } else {
        s32Ret = jpegd_start_get_pic(&jpegdGetThreadTid[0]);
        if (s32Ret != 0) {
            jpegd_stop_send_stream();
            jpegd_stop_get_pic();
        }
    }

    // Waiting for thread to end
    jpegd_cmd_ctrl(&jpegdSendThreadTid[0], &jpegdGetThreadTid[0]);

    // Print fps for each channel
    jpegd_show_decode_state();

    s32Ret = destory_resource_jpegd_mod();
    if (s32Ret != HI_SUCCESS) {
        SAMPLE_PRT("exit JPEGD fail for %#x!\n", s32Ret);
    }

    return s32Ret;
}

