/*
 * diompgroup.cpp - Group management implementation for DiOMP
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "diomp.h"
#include "tools.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>

#ifdef DIOMP_ENABLE_CUDA
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#ifdef DIOMP_ENABLE_HIP
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#endif

// Group creation function
int ompx_group_create(ompx_group_t **group, int *ranks, int size) {
    if (!group || !ranks || size <= 0) {
        return -1; // Invalid parameters
    }

    // Allocate group structure
    ompx_group_t *new_group = (ompx_group_t*)malloc(sizeof(ompx_group_t));
    if (!new_group) {
        return -1; // Memory allocation failed
    }

    // Allocate and copy ranks array
    new_group->ranks = (int*)malloc(size * sizeof(int));
    if (!new_group->ranks) {
        free(new_group);
        return -1; // Memory allocation failed
    }

    memcpy(new_group->ranks, ranks, size * sizeof(int));
    new_group->size = size;

    // Find my rank within the group
    int my_global_rank = omp_get_rank_num();
    new_group->rank = -1; // Not in group by default
    
    for (int i = 0; i < size; i++) {
        if (ranks[i] == my_global_rank) {
            new_group->rank = i;
            break;
        }
    }

    // Create GASNet team using color/key approach (similar to MPI_Comm_split)
    // All ranks in the group get the same color (1), others get different color (0)
    int color = 0;  // Default: not in group
    int key = my_global_rank;  // Use global rank as key for ordering
    
    // Check if this rank is in the group
    for (int i = 0; i < size; i++) {
        if (ranks[i] == my_global_rank) {
            color = 1;  // This rank is in the group
            key = i;    // Use group position as key
            break;
        }
    }

    gex_TM_t new_team;
    int result = gex_TM_Split(&new_team, diompTeam, color, key, 
                             NULL, 0, 0);

    if (result != GASNET_OK) {
        free(new_group->ranks);
        free(new_group);
        return -1; // Team creation failed
    }

    new_group->team = new_team;

    // Initialize device communicator fields
#ifdef DIOMP_ENABLE_CUDA
    new_group->nccl_initialized = false;
    new_group->nccl_comm = nullptr;
    new_group->nccl_stream = nullptr;
    new_group->nccl_comms = nullptr;
    new_group->nccl_streams = nullptr;
    new_group->devices_num = 0;
#endif

#ifdef DIOMP_ENABLE_HIP
    new_group->rccl_initialized = false;
    new_group->rccl_comm = nullptr;
    new_group->rccl_stream = nullptr;
    new_group->rccl_comms = nullptr;
    new_group->rccl_streams = nullptr;
    new_group->devices_num = 0;
#endif

    *group = new_group;
    return 0; // Success
}

// Group destruction function
int ompx_group_destroy(ompx_group_t *group) {
    if (!group) {
        return -1; // Invalid parameter
    }

    // Cleanup device communicators
#ifdef DIOMP_ENABLE_CUDA
    if (group->nccl_initialized) {
        if (group->nccl_comms) {
            for (int i = 0; i < group->devices_num; i++) {
                if (group->nccl_comms[i] != nullptr) {
                    ncclCommDestroy(group->nccl_comms[i]);
                }
                if (group->nccl_streams[i] != nullptr) {
                    cudaStreamDestroy(group->nccl_streams[i]);
                }
            }
            free(group->nccl_comms);
            free(group->nccl_streams);
        } else if (group->nccl_comm != nullptr) {
            ncclCommDestroy(group->nccl_comm);
            if (group->nccl_stream != nullptr) {
                cudaStreamDestroy(group->nccl_stream);
            }
        }
    }
#endif

#ifdef DIOMP_ENABLE_HIP
    if (group->rccl_initialized) {
        if (group->rccl_comms) {
            for (int i = 0; i < group->devices_num; i++) {
                if (group->rccl_comms[i] != nullptr) {
                    ncclCommDestroy(group->rccl_comms[i]);
                }
                if (group->rccl_streams[i] != nullptr) {
                    hipStreamDestroy(group->rccl_streams[i]);
                }
            }
            free(group->rccl_comms);
            free(group->rccl_streams);
        } else if (group->rccl_comm != nullptr) {
            ncclCommDestroy(group->rccl_comm);
            if (group->rccl_stream != nullptr) {
                hipStreamDestroy(group->rccl_stream);
            }
        }
    }
#endif

    // Destroy GASNet team
    if (group->team != GEX_TM_INVALID) {
        gex_TM_Destroy(group->team, NULL, 0);
    }

    // Free allocated memory
    if (group->ranks) {
        free(group->ranks);
    }
    free(group);
    
    return 0; // Success
}

// Get group size
int ompx_group_size(ompx_group_t *group) {
    if (!group) {
        return -1;
    }
    return group->size;
}

// Get rank within group
int ompx_group_rank(ompx_group_t *group) {
    if (!group) {
        return -1;
    }
    return group->rank;
}

#ifdef DIOMP_ENABLE_CUDA
// Initialize NCCL communicators for group
extern "C" int init_group_nccl_comms(ompx_group_t *group, int devices_num) {
    if (!group || group->nccl_initialized) {
        return -1;
    }

    group->devices_num = devices_num;

    if (devices_num == 1) {
        // Single device per process mode
        ncclUniqueId id;
        if (group->rank == 0) {
            NCCLCHECK(ncclGetUniqueId(&id));
        }

        // Broadcast NCCL unique ID across group
        gex_Event_Wait(gex_Coll_BroadcastNB(group->team, 0, &id, &id, sizeof(ncclUniqueId), 0));

        NCCLCHECK(ncclCommInitRank(&group->nccl_comm, group->size, id, group->rank));
        CUDACHECK(cudaStreamCreate(&group->nccl_stream));
    } else {
        // Multiple devices per process mode - fallback to individual communicator creation
        group->nccl_comms = (ncclComm_t*)malloc(devices_num * sizeof(ncclComm_t));
        group->nccl_streams = (cudaStream_t*)malloc(devices_num * sizeof(cudaStream_t));
        
        if (!group->nccl_comms || !group->nccl_streams) {
            if (group->nccl_comms) free(group->nccl_comms);
            if (group->nccl_streams) free(group->nccl_streams);
            return -1;
        }

        // For multi-device mode, create individual communicators
        // This is a simplified approach - in production, you'd want proper multi-GPU coordination
        for (int i = 0; i < devices_num; i++) {
            ncclUniqueId id;
            if (group->rank == 0) {
                NCCLCHECK(ncclGetUniqueId(&id));
            }
            
            // Broadcast NCCL unique ID across group for each device
            gex_Event_Wait(gex_Coll_BroadcastNB(group->team, 0, &id, &id, sizeof(ncclUniqueId), 0));
            
            // Initialize communicator for this device
            NCCLCHECK(ncclCommInitRank(&group->nccl_comms[i], group->size, id, group->rank));
            CUDACHECK(cudaStreamCreate(&group->nccl_streams[i]));
        }
    }

    group->nccl_initialized = true;
    return 0;
}
#endif

#ifdef DIOMP_ENABLE_HIP
// Initialize RCCL communicators for group
extern "C" int init_group_rccl_comms(ompx_group_t *group, int devices_num) {
    if (!group || group->rccl_initialized) {
        return -1;
    }

    group->devices_num = devices_num;

    if (devices_num == 1) {
        // Single device per process mode
        ncclUniqueId id;
        if (group->rank == 0) {
            RCCLCHECK(ncclGetUniqueId(&id));
        }

        // Broadcast RCCL unique ID across group
        gex_Event_Wait(gex_Coll_BroadcastNB(group->team, 0, &id, &id, sizeof(ncclUniqueId), 0));

        RCCLCHECK(ncclCommInitRank(&group->rccl_comm, group->size, id, group->rank));
        HIPCHECK(hipStreamCreate(&group->rccl_stream));
    } else {
        // Multiple devices per process mode - fallback to individual communicator creation
        group->rccl_comms = (ncclComm_t*)malloc(devices_num * sizeof(ncclComm_t));
        group->rccl_streams = (hipStream_t*)malloc(devices_num * sizeof(hipStream_t));
        
        if (!group->rccl_comms || !group->rccl_streams) {
            if (group->rccl_comms) free(group->rccl_comms);
            if (group->rccl_streams) free(group->rccl_streams);
            return -1;
        }

        // For multi-device mode, create individual communicators
        // This is a simplified approach - in production, you'd want proper multi-GPU coordination
        for (int i = 0; i < devices_num; i++) {
            ncclUniqueId id;
            if (group->rank == 0) {
                RCCLCHECK(ncclGetUniqueId(&id));
            }
            
            // Broadcast RCCL unique ID across group for each device
            gex_Event_Wait(gex_Coll_BroadcastNB(group->team, 0, &id, &id, sizeof(ncclUniqueId), 0));
            
            // Initialize communicator for this device
            RCCLCHECK(ncclCommInitRank(&group->rccl_comms[i], group->size, id, group->rank));
            HIPCHECK(hipStreamCreate(&group->rccl_streams[i]));
        }
    }

    group->rccl_initialized = true;
    return 0;
}
#endif