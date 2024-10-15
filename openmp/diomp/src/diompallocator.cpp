#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <stdio.h>


#define TLSF_GRAIN_LOG 5  // 32-byte minimum allocation
#define TLSF_MAX_LOG2_SIZE 32
#define TLSF_MAX_LEVELS 32
#define TLSF_SL_SHIFT 3
#define TLSF_SL_SIZE (1 << TLSF_SL_SHIFT)

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

typedef struct TLSF {
    uint32_t FlBitmap;
    uint32_t SlBitmap[TLSF_MAX_LEVELS];
    uint64_t Blocks[TLSF_MAX_LEVELS][TLSF_SL_SIZE];
} TLSF;

typedef struct BlockHeader {
    uint64_t Size;
    uint64_t PrevPhysBlock;
} BlockHeader;

typedef struct TLSFManager {
    TLSF Allocator;
    char* Base;
    size_t Capacity;
} TLSFManager;

static TLSFManager TlsfManagers[8];

static inline int FindLastSet(uint32_t X) {
    return X ? 32 - __builtin_clz(X) : 0;
}

static inline int FindFirstSet(uint32_t X) {
    return __builtin_ffs(X) - 1;
}

static void MapSize(size_t Size, int* Fl, int* Sl) {
    if (Size < (1ULL << TLSF_MAX_LOG2_SIZE)) {
        *Fl = FindLastSet(Size);
        *Sl = (Size >> (*Fl - TLSF_SL_SHIFT)) & (TLSF_SL_SIZE - 1);
    } else {
        *Fl = TLSF_MAX_LEVELS - 1;
        *Sl = TLSF_SL_SIZE - 1;
    }
}

static void InsertFreeBlock(TLSFManager* Manager, uint64_t BlockOffset, size_t Size) {
    int Fl, Sl;
    MapSize(Size, &Fl, &Sl);
    
    *(uint64_t*)(Manager->Base + BlockOffset) = Manager->Allocator.Blocks[Fl][Sl];
    Manager->Allocator.Blocks[Fl][Sl] = BlockOffset;
    Manager->Allocator.FlBitmap |= 1U << Fl;
    Manager->Allocator.SlBitmap[Fl] |= 1U << Sl;
}

static void RemoveFreeBlock(TLSFManager* Manager, uint64_t BlockOffset, int Fl, int Sl) {
    uint64_t* Current = &Manager->Allocator.Blocks[Fl][Sl];
    while (*Current != BlockOffset) {
        Current = (uint64_t*)(Manager->Base + *Current);
    }
    *Current = *(uint64_t*)(Manager->Base + BlockOffset);
    if (!Manager->Allocator.Blocks[Fl][Sl]) {
        Manager->Allocator.SlBitmap[Fl] &= ~(1U << Sl);
        if (!Manager->Allocator.SlBitmap[Fl]) {
            Manager->Allocator.FlBitmap &= ~(1U << Fl);
        }
    }
}

uint64_t TLSFAlloc(TLSFManager* Manager, size_t Size) {
    Size = MAX(Size + sizeof(BlockHeader), (1 << TLSF_GRAIN_LOG));
    int Fl, Sl;
    MapSize(Size, &Fl, &Sl);

    uint32_t SlMap = Manager->Allocator.SlBitmap[Fl] & (~0U << Sl);
    if (!SlMap) {
        uint32_t FlMap = Manager->Allocator.FlBitmap & (~0U << (Fl + 1));
        if (!FlMap) return 0;  // Out of memory
        Fl = FindFirstSet(FlMap);
        SlMap = Manager->Allocator.SlBitmap[Fl];
    }
    Sl = FindFirstSet(SlMap);

    uint64_t BlockOffset = Manager->Allocator.Blocks[Fl][Sl];
    RemoveFreeBlock(Manager, BlockOffset, Fl, Sl);

    BlockHeader* Header = (BlockHeader*)(Manager->Base + BlockOffset);
    size_t BlockSize = Header->Size;
    if (BlockSize - Size >= (1 << TLSF_GRAIN_LOG)) {
        BlockHeader* NewBlock = (BlockHeader*)(Manager->Base + BlockOffset + Size);
        NewBlock->Size = BlockSize - Size;
        NewBlock->PrevPhysBlock = Size;
        InsertFreeBlock(Manager, BlockOffset + Size, NewBlock->Size);
        Header->Size = Size;
    }

    return BlockOffset + sizeof(BlockHeader);
}

void TLSFFree(TLSFManager* Manager, uint64_t Offset) {
    BlockHeader* Header = (BlockHeader*)(Manager->Base + Offset - sizeof(BlockHeader));
    size_t Size = Header->Size;

    // Try to merge with the next block
    BlockHeader* NextBlock = (BlockHeader*)(Manager->Base + Offset - sizeof(BlockHeader) + Size);
    if (!(NextBlock->Size & 1)) {  // Next block is free
        RemoveFreeBlock(Manager, Offset - sizeof(BlockHeader) + Size, 0, 0);  // Remove from free list
        Size += NextBlock->Size;
        Header->Size = Size;
    }

    // Try to merge with the previous block
    if (Header->PrevPhysBlock) {
        BlockHeader* PrevBlock = (BlockHeader*)(Manager->Base + Offset - sizeof(BlockHeader) - Header->PrevPhysBlock);
        if (!(PrevBlock->Size & 1)) {  // Previous block is free
            RemoveFreeBlock(Manager, Offset - sizeof(BlockHeader) - Header->PrevPhysBlock, 0, 0);  // Remove from free list
            Size += PrevBlock->Size;
            PrevBlock->Size = Size;
            Header = PrevBlock;
        }
    }

    InsertFreeBlock(Manager, (char*)Header - Manager->Base, Size);
}

void ShmemaTLSFInit(void *Base, size_t Capacity, int8_t Id) {
    TlsfManagers[Id].Base = (char*)Base;
    TlsfManagers[Id].Capacity = Capacity;
    memset(&TlsfManagers[Id].Allocator, 0, sizeof(TLSF));
    
    // Initialize the entire memory as one big free block
    BlockHeader* InitialBlock = (BlockHeader*)Base;
    InitialBlock->Size = Capacity;
    InitialBlock->PrevPhysBlock = 0;
    TLSFFree(&TlsfManagers[Id], sizeof(BlockHeader));
}

void ShmemaTLSFFinalize(int8_t Id) {
    // No need to free anything, as we're assuming the GPU memory is managed externally
}

uint64_t ShmemaTLSFMalloc(size_t Size, int8_t Id) {
    return TLSFAlloc(&TlsfManagers[Id], Size);
}

void ShmemaTLSFFree(uint64_t Offset, int8_t Id) {
    TLSFFree(&TlsfManagers[Id], Offset);
}
int main()
{
    size_t Capacity = 1 << 10;
    void* Base = omp_target_alloc(Capacity, 0);
    
    ShmemaTLSFInit(Base, Capacity, 0);
    printf("address of base %llu\n", Base);
    uint64_t Ptr1 = ShmemaTLSFMalloc(125, 0);
    printf("address of Ptr1 %llu\n", Ptr1);
    assert(Ptr1 != 0);
    uint64_t Ptr2 = ShmemaTLSFMalloc(256, 0);
    printf("address of Ptr2 %llu\n", Ptr2);
    assert(Ptr2 != 0);
    uint64_t Ptr3 = ShmemaTLSFMalloc(512, 0);
    assert(Ptr3 != 0);
    printf("address of Ptr3 %llu\n", Ptr3);

    // ShmemaTLSFFree(Ptr2, 0);
    // ShmemaTLSFFree(Ptr1, 0);
    // ShmemaTLSFFree(Ptr3, 0);

    return 0;
}