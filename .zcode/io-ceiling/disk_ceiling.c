/* Replay one token's routed-expert read shape and split read time from
 * buffer-allocation time, so the disk ceiling and the code's overhead can be
 * told apart.
 *
 * usage: disk_ceiling <plan.tsv> <mode> <direct> <threads>
 *   mode:   reuse        one buffer per size, allocated once per thread
 *           pinned       same per-size cache discipline, but the cached
 *                        buffers are cuMemHostAlloc page-locked memory — the
 *                        exact staging state the engine's expert slots use
 *           fresh        posix_memalign + free on every read
 *           fresh_mmap   mmap + munmap on every read (untouched pages)
 *   direct: 0            buffered pread straight into the destination buffer
 *           1            engine-shaped O_DIRECT: read an aligned superset into a
 *                        fresh aligned buffer, copy the requested window out
 *                        (safetensors.rs::read_direct_aligned)
 * plan.tsv columns: layer<TAB>unit<TAB>path<TAB>offset<TAB>len
 * Reads are assigned to thread (unit % threads), so 6 threads reads the six
 * experts of a layer in parallel. Prints one JSON line.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 4096
#define MAXFILES 64

typedef struct {
    int layer;
    int unit;
    int file;
    long long off;
    long long len;
} Read;

static Read *g_reads;
static long g_nreads;
static char *g_paths[MAXFILES];
static int g_fds[MAXFILES];
static int g_nfiles;
static int g_mode;   /* 0 reuse, 1 fresh, 2 fresh_mmap, 3 pinned */
static int g_direct;
static int g_threads;
static long long g_digest_stride = 512;
static long g_layers;
static pthread_barrier_t g_barrier;

/* CUDA driver entry points for the pinned mode, dlopen'd so every other mode
 * keeps working on a machine without a GPU. cuMemHostAlloc is the same call
 * the engine's PinnedHostBuffer goes through, so the replay measures the real
 * staging state rather than an mlock approximation. */
static void *g_libcuda;
static void *g_cu_ctx;
static int (*p_cuInit)(unsigned int);
static int (*p_cuDeviceGet)(void **, int);
static int (*p_cuCtxCreate)(void **, unsigned int, void *);
static int (*p_cuCtxSetCurrent)(void *);
static int (*p_cuMemHostAlloc)(void **, size_t, unsigned int);
static int (*p_cuMemFreeHost)(void *);

static int init_cuda(void) {
    g_libcuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (g_libcuda == NULL) g_libcuda = dlopen("libcuda.so", RTLD_NOW);
    if (g_libcuda == NULL) {
        fprintf(stderr, "dlopen libcuda: %s\n", dlerror());
        return -1;
    }
    p_cuInit = (int (*)(unsigned int))dlsym(g_libcuda, "cuInit");
    p_cuDeviceGet = (int (*)(void **, int))dlsym(g_libcuda, "cuDeviceGet");
    p_cuCtxCreate = (int (*)(void **, unsigned int, void *))dlsym(g_libcuda, "cuCtxCreate");
    p_cuCtxSetCurrent = (int (*)(void *))dlsym(g_libcuda, "cuCtxSetCurrent");
    p_cuMemHostAlloc = (int (*)(void **, size_t, unsigned int))dlsym(g_libcuda, "cuMemHostAlloc");
    p_cuMemFreeHost = (int (*)(void *))dlsym(g_libcuda, "cuMemFreeHost");
    if (p_cuInit == NULL || p_cuDeviceGet == NULL || p_cuCtxCreate == NULL ||
        p_cuCtxSetCurrent == NULL || p_cuMemHostAlloc == NULL || p_cuMemFreeHost == NULL) {
        fprintf(stderr, "libcuda is missing a driver entry point\n");
        return -1;
    }
    int status = p_cuInit(0);
    if (status != 0) {
        fprintf(stderr, "cuInit failed with %d\n", status);
        return -1;
    }
    void *device = NULL;
    status = p_cuDeviceGet(&device, 0);
    if (status != 0) {
        fprintf(stderr, "cuDeviceGet(0) failed with %d\n", status);
        return -1;
    }
    status = p_cuCtxCreate(&g_cu_ctx, 0, device);
    if (status != 0) {
        fprintf(stderr, "cuCtxCreate failed with %d\n", status);
        return -1;
    }
    return 0;
}

typedef struct {
    long long read_ns;
    long long alloc_ns;
    long long physical;
    long long wall_ns;
    int errors;
    uint64_t digest;
    double user_s;
    double sys_s;
} Stat;

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static int file_index(const char *path) {
    for (int i = 0; i < g_nfiles; i++) {
        if (strcmp(g_paths[i], path) == 0) return i;
    }
    if (g_nfiles >= MAXFILES) return -1;
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (g_direct) flags |= O_DIRECT;
#endif
    int fd = open(path, flags);
    if (fd < 0) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return -1;
    }
    g_paths[g_nfiles] = strdup(path);
    g_fds[g_nfiles] = fd;
    return g_nfiles++;
}

/* Destination buffer, plus its release hook, for one read. */
static void *take_buffer(long long len, int *slot) {
    *slot = -1;
    if (g_mode == 0) return NULL; /* filled in from the thread-local cache */
    if (g_mode == 1) {
        void *pointer = NULL;
        if (posix_memalign(&pointer, ALIGN, (size_t)len) != 0) return NULL;
        return pointer;
    }
    void *pointer = mmap(NULL, (size_t)len, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return pointer == MAP_FAILED ? NULL : pointer;
}

static void drop_buffer(void *pointer, long long len) {
    if (g_mode == 1) free(pointer);
    else if (g_mode == 2) munmap(pointer, (size_t)len);
}

static void *worker(void *argument) {
    const int index = (int)(intptr_t)argument;
    Stat stat;
    memset(&stat, 0, sizeof(stat));
    stat.digest = UINT64_C(0xcbf29ce484222325);

    void *cache_ptr[8];
    long long cache_len[8];
    int cache_n = 0;

    /* cuMemHostAlloc binds to the calling thread's current context, so every
     * worker adopts the one main created before the barrier. */
    if (g_mode == 3 && p_cuCtxSetCurrent(g_cu_ctx) != 0) {
        fprintf(stderr, "thread %d: cuCtxSetCurrent failed\n", index);
        return NULL;
    }

    pthread_barrier_wait(&g_barrier);
    double wall_start = now_ns();
    long long layer_read_ns[128];
    for (int i = 0; i < 128; i++) layer_read_ns[i] = 0;

    for (long i = 0; i < g_nreads; i++) {
        if (g_reads[i].unit % g_threads != index) continue;
        Read *read = &g_reads[i];
        void *destination = NULL;
        int cache_slot = -1;
        double alloc_start = now_ns();

        if (g_mode == 0 || g_mode == 3) {
            for (int k = 0; k < cache_n; k++) {
                if (cache_len[k] == read->len) {
                    destination = cache_ptr[k];
                    cache_slot = k;
                    break;
                }
            }
            if (destination == NULL) {
                if (g_mode == 3) {
                    if (p_cuMemHostAlloc(&destination, (size_t)read->len, 0) != 0) {
                        stat.errors++;
                        continue;
                    }
                } else if (posix_memalign(&destination, ALIGN, (size_t)read->len) != 0) {
                    stat.errors++;
                    continue;
                }
                if (cache_n < 8) {
                    cache_ptr[cache_n] = destination;
                    cache_len[cache_n] = read->len;
                    cache_slot = cache_n;
                    cache_n++;
                }
            }
        } else {
            destination = take_buffer(read->len, &cache_slot);
            if (destination == NULL) {
                stat.errors++;
                continue;
            }
        }
        double alloc_done = now_ns();
        stat.alloc_ns += (long long)(alloc_done - alloc_start);

        ssize_t got = -1;
        if (!g_direct) {
            got = pread(g_fds[read->file], destination, (size_t)read->len, read->off);
            stat.physical += got > 0 ? got : 0;
        } else {
            long long aligned_start = read->off / ALIGN * ALIGN;
            long long prefix = read->off - aligned_start;
            long long needed = prefix + read->len;
            long long physical = (needed + ALIGN - 1) / ALIGN * ALIGN;
            void *staging = NULL;
            if (posix_memalign(&staging, ALIGN, (size_t)physical) != 0) {
                stat.errors++;
                drop_buffer(destination, read->len);
                continue;
            }
            ssize_t actual = pread(g_fds[read->file], staging, (size_t)physical, aligned_start);
            if (actual >= needed) {
                memcpy(destination, (char *)staging + prefix, (size_t)read->len);
                got = read->len;
                stat.physical += actual;
            }
            free(staging);
        }
        double read_done = now_ns();
        stat.read_ns += (long long)(read_done - alloc_done);
        layer_read_ns[read->layer < 128 ? read->layer : 0] += (long long)(read_done - alloc_done);

        if (got == read->len) {
            const unsigned char *bytes = destination;
            for (long long k = 0; k < read->len; k += g_digest_stride) {
                stat.digest ^= (uint64_t)bytes[k] + 0x9e3779b97f4a7c15ULL * (uint64_t)k;
            }
        } else {
            stat.errors++;
        }

        if (g_mode == 1 || g_mode == 2) drop_buffer(destination, read->len);
        else if (cache_slot < 0) {
            if (g_mode == 3) p_cuMemFreeHost(destination);
            else free(destination);
        }
    }

    /* The pinned cache is the one state the engine would keep across layers,
     * so hold it until the thread's last read is done, then release. */
    if (g_mode == 3) {
        for (int k = 0; k < cache_n; k++) p_cuMemFreeHost(cache_ptr[k]);
    }

    stat.wall_ns = (long long)(now_ns() - wall_start);
    struct rusage usage;
    if (getrusage(RUSAGE_THREAD, &usage) == 0) {
        stat.user_s = (double)usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
        stat.sys_s = (double)usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
    }
    Stat *out = malloc(sizeof(Stat) + 128 * sizeof(long long));
    memcpy(out, &stat, sizeof(Stat));
    memcpy((char *)out + sizeof(Stat), layer_read_ns, 128 * sizeof(long long));
    return out;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <plan.tsv> reuse|pinned|fresh|fresh_mmap <direct 0|1> <threads>\n", argv[0]);
        return 1;
    }
    g_mode = strcmp(argv[2], "reuse") == 0 ? 0
               : (strcmp(argv[2], "pinned") == 0 ? 3
               : (strcmp(argv[2], "fresh") == 0 ? 1 : 2));
    g_direct = atoi(argv[3]) != 0;
    g_threads = atoi(argv[4]);
    const char *stride = getenv("DIGEST_STRIDE");
    if (stride != NULL) g_digest_stride = atoll(stride) > 0 ? atoll(stride) : 1LL << 40;
    if (g_threads < 1 || g_threads > 64) {
        fprintf(stderr, "threads out of range\n");
        return 1;
    }
    if (g_mode == 3 && init_cuda() != 0) return 1;

    FILE *plan = fopen(argv[1], "r");
    if (plan == NULL) {
        perror("plan");
        return 1;
    }
    long capacity = 4096;
    g_reads = malloc((size_t)capacity * sizeof(Read));
    char line[4096];
    while (fgets(line, sizeof(line), plan) != NULL) {
        int layer, unit;
        char path[2048];
        long long off, len;
        if (sscanf(line, "%d\t%d\t%2047s\t%lld\t%lld", &layer, &unit, path, &off, &len) != 5) continue;
        if (g_nreads == capacity) {
            capacity *= 2;
            g_reads = realloc(g_reads, (size_t)capacity * sizeof(Read));
        }
        int index = file_index(path);
        if (index < 0) return 1;
        g_reads[g_nreads++] = (Read){layer, unit, index, off, len};
        if (layer + 1 > g_layers) g_layers = layer + 1;
    }
    fclose(plan);

    Stat *stats[64];
    pthread_t threads[64];
    pthread_barrier_init(&g_barrier, NULL, (unsigned)g_threads);
    for (long t = 0; t < g_threads; t++) {
        if (pthread_create(&threads[t], NULL, worker, (void *)(intptr_t)t) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return 1;
        }
    }
    for (int t = 0; t < g_threads; t++) {
        void *result = NULL;
        pthread_join(threads[t], &result);
        if (result == NULL) {
            fprintf(stderr, "worker %d failed before the barrier\n", t);
            return 1;
        }
        stats[t] = result;
    }

    long long read_ns = 0, alloc_ns = 0, physical = 0;
    long long wall_ns = 0;
    double user_s = 0, sys_s = 0;
    int errors = 0;
    uint64_t digest = 0;
    long long layer_sum[128];
    for (int i = 0; i < 128; i++) layer_sum[i] = 0;
    for (int t = 0; t < g_threads; t++) {
        read_ns += stats[t]->read_ns;
        alloc_ns += stats[t]->alloc_ns;
        physical += stats[t]->physical;
        user_s += stats[t]->user_s;
        sys_s += stats[t]->sys_s;
        errors += stats[t]->errors;
        if (stats[t]->wall_ns > wall_ns) wall_ns = stats[t]->wall_ns;
        digest ^= stats[t]->digest;
        long long *layers = (long long *)((char *)stats[t] + sizeof(Stat));
        for (int i = 0; i < 128; i++) layer_sum[i] += layers[i];
    }
    long long logical = 0;
    for (long i = 0; i < g_nreads; i++) logical += g_reads[i].len;
    double wall_s = wall_ns / 1e9;
    double read_s = read_ns / 1e9;
    long long layer_max = 0;
    double layer_total = 0;
    for (int i = 0; i < (int)g_layers; i++) {
        if (layer_sum[i] > layer_max) layer_max = layer_sum[i];
        layer_total += (double)layer_sum[i];
    }
    printf("{\"mode\":\"%s\",\"direct\":%d,\"threads\":%d,\"reads\":%ld,\"layers\":%ld,"
           "\"logical_bytes\":%lld,\"physical_bytes\":%lld,\"wall_s\":%.6f,\"read_s\":%.6f,"
           "\"alloc_s\":%.6f,\"wall_gbps\":%.4f,\"read_gbps\":%.4f,"
           "\"per_layer_read_ms_avg\":%.4f,\"per_layer_read_ms_max\":%.4f,"
           "\"user_s\":%.4f,\"sys_s\":%.4f,\"cores_busy\":%.3f,\"cpu_s_per_gb\":%.4f,"
           "\"digest\":\"%016llx\",\"errors\":%d}\n",
           argv[2], g_direct ? 1 : 0, g_threads, g_nreads, g_layers, logical, physical,
           wall_s, read_s, alloc_ns / 1e9,
           logical / 1e9 / (wall_s > 0 ? wall_s : 1e-9),
           logical / 1e9 / (read_s > 0 ? read_s : 1e-9),
           layer_total / (g_layers > 0 ? (double)g_layers : 1.0) / 1e6,
           layer_max / 1e6,
           user_s, sys_s, (user_s + sys_s) / (wall_s > 0 ? wall_s : 1e-9),
           (user_s + sys_s) / (logical / 1e9),
           (unsigned long long)digest, errors);
    for (int i = 0; i < g_threads; i++) free(stats[i]);
    free(g_reads);
    return errors == 0 ? 0 : 1;
}
