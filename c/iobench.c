/* Microbench: banda in lettura RANDOM con blocchi tipo-expert (~19 MB int4).
 * Misura cio' che fa il motore davvero: N thread che leggono in parallelo
 * (expert_load sotto omp parallel for), buffered oppure O_DIRECT.
 * uso: ./iobench <file_grande> [blocco_MB] [n_letture] [threads] [direct 0/1]
 * build: gcc -O2 -fopenmp iobench.c -o iobench */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include "compat.h"
#ifdef _OPENMP
#include <omp.h>
#endif
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint64_t mix64(uint64_t x){
    x^=x>>30; x*=UINT64_C(0xbf58476d1ce4e5b9);
    x^=x>>27; x*=UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}
static uint64_t buffer_digest(const void *buf,size_t n,uint64_t seed){
    const unsigned char *p=buf; uint64_t h=mix64(seed^(uint64_t)n);
    size_t i=0;
    for(;i+8<=n;i+=8){ uint64_t word; memcpy(&word,p+i,8); h=mix64(h^word^(uint64_t)i); }
    for(;i<n;i++) h=mix64(h^p[i]^(uint64_t)i);
    return h;
}
int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s file [blkMB] [n] [threads] [direct 0/1]\n",argv[0]);return 1;}
    long mb=argc>2?atol(argv[2]):19;
    int n=argc>3?atoi(argv[3]):64;
    int nth=argc>4?atoi(argv[4]):8;
    int requested_direct=argc>5?atoi(argv[5]):1, direct=requested_direct;
    if(mb<=0 || mb>LONG_MAX/(1024*1024) || n<=0 || nth<=0 ||
       (requested_direct!=0 && requested_direct!=1)){
        fprintf(stderr,"invalid block/read/thread/direct argument\n"); return 1;
    }
    long blk=mb*1024*1024;
#ifdef O_DIRECT
    int fd=open(argv[1],O_RDONLY|(direct?O_DIRECT:0));
    if(fd<0 && direct){ fprintf(stderr,"O_DIRECT is unavailable (%s); using buffered I/O\n",strerror(errno));
        direct=0; fd=open(argv[1],O_RDONLY); }
#elif defined(_WIN32)
    int fd = direct ? compat_open_direct(argv[1]) : open(argv[1],COMPAT_O_RDONLY);
    if(fd<0 && direct){ fprintf(stderr,"NO_BUFFERING is unavailable; using buffered I/O\n");
        direct=0; fd=open(argv[1],COMPAT_O_RDONLY); }
#else
    int fd=open(argv[1],O_RDONLY);                 /* macOS: F_NOCACHE ~ O_DIRECT */
#ifdef __APPLE__
    if(direct && fd>=0) fcntl(fd,F_NOCACHE,1);
#else
    if(direct){ fprintf(stderr,"O_DIRECT is unavailable; using buffered I/O\n"); direct=0; }
#endif
#endif
    if(fd<0){perror("open");return 1;}
#ifdef _WIN32
    off_t sz=compat_fsize(fd);     /* CRT lseek(SEEK_END) ritorna -1 sui fd NO_BUFFERING */
#else
    off_t sz=lseek(fd,0,SEEK_END);
#endif
    if(sz<blk*2){fprintf(stderr,"file is too small\n");return 1;}
    /* offset random pre-generati (stessi per ogni configurazione: srand fisso).
     * 30 bit di rand combinati: su Windows RAND_MAX=32767 e un singolo rand()*4096
     * copre solo i primi 134 MB del file (tutti in page cache = misura falsa). */
    off_t *offs=malloc(n*sizeof(off_t)); uint64_t *digests=calloc((size_t)n,sizeof(uint64_t));
    if(!offs || !digests){fprintf(stderr,"offset/digest allocation failed\n");close(fd);free(offs);free(digests);return 1;}
    srand(1234);
    for(int i=0;i<n;i++){
        off_t r30=((off_t)rand()<<15)|rand();
        off_t o=sz==blk?0:(r30*4096)%(sz-blk);
        offs[i]=o&~4095L;
    }
    double verified_t0=now(), pread_return_wall_started=verified_t0, last_pread_finished=0, max_thread_pread_elapsed=0;
    int64_t tot=0; int errors=0; /* long e' 32-bit su Windows (LLP64): >2GB andava in overflow */
    #pragma omp parallel num_threads(nth) reduction(+:tot,errors)
    {
        void *buf=NULL; int allocated=posix_memalign(&buf,4096,(size_t)blk)==0;
        double thread_read_elapsed=0;
        #pragma omp for schedule(dynamic,1)
        for(int i=0;i<n;i++){
            if(!allocated){ errors++; continue; }
            double read_t0=now();
            ssize_t r=pread(fd,buf,blk,offs[i]);
            double read_finished=now();
            thread_read_elapsed+=read_finished-read_t0;
            #pragma omp critical
            if(read_finished>last_pread_finished) last_pread_finished=read_finished;
            if(r!=blk){
                if(r<0) perror("pread"); else fprintf(stderr,"short pread: %zd of %ld\n",r,blk);
                errors++;
            }else{
                tot+=r;
                digests[i]=buffer_digest(buf,(size_t)r,(uint64_t)offs[i]);
            }
        }
        #pragma omp critical
        if(thread_read_elapsed>max_thread_pread_elapsed) max_thread_pread_elapsed=thread_read_elapsed;
        if(allocated) compat_aligned_free(buf); /* Windows posix_memalign=_aligned_malloc */
    }
    /* Earlier digests can delay later reads; final aggregation is deliberately outside this wall. */
    double pread_return_wall_elapsed=last_pread_finished-pread_return_wall_started;
    if(pread_return_wall_elapsed<=0){fprintf(stderr,"iobench recorded no successful read interval\n");close(fd);free(offs);free(digests);return 1;}
    if(errors){fprintf(stderr,"iobench failed: %d read/allocation errors\n",errors);close(fd);free(offs);free(digests);return 1;}
    uint64_t digest=UINT64_C(0x6a09e667f3bcc909);
    for(int i=0;i<n;i++) digest=mix64(digest^digests[i]^mix64((uint64_t)offs[i])^(uint64_t)i);
    double verified_elapsed=now()-verified_t0;
    double pread_return_wall_gbps=tot/1e9/pread_return_wall_elapsed;
    double verified_pipeline_gbps=tot/1e9/verified_elapsed;
    printf("%s x%d threads: %d reads x %ldMB = %.1f GB, pread-return wall %.2fs -> %.2f GB/s; verified pipeline %.2fs -> %.2f GB/s\n",
        direct?"O_DIRECT":"buffered", nth, n, blk/1024/1024, tot/1e9,
        pread_return_wall_elapsed,pread_return_wall_gbps,verified_elapsed,verified_pipeline_gbps);
    printf("IOBENCH_JSON {\"requested_mode\":\"%s\",\"actual_mode\":\"%s\",\"threads\":%d,\"reads\":%d,"
           "\"block_mib\":%ld,\"bytes\":%lld,\"pread_return_wall_elapsed_s\":%.9f,\"pread_return_wall_definition\":\"before_parallel_read_loop_to_last_pread_return\",\"intermediate_digest_in_pread_return_wall\":true,\"final_aggregate_digest_in_pread_return_wall\":false,\"pread_return_wall_gbps\":%.9f,\"pread_return_wall_gbps_definition\":\"bytes / pread_return_wall_elapsed_s\",\"read_elapsed_s\":%.9f,\"read_elapsed_definition\":\"deprecated_alias_of_pread_return_wall_elapsed_s\",\"read_timing\":\"pread_return_wall\",\"max_thread_pread_elapsed_s\":%.9f,\"verified_elapsed_s\":%.9f,\"verified_elapsed_definition\":\"before_parallel_read_loop_to_final_aggregate_digest\",\"verified_pipeline_gbps\":%.9f,\"verified_pipeline_gbps_definition\":\"bytes / verified_elapsed_s\",\"read_gbps\":%.9f,\"read_gbps_definition\":\"deprecated_alias_bytes / pread_return_wall_elapsed_s\","
           "\"digest\":\"%016llx\",\"digest_algorithm\":\"mix64-full-buffer-v1\","
           "\"digest_in_read_timing\":true}\n",
        requested_direct?"direct_uncached":"cached",
        direct?"direct_uncached":"cached",nth,n,mb,(long long)tot,
        pread_return_wall_elapsed,pread_return_wall_gbps,pread_return_wall_elapsed,
        max_thread_pread_elapsed,verified_elapsed,verified_pipeline_gbps,pread_return_wall_gbps,
        (unsigned long long)digest);
    close(fd); free(offs); free(digests); return 0;
}
