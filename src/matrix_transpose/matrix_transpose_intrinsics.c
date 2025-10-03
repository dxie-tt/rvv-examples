#include <riscv_vector.h>
#include <stddef.h>
#include <bench_matrix_utils.h>


/** generic benchmark wrapper for 4x4 matrix transpose implementations */
unsigned long matrix_4x4_transpose_bench(float* dst, float* src, matrix_transpose_4x4_func_t func) {
    unsigned long start, stop;
    start = read_perf_counter();
    for (int iter=0; iter<4; iter++) {
        func(dst, src);
        func(src, dst);
    }
    stop = read_perf_counter();
    return (stop - start)/8;
}

/** generic benchmark wrapper for 4x4 matrix transpose implementations */
unsigned long matrix_nxn_transpose_bench(float* dst, float* src, size_t n, matrix_transpose_nxn_func_t func) {
    unsigned long start, stop;
    start = read_perf_counter();
    for (int iter=0; iter<4; iter++) {
        func(dst, src, n);
        func(src, dst, n);
    }
    stop = read_perf_counter();
    return (stop - start)/8;
}

/** transpose of a n x n matrix
 *
 *  Baseline implementation (not using RVV explicitly through intrinsics)
 *
 * @param dst address of destination matrix
 * @param src address of source matrix
 * @param n matrix dimensions
 */
void matrix_transpose(float *dst,
                      float *src,
                      size_t n) 
{
    size_t i, j;
    for (i = 0; i < n; ++i)
        for (j = 0; j < n; ++j) dst[i * n + j] = src[j * n + i];
};

/** transpose of a 4 x 4 matrix
 *
 *  Baseline implementation (not using RVV explicitly through intrinsics)
 *
 * @param dst address of destination matrix
 * @param src address of source matrix
 */
void matrix_transpose_4x4(float *dst, float *src) 
{
    size_t i, j;
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j) dst[i * 4 + j] = src[j * 4 + i];
};

void matrix_transpose_rev_4x4(float *dst, float *src) 
{
    size_t i, j;
    for (j = 0; j < 4; ++j)
        for (i = 0; i < 4; ++i) dst[i * 4 + j] = src[j * 4 + i];
};

/** 4x4 matrix transpose using strided stores */
void matrix_transpose_intrinsics_4x4(float *dst,
                                     float *src) 
{
    unsigned i;
    for (i = 0; i < 4; ++i) {
        vfloat32m1_t row = __riscv_vle32_v_f32m1(src + 4 * i, 4);
        __riscv_vsse32(dst + i, sizeof(float) * 4,row, 4);
    }
};

/** 4x4 matrix transpose using strided loads */
void matrix_transpose_intrinsics_loads_4x4(float *dst,
                                           float *src) 
{
    unsigned i;
    for (i = 0; i < 4; ++i) {
        vfloat32m1_t col = __riscv_vlse32_v_f32m1(src + i, sizeof(float) * 4, 4);
        __riscv_vse32(dst + i*4, col, 4);
    }
};


/** n x n matrix transpose using strided stores */
void matrix_transpose_intrinsics(float *dst,
                                 float *src,
                                 size_t n) 
{
    for (size_t row_id = 0; row_id < n; ++row_id) { // input row-index
        size_t avl = n;
        float* row_src = src + row_id * n;
        float* row_dst = dst + row_id;
        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t row = __riscv_vle32_v_f32m1(row_src, vl);
            __riscv_vsse32(row_dst, sizeof(float) * n, row, vl);
            avl -= vl;
            row_src += vl;
            row_dst += vl * n;
        }
    }
};


/** n x n matrix transpose using strided loads */
void matrix_transpose_intrinsics_loads(float *dst,
                                       float *src,
                                       size_t n) 
{
    for (size_t row_id = 0; row_id < n; ++row_id) { // input row-index
        size_t avl = n;
        float* col_src = src + row_id;
        float* row_dst = dst + row_id * n;
        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t row = __riscv_vlse32_v_f32m1(col_src, sizeof(float) * n, vl);
            __riscv_vse32(row_dst, row, vl);
            avl -= vl;
            col_src += vl * n;
            row_dst += vl;
        }
    }
};

/** n x n matrix transpose using strided loads */
void matrix_transpose_segmented_load_intrinsics(float *dst,
                                                float *src,
                                                size_t n) 
{
    // Fall back to generic if size is not a multiple of 8
    if (n & 7) {
	   matrix_transpose_intrinsics_loads(dst, src, n);
    } else {
       for (size_t row_id = 0; row_id < n; row_id+=8) { // input row-index
           size_t avl = n;
           float* col_src = src + row_id;
           float* row_dst = dst + row_id * n;
           while (avl > 0) {
               size_t vl = __riscv_vsetvl_e32m1(avl);
               vfloat32m1x8_t row = __riscv_vlsseg8e32_v_f32m1x8(col_src, sizeof(float) * n, vl);
               __riscv_vse32_v_f32m1(row_dst,     __riscv_vget_v_f32m1x8_f32m1(row, 0), vl);
               __riscv_vse32_v_f32m1(row_dst+1*n, __riscv_vget_v_f32m1x8_f32m1(row, 1), vl);
               __riscv_vse32_v_f32m1(row_dst+2*n, __riscv_vget_v_f32m1x8_f32m1(row, 2), vl);
               __riscv_vse32_v_f32m1(row_dst+3*n, __riscv_vget_v_f32m1x8_f32m1(row, 3), vl);
               __riscv_vse32_v_f32m1(row_dst+4*n, __riscv_vget_v_f32m1x8_f32m1(row, 4), vl);
               __riscv_vse32_v_f32m1(row_dst+5*n, __riscv_vget_v_f32m1x8_f32m1(row, 5), vl);
               __riscv_vse32_v_f32m1(row_dst+6*n, __riscv_vget_v_f32m1x8_f32m1(row, 6), vl);
               __riscv_vse32_v_f32m1(row_dst+7*n, __riscv_vget_v_f32m1x8_f32m1(row, 7), vl);
               avl -= vl;
               col_src += vl * n;
               row_dst += vl;
           }
       }
   }
};

/** Intrinsics based implementation of 4x4 32-bit matrix transpose */
void matrix_transpose_in_register(uint32_t* outputMat, uint32_t* inputMat) {
    vuint32m1_t matRegIn0 = __riscv_vle32_v_u32m1(inputMat, 4);
    vuint32m1_t matRegIn1 = __riscv_vle32_v_u32m1(inputMat + 4, 4);
    vuint32m1_t matRegIn2 = __riscv_vle32_v_u32m1(inputMat + 8, 4);
    vuint32m1_t matRegIn3 = __riscv_vle32_v_u32m1(inputMat + 12, 4);

    vbool32_t oddMask = __riscv_vreinterpret_v_u32m1_b32(__riscv_vmv_v_x_u32m1(0xaaaa, 1));
    // vl=4 in the following
    // should be mapped to vslideup.vi
    vuint32m1_t smallTransposeMat0 = __riscv_vslideup_vx_u32m1_tumu(oddMask, matRegIn0, matRegIn1, 1, 4);
    vuint32m1_t smallTransposeMat2 = __riscv_vslideup_vx_u32m1_tumu(oddMask, matRegIn2, matRegIn3, 1, 4);

    vbool32_t evenMask = __riscv_vreinterpret_v_u32m1_b32(__riscv_vmv_v_x_u32m1(0x5555, 1));
    // should me mapped to vslidedown.vi
    vuint32m1_t smallTransposeMat1 = __riscv_vslidedown_vx_u32m1_tumu(evenMask, matRegIn1, matRegIn0, 1, 4);
    vuint32m1_t smallTransposeMat3 = __riscv_vslidedown_vx_u32m1_tumu(evenMask, matRegIn3, matRegIn2, 1, 4);

    // should be mapped to vslideup.vi
    vuint32m1_t outMat0 = __riscv_vslideup_vx_u32m1_tu(smallTransposeMat0, smallTransposeMat2, 2, 4);
    vuint32m1_t outMat1 = __riscv_vslideup_vx_u32m1_tu(smallTransposeMat1, smallTransposeMat3, 2, 4);

    // vl=2 in the following
    // should me mapped to vslidedown.vi
    vuint32m1_t outMat2 = __riscv_vslidedown_vx_u32m1_tu(smallTransposeMat2, smallTransposeMat0, 2, 2);
    vuint32m1_t outMat3 = __riscv_vslidedown_vx_u32m1_tu(smallTransposeMat3, smallTransposeMat1, 2, 2);
    __riscv_vse32_v_u32m1(outputMat, outMat0, 4);
    __riscv_vse32_v_u32m1(outputMat + 4, outMat1, 4);
    __riscv_vse32_v_u32m1(outputMat + 8, outMat2, 4);
    __riscv_vse32_v_u32m1(outputMat + 12, outMat3, 4);
}


void matrix_4x4_transpose_segmented_load_intrinsics(float* dst, float* src) {
    vfloat32m1x4_t data = __riscv_vlseg4e32_v_f32m1x4(src, 4);
    vfloat32m1_t data0 = __riscv_vget_v_f32m1x4_f32m1(data, 0);
    vfloat32m1_t data1 = __riscv_vget_v_f32m1x4_f32m1(data, 1);
    vfloat32m1_t data2 = __riscv_vget_v_f32m1x4_f32m1(data, 2);
    vfloat32m1_t data3 = __riscv_vget_v_f32m1x4_f32m1(data, 3);
    __riscv_vse32_v_f32m1(dst,    data0, 4);
    __riscv_vse32_v_f32m1(dst+4,  data1, 4);
    __riscv_vse32_v_f32m1(dst+8,  data2, 4);
    __riscv_vse32_v_f32m1(dst+12, data3, 4);
}


void matrix_4x4_transpose_segmented_store_intrinsics(float* dst, float* src) {
    vfloat32m1_t data0 = __riscv_vle32_v_f32m1(src,    4);
    vfloat32m1_t data1 = __riscv_vle32_v_f32m1(src+4,  4);
    vfloat32m1_t data2 = __riscv_vle32_v_f32m1(src+8,  4);
    vfloat32m1_t data3 = __riscv_vle32_v_f32m1(src+12, 4);
    vfloat32m1x4_t packedData = __riscv_vcreate_v_f32m1x4(data0, data1, data2, data3);
    __riscv_vsseg4e32_v_f32m1x4(dst, packedData, 4);
}


/** extracting the array from the function matrix_4x4_transpose_vrgather
 *  seems to allow more optimization
*/
const uint16_t offsetIndex[] = {
    0, 4, 8, 12,
    1, 5, 9, 13,
    2, 6, 10, 14,
    3, 7, 11, 15
};

vfloat32m4_t matrix_4x4_transpose_vrgather(vfloat32m4_t src) {
    // loading array of indices
    vuint16m2_t indices = __riscv_vle16_v_u16m2(offsetIndex, 16);
    // performing transpose with in-register permutation
    vfloat32m4_t result = __riscv_vrgatherei16_vv_f32m4(src, indices, 16);
    return result;
}


/** To measure only the register-to-register count of instruction
 *  we wrap the bench to extract matrix load and store
*/
unsigned long matrix_4x4_transpose_vrgather_bench (float* dst, float* src) {
    unsigned long start, stop;
    vfloat32m4_t data = __riscv_vle32_v_f32m4(src, 16);
    start = read_perf_counter();
    vfloat32m4_t result;
    for (int iter=0; iter<4; iter++) {
        result = matrix_4x4_transpose_vrgather(data);
        data = matrix_4x4_transpose_vrgather(result);
    }
    stop = read_perf_counter();
    __riscv_vse32_v_f32m4(dst, result, 16);

    return (stop - start)/8;
}

vfloat32m4_t matrix_4x4_transpose_vslide(vfloat32m4_t src) {
    vfloat32m1_t inMat0 = __riscv_vget_v_f32m4_f32m1(src, 0);
    vfloat32m1_t inMat1 = __riscv_vget_v_f32m4_f32m1(src, 1);
    vfloat32m1_t inMat2 = __riscv_vget_v_f32m4_f32m1(src, 2);
    vfloat32m1_t inMat3 = __riscv_vget_v_f32m4_f32m1(src, 3);

    vbool32_t oddMask = __riscv_vreinterpret_v_u32m1_b32(__riscv_vmv_v_x_u32m1(0xaaaa, 1));
    // vl=4 in the following
    // should be mapped to vslideup.vi
    vfloat32m1_t transMat0 = __riscv_vslideup_vx_f32m1_tumu(oddMask, inMat0, inMat1, 1, 4);
    vfloat32m1_t transMat2 = __riscv_vslideup_vx_f32m1_tumu(oddMask, inMat2, inMat3, 1, 4);

    vbool32_t evenMask = __riscv_vreinterpret_v_u32m1_b32(__riscv_vmv_v_x_u32m1(0x5555, 1));
    // should be mapped to vslidedown.vi
    vfloat32m1_t transMat1 = __riscv_vslidedown_vx_f32m1_tumu(evenMask, inMat1, inMat0, 1, 4);
    vfloat32m1_t transMat3 = __riscv_vslidedown_vx_f32m1_tumu(evenMask, inMat3, inMat2, 1, 4);

    // should be mapped to vslideup.vi
    vfloat32m1_t outMat0 = __riscv_vslideup_vx_f32m1_tu(transMat0, transMat2, 2, 4);
    vfloat32m1_t outMat1 = __riscv_vslideup_vx_f32m1_tu(transMat1, transMat3, 2, 4);

    // vl=2 in the following
    // should be mapped to vslidedown.vi
    vfloat32m1_t outMat2 = __riscv_vslidedown_vx_f32m1_tu(transMat2, transMat0, 2, 2);
    vfloat32m1_t outMat3 = __riscv_vslidedown_vx_f32m1_tu(transMat3, transMat1, 2, 2);

    return __riscv_vcreate_v_f32m1_f32m4(outMat0, outMat1, outMat2, outMat3);
}


/** Benchmark wrapper for nxn baseline matrix transpose */
unsigned long matrix_transpose_nxn_bench(float* dst, float* src, size_t n)
{
    return matrix_nxn_transpose_bench(dst, src, n, matrix_transpose);
}


/** Benchmark wrapper for 4x4 baseline matrix transpose */
unsigned long matrix_transpose_4x4_bench(float* dst, float* src)
{
    return matrix_4x4_transpose_bench(dst, src, matrix_transpose_4x4);
}

/** Benchmark wrapper for 4x4 baseline matrix transpose */
unsigned long matrix_transpose_rev_4x4_bench(float* dst, float* src)
{
    return matrix_4x4_transpose_bench(dst, src, matrix_transpose_rev_4x4);
}

/** Benchmark wrapper for 4x4 intrinsic based matrix transpose */
unsigned long matrix_transpose_intrinsics_4x4_bench(float* dst, float* src)
{
    return matrix_4x4_transpose_bench(dst, src, matrix_transpose_intrinsics_4x4);
}

/** Benchmark wrapper for 4x4 intrinsic loads based matrix transpose */
unsigned long matrix_transpose_intrinsics_loads_4x4_bench(float* dst, float* src)
{
    return matrix_4x4_transpose_bench(dst, src, matrix_transpose_intrinsics_loads_4x4);
}

/** Benchmark wrapper for nxn intrinsic based matrix transpose */
unsigned long matrix_transpose_intrinsics_nxn_bench(float* dst, float* src, size_t n)
{
    return matrix_nxn_transpose_bench(dst, src, n, matrix_transpose_intrinsics);
}

/** Benchmark wrapper for nxn intrinsic based matrix transpose */
unsigned long matrix_transpose_intrinsics_loads_nxn_bench(float* dst, float* src, size_t n)
{
    return matrix_nxn_transpose_bench(dst, src, n, matrix_transpose_intrinsics_loads);
}

/** Benchmark wrapper for nxn intrinsic based matrix transpose */
unsigned long matrix_transpose_segmented_load_intrinsics_nxn_bench(float* dst, float* src, size_t n)
{
    return matrix_nxn_transpose_bench(dst, src, n, matrix_transpose_segmented_load_intrinsics);
}

/** wrapping segmented load based implementation */
unsigned long matrix_4x4_transpose_segmented_load_intrinsics_bench(float* dst, float* src) {
    return matrix_4x4_transpose_bench(dst, src, matrix_4x4_transpose_segmented_load_intrinsics);
}

/** wrapping segmented store based implementation */
unsigned long matrix_4x4_transpose_segmented_store_intrinsics_bench(float* dst, float* src) {
    return matrix_4x4_transpose_bench(dst, src, matrix_4x4_transpose_segmented_store_intrinsics);
}

/** Intrinsics based implementation of 4x4 32-bit matrix transpose */
unsigned long matrix_4x4_transpose_vslide_bench(float* dst, float* src) {
    unsigned long start, stop;
    vfloat32m4_t data = __riscv_vle32_v_f32m4(src, 16);
    start = read_perf_counter();
    vfloat32m4_t result = matrix_4x4_transpose_vslide(data);
    stop = read_perf_counter();
    __riscv_vse32_v_f32m4(dst, result, 16);

    return stop - start;
}
