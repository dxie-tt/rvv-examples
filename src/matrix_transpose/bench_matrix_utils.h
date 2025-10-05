#include <stddef.h>

typedef void(matrix_transpose_4x4_func_t)(float* dst, float* src);
typedef void(matrix_transpose_nxn_func_t)(float* dst, float* src, size_t n);

extern unsigned long counter_value_prev;
extern unsigned long counter_value_hi;

/** return the value of selected perf counter
 * 
 * perf counter is selected through a macro:
 * - defining COUNT_INSTRET selects the instret counter
 *    The instret counter counts the number of retired (executed) instructions.
 * - defining COUNT_CYCLE selects cycle count
*/
static unsigned long read_perf_counter(void)
{
  unsigned long counter_value;
#if defined(COUNT_INSTRET)
#define PERF_METRIC "instruction"
  asm volatile ("rdinstret %0" : "=r" (counter_value));
#elif defined(COUNT_CYCLE)
#define PERF_METRIC "cycle"
  asm volatile ("rdcycle %0" : "=r" (counter_value));
#else
  // instret is also the default
#define PERF_METRIC "instruction"
  asm volatile ("rdinstret %0" : "=r" (counter_value));
#endif

  // Work around issue when some design only report 32-bit counter value
  if (counter_value < counter_value_prev) {
     counter_value_hi += 1;
  }
  counter_value_prev = counter_value;
  unsigned long counter_value_adj = counter_value + (counter_value_hi << 32); 
  return counter_value_adj;
}
