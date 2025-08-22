#ifndef __MEDIAN_FILTER_UINT16_H__
#define __MEDIAN_FILTER_UINT16_H__

#include <stdint.h>

void median_filter_uint16(const uint16_t *src,
                     uint16_t       *dst,
                     size_t          width,
                     size_t          height,
                     size_t          ksize);

#endif /* __MEDIAN_FILTER_UINT16_H__ */
