/*********************************************************************
 * median_filter_uint16.c
 *
 * A C implementation equivalent to scipy.ndimage.median_filter for
 * 2‑D uint16_t images.
 *
 * Author : gpt oss 20b
 * Date   : 2025-08-22
 *********************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Helper: binary search to find insertion point for a value          */
/* ------------------------------------------------------------------ */
static size_t
bisect_left(const uint16_t *arr, size_t n, uint16_t val)
{
    size_t lo = 0;
    size_t hi = n;

    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (arr[mid] < val)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;   /* position where val would be inserted */
}

/* ------------------------------------------------------------------ */
/* Helper: get a pixel value with mirror padding                      */
/* ------------------------------------------------------------------ */
static uint16_t
getp(const uint16_t *src, size_t width, size_t height,
     int r, int c)
{
    if (r < 0)       r = -r;
    else if ((size_t)r >= height) r = (int)(2 * height - r - 2);

    if (c < 0)       c = -c;
    else if ((size_t)c >= width)  c = (int)(2 * width  - c - 2);

    return src[r * width + c];
}

/* ------------------------------------------------------------------ */
/* Comparator used by qsort for uint16_t                              */
/* ------------------------------------------------------------------ */
static int
cmp_uint16(const void *a, const void *b)
{
    uint16_t va = *(const uint16_t *)a;
    uint16_t vb = *(const uint16_t *)b;

    if (va < vb) return -1;
    if (va > vb) return  1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main filter routine                                                */
/* ------------------------------------------------------------------ */
void
median_filter_uint16(const uint16_t *src,
                     uint16_t       *dst,
                     size_t          width,
                     size_t          height,
                     size_t          ksize)
{
    /* ksize must be odd – otherwise we just copy the image.        */
    if (ksize % 2 == 0) {
        memcpy(dst, src, width * height * sizeof(uint16_t));
        return;
    }

    const size_t half = ksize / 2;
    const size_t win_area = ksize * ksize;

    /* Allocate buffer for the sorted window */
    uint16_t *win = (uint16_t *)malloc(win_area * sizeof(uint16_t));
    if (!win) return;   /* allocation failed – silently exit */

    /* ------------------------------------------------------------------
     * Build the initial window for the first pixel of the first row
     * ------------------------------------------------------------------ */
    size_t idx = 0;
    for (int dr = -((int)half); dr <= ((int)half); ++dr) {
        for (int dc = -((int)half); dc <= ((int)half); ++dc) {
            win[idx++] = getp(src, width, height,
                               (int)(0 + dr), (int)(0 + dc));
        }
    }

    /* Sort the initial window */
    qsort(win, win_area, sizeof(uint16_t), cmp_uint16);

    /* ------------------------------------------------------------------
     * Process each pixel
     * ------------------------------------------------------------------ */
    for (size_t r = 0; r < height; ++r) {
        /* First column of this row – we already have the window.   */
        dst[r * width] = win[half * ksize + half];   /* median */

        /* Shift window to the right by one pixel for subsequent columns */
        for (size_t c = 1; c < width; ++c) {
            /* Remove the leftmost column from the sorted window.
               The leftmost column starts at index 'half' in the sorted
               array because we always keep the array sorted.          */

            size_t pos = bisect_left(win, win_area, win[0]);   /* first element */
            for (size_t i = pos; i < win_area - 1; ++i)
                win[i] = win[i + 1];

            /* Insert the new rightmost column values one by one      */
            uint16_t new_col[15];    /* ksize <= 15 is safe on stack */
            for (int dr = -((int)half); dr <= ((int)half); ++dr) {
                int rr = (int)(r + dr);
                int cc = (int)(c + half);
                new_col[dr + (int)half] = getp(src, width, height, rr, cc);
            }

            for (size_t i = 0; i < ksize; ++i) {
                uint16_t val = new_col[i];
                size_t ins_pos = bisect_left(win, win_area - 1, val);
                for (size_t j = win_area - 1; j > ins_pos; --j)
                    win[j] = win[j - 1];
                win[ins_pos] = val;
            }

            /* The median is now at the centre position */
            dst[r * width + c] = win[half * ksize + half];
        }

        /* ------------------------------------------------------------------
         * Move to next row: rebuild window from scratch (simpler than sliding)
         * ------------------------------------------------------------------ */
        if (r + 1 < height) {
            idx = 0;
            for (int dr = -((int)half); dr <= ((int)half); ++dr) {
                for (int dc = -((int)half); dc <= ((int)half); ++dc) {
                    win[idx++] = getp(src, width, height,
                                       (int)(r + 1 + dr), (int)(0 + dc));
                }
            }
            qsort(win, win_area, sizeof(uint16_t), cmp_uint16);
        }
    }

    free(win);
}

