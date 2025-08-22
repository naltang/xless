#include <stdio.h>
#include "correction_param.h"
#include "median_filter_uint16.h"

void correction_raw_read(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Read and process the raw data
    // ...

    fclose(file);
}


#include <stdio.h>
#include <stdint.h>

/* paste the function above here */

int main(void)
{
    /* Example image 5×4 (row major) */
    uint16_t src[20] = {
         10, 12, 13, 15, 17,
          9, 11, 14, 16, 18,
         20, 22, 23, 25, 27,
         19, 21, 24, 26, 28
    };
    uint16_t dst[20];

    /* Apply a 3×3 median filter */
    median_filter_uint16(src, dst, 5, 4, 3);

    /* Print result */
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 5; ++c)
            printf("%4d ", dst[r * 5 + c]);
        puts("");
    }
}
