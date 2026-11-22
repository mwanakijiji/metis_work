# How HDRL makes a master flat (`hdrl_flat_compute`)

A plain-English walk-through of `hdrl_flat.c` from HDRL 1.6.0
(`~/Documents/git.repos/eso_cpl_hdrl/hdrl-1.6.0/hdrl_flat.c`). HDRL is ESO's
High-level Data Reduction Library. PyHDRL calls this C function through its
`Flat.compute()` binding (`flat.cpp`).

## What goes in

| Input | What it is |
|---|---|
| `hdrl_data` | A list of flat-field exposures: several frames of a uniformly lit source. Each pixel carries a value, an error and a bad-pixel flag. **This list is overwritten** to save memory. Don't use it after the call; just delete it. |
| `stat_mask` | Optional static mask (may be `NULL`). In CPL a mask pixel set to *true* means "reject". So the pixels flagged here are the ones **left out** of the statistics and the smoothing (e.g. the unilluminated part of the detector). |
| `collapse_params` | How to combine the frames into one: any HDRL collapse method (mean, weighted mean, median, sigma-clipping, min-max rejection, mode). |
| `flat_params` | Made by `hdrl_flat_parameter_create(filter_size_x, filter_size_y, method)`. It holds the median-filter box size and the method (`HDRL_FLAT_FREQ_LOW` or `HDRL_FLAT_FREQ_HIGH`). |

Requirements on the parameters (checked by `hdrl_flat_parameter_verify`):

- `filter_size_x` and `filter_size_y` must be **positive odd numbers**. They give the
  size in pixels of the box used for median smoothing.
- `method` must be `LOW` or `HIGH`.
- The frames, the collapse parameter and the flat parameter must not be `NULL`.

## What comes out

- **`master`:** the master flat, with values and errors.
- **`contrib_map`:** for each pixel, how many input frames contributed to it after bad
  pixels and the collapse's rejection. It comes straight from the collapse step.

## The two methods

The median filter is a full rectangular box of `filter_size_x` × `filter_size_y`
pixels. Throughout, pixels flagged bad in a frame are ignored by the medians and the
smoothing.

### Low-frequency flat: the large-scale illumination pattern

For vignetting, gradients across the field, and the like.

1. **Normalise each frame to 1.** Take the median of the frame and divide the
   frame (and its error) by it.
   - With a `stat_mask`, the median uses only the pixels the mask doesn't flag,
     e.g. only the illuminated area.
   - The median is treated as exact: no error is propagated from it.
2. **Combine** the normalised frames pixel by pixel with the chosen collapse
   method. This also gives the contribution map.
3. **Smooth** the combined image with the median filter. That keeps only the broad
   shape and removes pixel-to-pixel structure. The error attached to the master is
   the combined error from step 2; the smoothing is treated as noise-free and doesn't
   change it.

### High-frequency flat: the pixel-to-pixel sensitivity

For differences between neighbouring pixels, dust specks, and the like.

1. **Remove the large-scale pattern from each frame.** Median-smooth the frame, then
   divide the frame (and its error) by its smoothed version. What remains are
   values around 1 that describe how each pixel differs from its neighbours. The
   smoothed image is treated as noise-free, so each pixel keeps the same relative
   error.
   - **With a `stat_mask`,** the smoothing is done **twice**: once over the pixels
     the mask doesn't flag (e.g. the illuminated region) and once over the flagged
     ones (e.g. the dark region). The two results are then joined into one smoothed
     image. This stops the smoothing box from mixing bright and dark regions at their
     boundary, which would otherwise create false structure along the edge. After
     each smoothing the mask is re-applied, so the boundary stays sharp.
   - Without a `stat_mask`, the whole frame is smoothed at once.
2. **Combine** the ratio frames pixel by pixel with the chosen collapse method. The
   result is the master flat; no further smoothing is done.

For both methods the errors are made positive (absolute value) after the division.
A comment in the code (`TODO`) questions whether that follows strict error
propagation.

## Summary

| | Low frequency | High frequency |
|---|---|---|
| Per-frame step | divide by the frame's median | divide by the frame's median-smoothed copy |
| `stat_mask` used for | which pixels the median uses | smoothing the two regions separately |
| After combining | median-smooth the result | nothing |
| Keeps | large-scale illumination | pixel-to-pixel sensitivity |

The two kinds of flat are complementary. A full flat-field correction often uses both,
multiplied together: low frequency × high frequency.

## Practical notes

- **Use several frames.** The collapse step is what rejects outliers (e.g. cosmic
  rays), so it only helps with more than one frame.
- **Choose the filter box against the structure you want to keep or remove.** It
  should be larger than the pixel-scale structure and smaller than the illumination
  gradients. Too small a box in the high-frequency method removes real pixel
  structure; too large a box in the low-frequency method washes out real gradients.
- **Flag everything that isn't uniformly lit in the `stat_mask`.** That includes
  unilluminated borders, reference pixels and masked areas, so they don't bias the
  normalisation or the smoothing.
- **Set bad-pixel masks on the input frames beforehand.** HDRL ignores flagged pixels
  everywhere, but it doesn't find them for you.
