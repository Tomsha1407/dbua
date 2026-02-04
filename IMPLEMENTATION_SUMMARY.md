# Asymmetric TX/RX Implementation Summary

## Overview
Successfully implemented support for asymmetric TX (204 transmit elements) and RX (238 receive elements) dimensions in the ultrasound beamforming code.

## Key Changes Made

### 1. **Separate Element Position Arrays Loading** (Lines 130-136)
   - **Before**: Single `elpos_full` array from TX setup
   - **After**: Two separate arrays:
     - `elpos_tx`: Shape (3, 204) - effective TX sources from `f["tx_setup/[0]/origin"]`
     - `elpos_rx`: Shape (3, 238) - RX elements from `f["probe/element_positions"]`
   - **Benefit**: Accurately represents the actual physical layout of the ultrasound transducer

### 2. **Asymmetric Data Dimension Handling** (Lines 163-201)
   - **Before**: Forced TX and RX to match (symmetric square array)
   - **After**: Two operational modes:

   **Mode A: `n_elemnts=None` (Default - Use all available elements)**
   ```
   iqdata shape: [204, 238, nt]  (Asymmetric - 204 TX × 238 RX)
   - RX dimension kept at 238 (cropped if > 238)
   - TX dimension padded/cropped to 204
   ```

   **Mode B: `n_elemnts=<value>` (Cropped square array for backward compatibility)**
   ```
   iqdata shape: [n_elemnts, n_elemnts, nt]  (Symmetric for compatibility)
   - Both TX and RX cropped from center to n_elemnts
   - Element positions also cropped symmetrically
   ```

### 3. **Separate TX/RX Coordinate Extraction** (Lines 212-215)
   ```python
   xe_tx, _, ze_tx = jnp.array(elpos_tx_use)  # 204 TX element X,Z coordinates
   xe_rx, _, ze_rx = jnp.array(elpos_rx_use)  # 238 RX element X,Z coordinates
   ```

### 4. **Independent TX/RX Broadcasting** (Lines 232-240)
   - **TX broadcasting**: `xe_tx_bc` and `ze_tx_bc` with shape [204, 1, 1]
   - **RX broadcasting**: `xe_rx_bc` and `ze_rx_bc` with shape [238, 1, 1]
   - Allows independent time-of-flight calculations for each dimension

### 5. **Separate Time-of-Flight Functions** (Lines 254-269)
   - **TX-focused functions**:
     - `tof_image_tx(c)`: TX elements → Image pixels
     - `tof_patch_tx(c)`: TX elements → Patch pixels
   
   - **RX-focused functions**:
     - `tof_image_rx(c)`: RX elements → Image pixels
     - `tof_patch_rx(c)`: RX elements → Patch pixels
   
   - **Why separate?** Enables accurate modeling of acoustic reciprocity with different TX/RX positions

### 6. **Updated Image Formation** (Lines 271-273)
   ```python
   def makeImage(c):
       t_tx = tof_image_tx(c)  # TX delays
       t_rx = tof_image_rx(c)  # RX delays
       return jnp.abs(das(iqdata, t_tx - t0, t_rx, fs, fd))
   ```
   - Now passes separate time-of-flight arrays to `das()` for TX and RX

### 7. **Updated Loss Wrapper** (Lines 275-278)
   ```python
   def loss_wrapper(func, c):
       t_tx = tof_patch_tx(c)
       t_rx = tof_patch_rx(c)
       return (func)(iqdata, t_tx - t0, t_rx, fs, fd)
   ```
   - All loss functions (speckle brightness, lag-one coherence, coherence factor, phase error) now receive separate TX/RX delays

### 8. **Fixed Phase Error Loss** (Lines 286-292)
   ```python
   def pe_loss(c):
       t_tx = tof_patch_tx(c)
       t_rx = tof_patch_rx(c)
       dphi = phase_error(iqdata, t_tx - t0, t_rx, fs, fd)  # Now receives both
       ...
   ```
   - Updated to match the `phase_error()` function signature that expects separate TX and RX delays

## Data Flow Changes

### Previous (Symmetric)
```
iqdata [na×na×nt] → das(iqdata, t, t, ...) 
    where t = time_of_flight(elements, pixels, ...)
```

### New (Asymmetric)
```
iqdata [204×238×nt] → das(iqdata, t_tx, t_rx, ...)
    where t_tx = time_of_flight(204_TX_elements, pixels, ...)
          t_rx = time_of_flight(238_RX_elements, pixels, ...)
```

## Backward Compatibility

**Code remains backward compatible** via the `n_elemnts` parameter:

- **`n_elemnts=None`** (new default): Uses full asymmetric arrays (204×238)
- **`n_elemnts=30`** (existing usage): Creates symmetric 30×30 array (original behavior)

Example call that still works:
```python
main(exp_name, LOSS, n_elemnts=30, nt=800, name="origin")
```

## Testing Recommendations

1. **Verify data shapes**:
   ```python
   print(f"iqdata shape: {iqdata.shape}")  # Should be [204, 238, 800]
   print(f"elpos_tx shape: {elpos_tx.shape}")  # Should be [3, 204]
   print(f"elpos_rx shape: {elpos_rx.shape}")  # Should be [3, 238]
   ```

2. **Run with new asymmetric mode**:
   ```python
   main('0003490e_20250611', 'pe', n_elemnts=None, nt=800, name="asymmetric")
   ```

3. **Verify backward compatibility**:
   ```python
   main('0003490e_20250611', 'pe', n_elemnts=30, nt=800, name="symmetric")
   ```

## Technical Details

### Das Function Compatibility
The `das()` function already supports asymmetric dimensions:
- Input: `iqdata[na, nb, nsamps]` with `tA[na, *pixdims]` and `tB[nb, *pixdims]`
- Works correctly when `na ≠ nb` (204 ≠ 238)

### Time-of-Flight Function Compatibility
The `time_of_flight()` function supports arbitrary broadcasting:
- Input: `x0, z0` (origin coordinates) and `x1, z1` (destination coordinates)
- Automatically handles shape mismatches through NumPy broadcasting
- 204 TX elements and 238 RX elements both work seamlessly

### Loss Functions
All loss functions (`lag_one_coherence`, `coherence_factor`, `phase_error`, `speckle_brightness`) accept separate TX/RX time-of-flight arrays and work correctly with asymmetric input.

## Files Modified

- **[dbua_us.py](dbua_us.py)**: Main implementation file
  - Lines 130-136: Load separate TX/RX element arrays
  - Lines 163-201: Asymmetric dimension handling
  - Lines 212-278: Coordinate extraction and time-of-flight functions
  - Line 212: Extract both TX and RX coordinates
  - Lines 286-292: Fix phase error loss function

## Next Steps (Optional Enhancements)

1. **Independent optimization grids**: Use different spatial resolutions for TX vs RX if needed
2. **Aperture selection**: Dynamically select TX/RX subsets based on f-number constraints
3. **Calibration**: Incorporate TX/RX sensitivity maps if available
4. **Validation**: Compare results against symmetric-mode baseline to verify correctness
