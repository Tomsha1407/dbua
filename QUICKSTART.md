# Quick Start Guide: Asymmetric TX/RX Implementation

## Usage Examples

### Option 1: Use Full Asymmetric Setup (Recommended for Real Data)
```python
# Uses 204 TX elements and 238 RX elements as-is
main(exp_name='0003490e_20250611', loss_name='pe', n_elemnts=None, nt=800, name="asymmetric")
```

**Data dimensions**:
- iqdata: `[204, 238, 800]` — Full asymmetric probe
- TX element positions: 204 elements from tx_setup/origin
- RX element positions: 238 elements from probe/element_positions

### Option 2: Use Symmetric Cropped Setup (Backward Compatible)
```python
# Crops to 30×30 symmetric array from center of both dimensions
main(exp_name='0003490e_20250611', loss_name='pe', n_elemnts=30, nt=800, name="symmetric")
```

**Data dimensions**:
- iqdata: `[30, 30, 800]` — Symmetric cropped array
- TX element positions: 30 cropped elements
- RX element positions: 30 cropped elements

## Understanding the Data Flow

### When you call `main()`:

1. **Load element positions**:
   ```
   elpos_tx (3×204) ← f["tx_setup/[0]/origin"]
   elpos_rx (3×238) ← f["probe/element_positions"]
   ```

2. **Load IQ data and convert** to element data:
   ```
   iqdata (204×238×nt) ← channel data remapped via channel_to_element()
   ```

3. **Compute separate delays**:
   ```
   TX delays: time_of_flight(204_TX_positions, ...) → [204, ...]
   RX delays: time_of_flight(238_RX_positions, ...) → [238, ...]
   ```

4. **Beamform with independent delays**:
   ```
   image = das(iqdata[204×238], tx_delays[204], rx_delays[238])
   ```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exp_name` | str | `'0003490e_20250611'` | Data file name (without .ush5 extension) |
| `loss_name` | str | `'pe'` | Loss function: 'pe'/'sb'/'cf'/'lc' |
| `n_elemnts` | int/None | `None` | If None: use full 204×238 asymmetric; If int: crop to n×n symmetric |
| `nt` | int/None | `None` | Number of time samples; if None: use all |
| `name` | str | `None` | Suffix for output filenames (videos, images) |

## Output Files

When running optimization:

```
videos/
  ├── 0003490e_20250611asymmetric_pe.mp4  (asymmetric 204×238 run)
  └── 0003490e_20250611symmetric_pe.mp4   (symmetric 30×30 run)

images/
  ├── losses_0003490e_20250611.png        (loss curves)
  └── 0003490e_20250611asymmetric.png     (final frame)

scratch/
  └── 0003490e_20250611asymmetric.png     (per-iteration preview)
```

## Verifying the Implementation

Add this debug code to the beginning of `main()` to verify shapes:

```python
def main(exp_name, loss_name, n_elemnts=None, nt=None, name=None):
    # ... [HDF5 loading code] ...
    
    # After dimension handling (around line 201):
    print(f"✓ iqdata shape: {iqdata.shape}")
    print(f"✓ TX elements: {elpos_tx_use.shape[1]} at {elpos_tx_use.shape}")
    print(f"✓ RX elements: {elpos_rx_use.shape[1]} at {elpos_rx_use.shape}")
    
    # After coordinate extraction (around line 215):
    print(f"✓ TX coords: xe_tx={xe_tx.shape}, ze_tx={ze_tx.shape}")
    print(f"✓ RX coords: xe_rx={xe_rx.shape}, ze_rx={ze_rx.shape}")
    
    # Continue with rest of function...
```

Expected output for `n_elemnts=None`:
```
✓ iqdata shape: (204, 238, 800)
✓ TX elements: 204 at (3, 204)
✓ RX elements: 238 at (3, 238)
✓ TX coords: xe_tx=(204,), ze_tx=(204,)
✓ RX coords: xe_rx=(238,), ze_rx=(238,)
```

Expected output for `n_elemnts=30`:
```
✓ iqdata shape: (30, 30, 800)
✓ TX elements: 30 at (3, 30)
✓ RX elements: 30 at (3, 30)
✓ TX coords: xe_tx=(30,), ze_tx=(30,)
✓ RX coords: xe_rx=(30,), ze_rx=(30,)
```

## Comparison: Before vs After

### Before Implementation
- **Limitation**: Code forced TX and RX dimensions to be equal (symmetric)
- **Physical Reality**: Ignored that you have 204 TX sources vs 238 RX elements
- **Data Loss**: Had to crop or pad to symmetric dimensions, losing information

### After Implementation
- **Advantage**: Properly models 204 TX × 238 RX transducer layout
- **Physical Accuracy**: Uses correct element positions from HDF5 file
- **Backward Compatible**: Old code using `n_elemnts=X` still works

## Troubleshooting

### Error: "TX element positions shape: (3, 0)"
**Cause**: `f["tx_setup/[0]/origin"]` is empty or doesn't exist  
**Solution**: Check HDF5 file structure with:
```python
import h5py
with h5py.File('data/0003490e_20250611.ush5', 'r') as f:
    print(list(f["tx_setup"].keys()))
    print(f["tx_setup/[0]/origin"].shape)
```

### Shape Mismatch Error in `das()`
**Cause**: iqdata dimension doesn't match time-of-flight array dimension  
**Solution**: Verify iqdata and delays are created correctly. Check logs for actual shapes.

### Memory Issues with 204×238 Array
**Cause**: Large memory footprint with asymmetric dimensions  
**Solution**: Use `nt` parameter to reduce time samples: `nt=400` instead of full length

## Technical Notes

- **Reciprocity**: Both TX→RX and RX→TX paths use the same time-of-flight calculation (acoustic reciprocity). The separate arrays allow flexibility if you need different models in future.
  
- **Broadcasting**: JAX handles 204 TX elements and 238 RX elements automatically through NumPy-style broadcasting in the `das()` function.

- **Loss Functions**: All loss functions (phase error, speckle brightness, coherence factor, lag-one coherence) receive separate TX and RX delays as their `t_tx` and `t_rx` parameters.

- **Sound Speed Map**: Single 19×31 sound speed grid applies to entire image space, independent of element count.
