//! Exact DeepSeek-V4 packed FP4 format checks; this is not a tensor expansion path.

pub const FP4_VALUES_PER_BYTE: usize = 2;
pub const FP4_BLOCK_ELEMENTS: usize = 32;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedFp4E2M1x2(pub u8);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct E8M0Scale(pub u8);

const _: () = assert!(std::mem::size_of::<PackedFp4E2M1x2>() == 1);
const _: () = assert!(std::mem::size_of::<E8M0Scale>() == 1);

#[derive(Debug, PartialEq, Eq)]
pub enum Fp4LayoutError {
    LogicalColumnsNotBlockAligned(usize),
    ElementCountOverflow,
    PackedLength { expected: usize, actual: usize },
    ScaleLength { expected: usize, actual: usize },
    InvalidE8M0Scale { index: usize },
    RowOutOfBounds { row: usize, rows: usize },
    BlockOutOfBounds { block: usize, blocks: usize },
    OutputLength { expected: usize, actual: usize },
}

impl std::fmt::Display for Fp4LayoutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Fp4LayoutError {}

#[derive(Debug, Clone, Copy)]
pub struct Fp4E2M1x2Layout<'a> {
    packed: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    logical_cols: usize,
}

impl<'a> Fp4E2M1x2Layout<'a> {
    pub fn new(
        packed: &'a [u8],
        scales: &'a [u8],
        rows: usize,
        logical_cols: usize,
    ) -> Result<Self, Fp4LayoutError> {
        if logical_cols % FP4_BLOCK_ELEMENTS != 0 {
            return Err(Fp4LayoutError::LogicalColumnsNotBlockAligned(logical_cols));
        }
        let elements = rows
            .checked_mul(logical_cols)
            .ok_or(Fp4LayoutError::ElementCountOverflow)?;
        let packed_len = elements / FP4_VALUES_PER_BYTE;
        if packed.len() != packed_len {
            return Err(Fp4LayoutError::PackedLength {
                expected: packed_len,
                actual: packed.len(),
            });
        }
        let scale_len = elements / FP4_BLOCK_ELEMENTS;
        if scales.len() != scale_len {
            return Err(Fp4LayoutError::ScaleLength {
                expected: scale_len,
                actual: scales.len(),
            });
        }
        if let Some((index, _)) = scales
            .iter()
            .enumerate()
            .find(|(_, scale)| **scale == 0 || **scale == 0xff)
        {
            return Err(Fp4LayoutError::InvalidE8M0Scale { index });
        }
        Ok(Self {
            packed,
            scales,
            rows,
            logical_cols,
        })
    }

    pub fn logical_cols(self) -> usize {
        self.logical_cols
    }

    pub fn element_len(self) -> usize {
        self.rows * self.logical_cols
    }

    pub fn packed_len(self) -> usize {
        self.packed.len()
    }

    pub fn scale_len(self) -> usize {
        self.scales.len()
    }

    pub fn decode_block(
        self,
        row: usize,
        block: usize,
        output: &mut [f32],
    ) -> Result<(), Fp4LayoutError> {
        if row >= self.rows {
            return Err(Fp4LayoutError::RowOutOfBounds {
                row,
                rows: self.rows,
            });
        }
        let blocks = self.logical_cols / FP4_BLOCK_ELEMENTS;
        if block >= blocks {
            return Err(Fp4LayoutError::BlockOutOfBounds { block, blocks });
        }
        if output.len() != FP4_BLOCK_ELEMENTS {
            return Err(Fp4LayoutError::OutputLength {
                expected: FP4_BLOCK_ELEMENTS,
                actual: output.len(),
            });
        }
        let packed_start = row * (self.logical_cols / FP4_VALUES_PER_BYTE)
            + block * (FP4_BLOCK_ELEMENTS / FP4_VALUES_PER_BYTE);
        let scale =
            decode_e8m0(self.scales[row * (self.logical_cols / FP4_BLOCK_ELEMENTS) + block]);
        for (index, value) in output.iter_mut().enumerate() {
            let byte = PackedFp4E2M1x2(self.packed[packed_start + index / FP4_VALUES_PER_BYTE]);
            *value = decode_fp4(byte, index % FP4_VALUES_PER_BYTE) * scale;
        }
        Ok(())
    }
}

pub fn decode_e8m0(scale: u8) -> f32 {
    f32::from_bits((E8M0Scale(scale).0 as u32) << 23)
}

pub fn decode_fp4(byte: PackedFp4E2M1x2, value_in_byte: usize) -> f32 {
    let nibble = if value_in_byte == 0 {
        byte.0 & 0x0f
    } else {
        byte.0 >> 4
    };
    let magnitude = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if magnitude == 0.0 || nibble & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_official_e2m1_nibbles_and_e8m0_scales() {
        let expected = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (nibble, expected) in expected.into_iter().enumerate() {
            assert_eq!(decode_fp4(PackedFp4E2M1x2(nibble as u8), 0), expected);
        }
        assert_eq!(decode_e8m0(126), 0.5);
        assert_eq!(decode_e8m0(127), 1.0);
        assert_eq!(decode_e8m0(128), 2.0);
    }

    #[test]
    fn rejects_malformed_fp4_block_storage() {
        assert_eq!(
            Fp4E2M1x2Layout::new(&[], &[], 1, 31).unwrap_err(),
            Fp4LayoutError::LogicalColumnsNotBlockAligned(31)
        );
        assert_eq!(
            Fp4E2M1x2Layout::new(&[0; 15], &[127], 1, 32).unwrap_err(),
            Fp4LayoutError::PackedLength {
                expected: 16,
                actual: 15,
            }
        );
        assert_eq!(
            Fp4E2M1x2Layout::new(&[0; 16], &[0xff], 1, 32).unwrap_err(),
            Fp4LayoutError::InvalidE8M0Scale { index: 0 }
        );
        assert_eq!(
            Fp4E2M1x2Layout::new(&[0; 16], &[0], 1, 32).unwrap_err(),
            Fp4LayoutError::InvalidE8M0Scale { index: 0 }
        );
    }
}
