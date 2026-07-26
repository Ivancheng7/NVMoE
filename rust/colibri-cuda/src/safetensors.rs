//! Synchronous, range-only safetensors access for streamed weights.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::os::raw::c_void;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

const MAX_HEADER_BYTES: u64 = 512 * 1024 * 1024;
const DIRECT_ALIGNMENT: u64 = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeReadMode {
    Buffered,
    Fadvise,
    Direct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeReadTelemetry {
    pub logical_bytes: usize,
    pub physical_bytes: usize,
}

/// Exact byte range in the backing shard, including the safetensors header offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileRange {
    pub offset: u64,
    pub len: u64,
}

impl FileRange {
    pub fn end(self) -> Option<u64> {
        self.offset.checked_add(self.len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    Bf16,
    F16,
    F32,
    I64,
    U8,
    /// DeepSeek V4 stores two FP4 E2M1 values in each I8 byte. This loader
    /// deliberately has no ordinary signed-I8 path.
    Fp4E2M1x2,
    F8E4M3,
    F8E4M3Fnuz,
    F8E5M2,
    F8E5M2Fnuz,
    F8E8M0,
}

impl TensorDtype {
    fn parse(value: &str) -> Result<Self, SafetensorsError> {
        match value {
            "BF16" => Ok(Self::Bf16),
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            "I64" => Ok(Self::I64),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::Fp4E2M1x2),
            "F8_E4M3" | "F8_E4M3FN" => Ok(Self::F8E4M3),
            "F8_E4M3FNUZ" => Ok(Self::F8E4M3Fnuz),
            "F8_E5M2" => Ok(Self::F8E5M2),
            "F8_E5M2FNUZ" => Ok(Self::F8E5M2Fnuz),
            "F8_E8M0" | "F8_E8M0FNU" => Ok(Self::F8E8M0),
            _ => Err(SafetensorsError::UnsupportedDtype(value.to_owned())),
        }
    }

    fn stored_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::I64 => 8,
            Self::Bf16 | Self::F16 => 2,
            Self::U8
            | Self::Fp4E2M1x2
            | Self::F8E4M3
            | Self::F8E4M3Fnuz
            | Self::F8E5M2
            | Self::F8E5M2Fnuz
            | Self::F8E8M0 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub byte_len: u64,
}

#[derive(Debug, Clone)]
struct TensorEntry {
    info: TensorInfo,
    offset: u64,
}

#[derive(Debug)]
pub enum SafetensorsError {
    Io(io::Error),
    InvalidIndex(String),
    UnknownShard(String),
    HeaderTooLarge(u64),
    InvalidHeader(String),
    UnsupportedDtype(String),
    MissingTensor(String),
    ShapeOverflow(String),
    ByteLengthMismatch {
        tensor: String,
        expected: u64,
        actual: u64,
    },
    InvalidOffsets(String),
    RangeOutOfBounds {
        tensor: String,
        offset: u64,
        len: u64,
        tensor_len: u64,
    },
    TruncatedRead {
        path: PathBuf,
        offset: u64,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::InvalidIndex(message) | Self::InvalidHeader(message) => {
                write!(formatter, "{message}")
            }
            Self::UnknownShard(shard) => {
                write!(formatter, "index references unknown shard {shard}")
            }
            Self::HeaderTooLarge(length) => {
                write!(formatter, "safetensors header is too large: {length} bytes")
            }
            Self::UnsupportedDtype(dtype) => {
                write!(formatter, "unsupported safetensors dtype {dtype}")
            }
            Self::MissingTensor(tensor) => write!(formatter, "missing tensor {tensor}"),
            Self::ShapeOverflow(tensor) => {
                write!(formatter, "tensor {tensor} shape overflows byte count")
            }
            Self::ByteLengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor {tensor} shape requires {expected} bytes but data span is {actual}"
            ),
            Self::InvalidOffsets(tensor) => {
                write!(formatter, "tensor {tensor} has invalid data offsets")
            }
            Self::RangeOutOfBounds {
                tensor,
                offset,
                len,
                tensor_len,
            } => write!(
                formatter,
                "tensor {tensor} range [{offset}, +{len}) exceeds {tensor_len} bytes"
            ),
            Self::TruncatedRead {
                path,
                offset,
                expected,
                actual,
            } => write!(
                formatter,
                "{} truncated at {offset}: read {actual} of {expected} bytes",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SafetensorsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for SafetensorsError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Caller-owned immutable source identity and shard locations.
#[derive(Debug, Clone)]
pub struct SourceManifest {
    revision: String,
    shards: BTreeMap<String, PathBuf>,
}

impl SourceManifest {
    pub fn new(
        revision: impl Into<String>,
        shards: impl IntoIterator<Item = (String, PathBuf)>,
    ) -> Self {
        Self {
            revision: revision.into(),
            shards: shards.into_iter().collect(),
        }
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    fn path(&self, shard: &str) -> Result<&Path, SafetensorsError> {
        self.shards
            .get(shard)
            .map(PathBuf::as_path)
            .ok_or_else(|| SafetensorsError::UnknownShard(shard.to_owned()))
    }
}

#[derive(Debug)]
pub struct ShardFile {
    path: PathBuf,
    file: File,
    data_start: u64,
    tensors: BTreeMap<String, TensorEntry>,
}

impl ShardFile {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, SafetensorsError> {
        let path = path.into();
        let file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        if file_len < 8 {
            return Err(SafetensorsError::InvalidHeader(format!(
                "{} is shorter than the safetensors prefix",
                path.display()
            )));
        }

        let mut prefix = [0_u8; 8];
        read_exact_at(&file, &path, 0, &mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(SafetensorsError::HeaderTooLarge(header_len));
        }
        let data_start = 8_u64
            .checked_add(header_len)
            .ok_or_else(|| SafetensorsError::InvalidHeader("header offset overflows u64".into()))?;
        if data_start > file_len {
            return Err(SafetensorsError::InvalidHeader(format!(
                "{} header extends past EOF",
                path.display()
            )));
        }
        let header_len =
            usize::try_from(header_len).map_err(|_| SafetensorsError::HeaderTooLarge(u64::MAX))?;
        let mut header = vec![0_u8; header_len];
        read_exact_at(&file, &path, 8, &mut header)?;
        let tensors = parse_header(&header, file_len - data_start)?;
        Ok(Self {
            path,
            file,
            data_start,
            tensors,
        })
    }

    pub fn advise_tensor_dontneed(&self, name: &str) -> Result<(), SafetensorsError> {
        let len = usize::try_from(self.tensor(name)?.byte_len)
            .map_err(|_| SafetensorsError::InvalidOffsets(name.to_owned()))?;
        self.advise_range_dontneed(name, 0, len)
    }

    pub fn advise_range_dontneed(
        &self,
        name: &str,
        offset: u64,
        len: usize,
    ) -> Result<(), SafetensorsError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_owned()))?;
        let len_u64 = u64::try_from(len).map_err(|_| SafetensorsError::RangeOutOfBounds {
            tensor: name.to_owned(),
            offset,
            len: u64::MAX,
            tensor_len: entry.info.byte_len,
        })?;
        if offset
            .checked_add(len_u64)
            .is_none_or(|end| end > entry.info.byte_len)
        {
            return Err(SafetensorsError::RangeOutOfBounds {
                tensor: name.to_owned(),
                offset,
                len: len_u64,
                tensor_len: entry.info.byte_len,
            });
        }
        let range = self.tensor_file_range(name, offset, len_u64)?;
        self.advise_dontneed_ranges(&[range])
    }

    /// Returns the exact backing-file span read by `read_at(name, offset, ..)`.
    pub fn tensor_file_range(
        &self,
        name: &str,
        offset: u64,
        len: u64,
    ) -> Result<FileRange, SafetensorsError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_owned()))?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > entry.info.byte_len)
        {
            return Err(SafetensorsError::RangeOutOfBounds {
                tensor: name.to_owned(),
                offset,
                len,
                tensor_len: entry.info.byte_len,
            });
        }
        let absolute = self
            .data_start
            .checked_add(entry.offset)
            .and_then(|start| start.checked_add(offset))
            .ok_or_else(|| SafetensorsError::InvalidOffsets(name.to_owned()))?;
        Ok(FileRange {
            offset: absolute,
            len,
        })
    }

    /// Hints the exact ranges on this shard FD after coalescing only overlap or
    /// adjacency; callers retain the unmerged source trace for accounting.
    pub fn advise_willneed_ranges(
        &self,
        ranges: &[FileRange],
    ) -> Result<Vec<FileRange>, SafetensorsError> {
        let ranges = merge_adjacent_ranges(ranges)?;
        advise_ranges(&self.file, &ranges, 3)?;
        Ok(ranges)
    }

    /// Drops every supplied exact range. A successful call is the gate's
    /// support check before either replay arm runs.
    pub fn advise_dontneed_ranges(&self, ranges: &[FileRange]) -> Result<(), SafetensorsError> {
        advise_ranges(&self.file, ranges, 4)
    }

    pub fn read_at_mode(
        &self,
        name: &str,
        offset: u64,
        out: &mut [u8],
        mode: RangeReadMode,
    ) -> Result<RangeReadTelemetry, SafetensorsError> {
        if mode == RangeReadMode::Buffered {
            self.read_at(name, offset, out)?;
            return Ok(RangeReadTelemetry {
                logical_bytes: out.len(),
                physical_bytes: out.len(),
            });
        }
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_owned()))?;
        let len = u64::try_from(out.len()).map_err(|_| SafetensorsError::RangeOutOfBounds {
            tensor: name.to_owned(),
            offset,
            len: u64::MAX,
            tensor_len: entry.info.byte_len,
        })?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > entry.info.byte_len)
        {
            return Err(SafetensorsError::RangeOutOfBounds {
                tensor: name.to_owned(),
                offset,
                len,
                tensor_len: entry.info.byte_len,
            });
        }
        let absolute = self
            .data_start
            .checked_add(entry.offset)
            .and_then(|start| start.checked_add(offset))
            .ok_or_else(|| SafetensorsError::InvalidOffsets(name.to_owned()))?;
        if mode == RangeReadMode::Fadvise {
            read_exact_at(&self.file, &self.path, absolute, out)?;
            advise_dontneed(&self.file, name, absolute, len)?;
            return Ok(RangeReadTelemetry {
                logical_bytes: out.len(),
                physical_bytes: out.len(),
            });
        }
        read_direct_aligned(&self.path, absolute, out)
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, SafetensorsError> {
        self.tensors
            .get(name)
            .map(|entry| &entry.info)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_owned()))
    }

    pub fn tensors_with_prefix(&self, prefix: &str) -> Vec<(String, TensorInfo)> {
        self.tensors
            .range(prefix.to_owned()..)
            .take_while(|(name, _)| name.starts_with(prefix))
            .map(|(name, entry)| (name.clone(), entry.info.clone()))
            .collect()
    }

    pub fn read_at(&self, name: &str, offset: u64, out: &mut [u8]) -> Result<(), SafetensorsError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_owned()))?;
        let len = u64::try_from(out.len()).map_err(|_| SafetensorsError::RangeOutOfBounds {
            tensor: name.to_owned(),
            offset,
            len: u64::MAX,
            tensor_len: entry.info.byte_len,
        })?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| SafetensorsError::RangeOutOfBounds {
                tensor: name.to_owned(),
                offset,
                len,
                tensor_len: entry.info.byte_len,
            })?;
        if end > entry.info.byte_len {
            return Err(SafetensorsError::RangeOutOfBounds {
                tensor: name.to_owned(),
                offset,
                len,
                tensor_len: entry.info.byte_len,
            });
        }
        let absolute = self
            .data_start
            .checked_add(entry.offset)
            .and_then(|start| start.checked_add(offset))
            .ok_or_else(|| SafetensorsError::InvalidOffsets(name.to_owned()))?;
        read_exact_at(&self.file, &self.path, absolute, out)
    }
}

/// Sorts a single-FD range set and combines only overlapping or exactly
/// adjacent ranges; it never fills a gap in the requested trace.
pub fn merge_adjacent_ranges(ranges: &[FileRange]) -> Result<Vec<FileRange>, SafetensorsError> {
    let mut ranges = ranges.to_vec();
    ranges.sort_unstable_by_key(|range| range.offset);
    let mut merged: Vec<FileRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if range.len == 0 {
            continue;
        }
        let end = range
            .end()
            .ok_or_else(|| SafetensorsError::InvalidOffsets("range end overflows u64".into()))?;
        if let Some(last) = merged.last_mut() {
            let last_end = last.end().ok_or_else(|| {
                SafetensorsError::InvalidOffsets("range end overflows u64".into())
            })?;
            if range.offset <= last_end {
                last.len = last_end.max(end) - last.offset;
                continue;
            }
        }
        merged.push(range);
    }
    Ok(merged)
}

fn advise_ranges(file: &File, ranges: &[FileRange], advice: i32) -> Result<(), SafetensorsError> {
    for range in ranges {
        let offset = i64::try_from(range.offset)
            .map_err(|_| SafetensorsError::InvalidOffsets("advice offset exceeds i64".into()))?;
        let len = i64::try_from(range.len)
            .map_err(|_| SafetensorsError::InvalidOffsets("advice length exceeds i64".into()))?;
        if len == 0 {
            continue;
        }
        // SAFETY: the descriptor remains live and the supplied range was
        // validated against this shard before entering the gate.
        let error = unsafe { posix_fadvise(file.as_raw_fd(), offset, len, advice) };
        if error != 0 {
            return Err(SafetensorsError::Io(io::Error::from_raw_os_error(error)));
        }
    }
    Ok(())
}

unsafe extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

fn advise_dontneed(file: &File, name: &str, offset: u64, len: u64) -> Result<(), SafetensorsError> {
    let _ = name;
    advise_ranges(file, &[FileRange { offset, len }], 4)
}

fn read_direct_aligned(
    path: &Path,
    absolute: u64,
    out: &mut [u8],
) -> Result<RangeReadTelemetry, SafetensorsError> {
    const O_DIRECT: i32 = 0x4000;
    let aligned_start = absolute / DIRECT_ALIGNMENT * DIRECT_ALIGNMENT;
    let prefix = usize::try_from(absolute - aligned_start)
        .map_err(|_| SafetensorsError::InvalidOffsets(path.display().to_string()))?;
    let needed = prefix
        .checked_add(out.len())
        .ok_or_else(|| SafetensorsError::InvalidOffsets(path.display().to_string()))?;
    let physical = needed
        .checked_add(DIRECT_ALIGNMENT as usize - 1)
        .map(|value| value / DIRECT_ALIGNMENT as usize * DIRECT_ALIGNMENT as usize)
        .ok_or_else(|| SafetensorsError::InvalidOffsets(path.display().to_string()))?;
    let direct = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(path)?;
    unsafe extern "C" {
        fn posix_memalign(pointer: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        fn free(pointer: *mut c_void);
    }
    let mut pointer = std::ptr::null_mut();
    // SAFETY: the successful allocation is converted to one temporary slice
    // and freed after the single direct read.
    let error = unsafe { posix_memalign(&mut pointer, DIRECT_ALIGNMENT as usize, physical) };
    if error != 0 {
        return Err(SafetensorsError::Io(io::Error::from_raw_os_error(error)));
    }
    let read_result = (|| {
        // SAFETY: posix_memalign returned `physical` writable bytes.
        let buffer = unsafe { std::slice::from_raw_parts_mut(pointer.cast::<u8>(), physical) };
        let actual = direct.read_at(buffer, aligned_start)?;
        if actual < needed {
            return Err(SafetensorsError::TruncatedRead {
                path: path.to_owned(),
                offset: aligned_start,
                expected: needed,
                actual,
            });
        }
        out.copy_from_slice(&buffer[prefix..prefix + out.len()]);
        Ok(RangeReadTelemetry {
            logical_bytes: out.len(),
            physical_bytes: actual,
        })
    })();
    // SAFETY: pointer came from posix_memalign and is no longer borrowed.
    unsafe { free(pointer) };
    read_result
}

#[derive(Debug)]
pub struct IndexedSafetensors {
    manifest: SourceManifest,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLocation {
    pub shard: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRange {
    pub location: TensorLocation,
    pub info: TensorInfo,
    pub offset: u64,
}

impl IndexedSafetensors {
    pub fn from_index_reader(
        manifest: SourceManifest,
        mut reader: impl Read,
    ) -> Result<Self, SafetensorsError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let index: IndexFile = serde_json::from_slice(&bytes)
            .map_err(|error| SafetensorsError::InvalidIndex(error.to_string()))?;
        if index.weight_map.is_empty() {
            return Err(SafetensorsError::InvalidIndex("weight_map is empty".into()));
        }
        Ok(Self {
            manifest,
            weight_map: index.weight_map,
        })
    }

    pub fn revision(&self) -> &str {
        self.manifest.revision()
    }

    pub fn resolve_tensor(&self, name: &str) -> Result<TensorLocation, SafetensorsError> {
        let shard = self
            .weight_map
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_owned()))?;
        Ok(TensorLocation {
            shard: shard.clone(),
            path: self.manifest.path(shard)?.to_owned(),
        })
    }

    pub fn open_tensor(&self, name: &str) -> Result<ShardFile, SafetensorsError> {
        let location = self.resolve_tensor(name)?;
        let shard = ShardFile::open(location.path)?;
        shard.tensor(name)?;
        Ok(shard)
    }

    pub fn resolve_tensor_range(&self, name: &str) -> Result<TensorRange, SafetensorsError> {
        let location = self.resolve_tensor(name)?;
        let shard = ShardFile::open(&location.path)?;
        let entry = shard
            .tensors
            .get(name)
            .ok_or_else(|| SafetensorsError::MissingTensor(name.to_owned()))?;
        Ok(TensorRange {
            location,
            info: entry.info.clone(),
            offset: shard
                .data_start
                .checked_add(entry.offset)
                .ok_or_else(|| SafetensorsError::InvalidOffsets(name.to_owned()))?,
        })
    }
}

#[derive(Deserialize)]
struct IndexFile {
    weight_map: BTreeMap<String, String>,
}

fn parse_header(
    header: &[u8],
    data_len: u64,
) -> Result<BTreeMap<String, TensorEntry>, SafetensorsError> {
    let value: Value = serde_json::from_slice(header)
        .map_err(|error| SafetensorsError::InvalidHeader(error.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| SafetensorsError::InvalidHeader("header must be a JSON object".into()))?;
    let mut tensors = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let record = value.as_object().ok_or_else(|| {
            SafetensorsError::InvalidHeader(format!("tensor {name} must be a JSON object"))
        })?;
        let dtype = record
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("tensor {name} lacks dtype")))?;
        let dtype = TensorDtype::parse(dtype)?;
        let shape = record
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| SafetensorsError::InvalidHeader(format!("tensor {name} lacks shape")))?
            .iter()
            .map(|dimension| {
                dimension.as_u64().ok_or_else(|| {
                    SafetensorsError::InvalidHeader(format!("tensor {name} has a non-u64 shape"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = record
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| SafetensorsError::InvalidOffsets(name.clone()))?;
        let offset = offsets[0]
            .as_u64()
            .ok_or_else(|| SafetensorsError::InvalidOffsets(name.clone()))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| SafetensorsError::InvalidOffsets(name.clone()))?;
        if end < offset || end > data_len {
            return Err(SafetensorsError::InvalidOffsets(name.clone()));
        }
        let elements = shape
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension));
        let elements = elements.ok_or_else(|| SafetensorsError::ShapeOverflow(name.clone()))?;
        let expected = elements
            .checked_mul(dtype.stored_bytes())
            .ok_or_else(|| SafetensorsError::ShapeOverflow(name.clone()))?;
        let byte_len = end - offset;
        if expected != byte_len {
            return Err(SafetensorsError::ByteLengthMismatch {
                tensor: name.clone(),
                expected,
                actual: byte_len,
            });
        }
        tensors.insert(
            name.clone(),
            TensorEntry {
                info: TensorInfo {
                    dtype,
                    shape,
                    byte_len,
                },
                offset,
            },
        );
    }
    Ok(tensors)
}

fn read_exact_at(
    file: &File,
    path: &Path,
    offset: u64,
    out: &mut [u8],
) -> Result<(), SafetensorsError> {
    let mut read = 0;
    while read < out.len() {
        let current = offset
            .checked_add(u64::try_from(read).expect("usize fits u64 on supported targets"))
            .ok_or_else(|| SafetensorsError::InvalidHeader("read offset overflows u64".into()))?;
        match file.read_at(&mut out[read..], current) {
            Ok(0) => {
                return Err(SafetensorsError::TruncatedRead {
                    path: path.to_owned(),
                    offset: current,
                    expected: out.len(),
                    actual: read,
                });
            }
            Ok(count) => read += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_file(name: &str, header: &str, data: &[u8]) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("colibri-{name}-{unique}.safetensors"));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn reads_a_bounded_raw_range_and_preserves_v4_packed_fp4() {
        let header = r#"{"dense":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"packed":{"dtype":"I8","shape":[2,2],"data_offsets":[8,12]}}"#;
        let path = temporary_file(
            "range",
            header,
            &[1, 2, 3, 4, 5, 6, 7, 8, 0x21, 0x43, 0x65, 0x87],
        );
        let manifest = SourceManifest::new(
            "pinned-test-revision",
            [("one.safetensors".to_owned(), path.clone())],
        );
        let index = IndexedSafetensors::from_index_reader(
            manifest,
            br#"{"weight_map":{"packed":"one.safetensors"}}"#.as_slice(),
        )
        .unwrap();
        assert_eq!(index.revision(), "pinned-test-revision");
        let range = index.resolve_tensor_range("packed").unwrap();
        assert_eq!(range.location.shard, "one.safetensors");
        assert_eq!(range.info.dtype, TensorDtype::Fp4E2M1x2);
        assert_eq!(range.info.shape, [2, 2]);
        assert_eq!(range.info.byte_len, 4);
        assert_eq!(range.offset, 8 + header.len() as u64 + 8);
        let shard = index.open_tensor("packed").unwrap();
        assert_eq!(
            shard.tensor("packed").unwrap().dtype,
            TensorDtype::Fp4E2M1x2
        );
        let mut output = [0_u8; 2];
        shard.read_at("packed", 1, &mut output).unwrap();
        assert_eq!(output, [0x43, 0x65]);
        assert!(matches!(
            shard.read_at("packed", 3, &mut output),
            Err(SafetensorsError::RangeOutOfBounds { .. })
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_a_corrupt_tensor_span() {
        let header = r#"{"bad":{"dtype":"F32","shape":[2],"data_offsets":[0,7]}}"#;
        let path = temporary_file("corrupt", header, &[0; 7]);
        assert!(matches!(
            ShardFile::open(&path),
            Err(SafetensorsError::ByteLengthMismatch { .. })
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fadvise_and_direct_ranges_match_buffered_bytes() {
        let header = r#"{"value":{"dtype":"U8","shape":[8192],"data_offsets":[0,8192]}}"#;
        let data = (0..8_192).map(|index| index as u8).collect::<Vec<_>>();
        let path = temporary_file("read-modes", header, &data);
        let shard = ShardFile::open(&path).unwrap();
        let mut buffered = vec![0_u8; 4_321];
        let mut fadvise = vec![0_u8; buffered.len()];
        let mut direct = vec![0_u8; buffered.len()];
        shard
            .read_at_mode("value", 137, &mut buffered, RangeReadMode::Buffered)
            .unwrap();
        shard
            .read_at_mode("value", 137, &mut fadvise, RangeReadMode::Fadvise)
            .unwrap();
        let direct_telemetry = shard
            .read_at_mode("value", 137, &mut direct, RangeReadMode::Direct)
            .unwrap();
        assert_eq!(buffered, fadvise);
        assert_eq!(buffered, direct);
        assert_eq!(direct_telemetry.logical_bytes, buffered.len());
        assert!(direct_telemetry.physical_bytes >= buffered.len());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn merges_only_overlapping_or_adjacent_file_ranges() {
        let merged = merge_adjacent_ranges(&[
            FileRange { offset: 40, len: 5 },
            FileRange {
                offset: 10,
                len: 10,
            },
            FileRange { offset: 20, len: 7 },
            FileRange { offset: 24, len: 9 },
            FileRange { offset: 60, len: 0 },
        ])
        .unwrap();
        assert_eq!(
            merged,
            [
                FileRange {
                    offset: 10,
                    len: 23
                },
                FileRange { offset: 40, len: 5 },
            ]
        );
    }

    #[test]
    #[ignore = "requires COLIBRI_DSV4_MODEL_DIR pinned ext4 snapshot"]
    fn reads_the_explicit_v4_config_and_first_shard() {
        let root = std::env::var_os("COLIBRI_DSV4_MODEL_DIR")
            .map(PathBuf::from)
            .expect("set COLIBRI_DSV4_MODEL_DIR to the pinned ext4 snapshot");
        crate::v4_state::validate_model_dir(&root).unwrap();
        let config = root.join("config.json");
        let shard = root.join("model-00001-of-00064.safetensors");
        assert!(config.is_file(), "missing pinned V4 config");
        assert!(shard.is_file(), "missing pinned V4 first shard");
        let config: Value = serde_json::from_slice(&fs::read(config).unwrap()).unwrap();
        assert_eq!(config["model_type"], "deepseek_v4");
        let shard = ShardFile::open(shard).unwrap();
        let embedding = shard.tensor("embed.weight").unwrap();
        assert_eq!(embedding.dtype, TensorDtype::Bf16);
        assert_eq!(embedding.shape, [129_280, 7_168]);

        let index = root.join("model.safetensors.index.json");
        if !index.is_file() {
            eprintln!(
                "NOTE: V4 index is unavailable; validated config and first shard header only"
            );
            return;
        }
        let manifest = SourceManifest::new(
            "b5968e9190ef611bbf34a7229255be88a0e937c1",
            [(
                "model-00001-of-00064.safetensors".to_owned(),
                root.join("model-00001-of-00064.safetensors"),
            )],
        );
        let indexed =
            IndexedSafetensors::from_index_reader(manifest, fs::File::open(index).unwrap())
                .unwrap();
        assert_eq!(
            indexed
                .open_tensor("embed.weight")
                .unwrap()
                .tensor("embed.weight")
                .unwrap()
                .dtype,
            TensorDtype::Bf16
        );
    }
}
