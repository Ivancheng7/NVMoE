use super::*;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RowsAddError {
    ZeroWidth,
    InputNotWholeRows,
    PartialLength { expected: usize, actual: usize },
    DeviceRowsLength { expected: usize, actual: usize },
    NegativeRow(i32),
    RowOutOfRange { row: i32, row_count: usize },
    DuplicateRow(i32),
    IntegerOverflow,
}

impl std::fmt::Display for RowsAddError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for RowsAddError {}

pub(crate) fn validate_rows_add(
    x_len: usize,
    partial_len: usize,
    device_rows_len: usize,
    rows: &[i32],
    d: usize,
) -> Result<Option<(u32, u32)>, RowsAddError> {
    if d == 0 {
        return Err(RowsAddError::ZeroWidth);
    }
    if x_len % d != 0 {
        return Err(RowsAddError::InputNotWholeRows);
    }
    let partial_len_expected = rows
        .len()
        .checked_mul(d)
        .ok_or(RowsAddError::IntegerOverflow)?;
    if partial_len != partial_len_expected {
        return Err(RowsAddError::PartialLength {
            expected: partial_len_expected,
            actual: partial_len,
        });
    }
    if device_rows_len != rows.len() {
        return Err(RowsAddError::DeviceRowsLength {
            expected: rows.len(),
            actual: device_rows_len,
        });
    }

    let row_count = x_len / d;
    let mut seen = HashSet::with_capacity(rows.len());
    for &row in rows {
        let row_index = usize::try_from(row).map_err(|_| RowsAddError::NegativeRow(row))?;
        if row_index >= row_count {
            return Err(RowsAddError::RowOutOfRange { row, row_count });
        }
        if !seen.insert(row) {
            return Err(RowsAddError::DuplicateRow(row));
        }
    }

    let elements =
        u32::try_from(partial_len_expected).map_err(|_| RowsAddError::IntegerOverflow)?;
    let d = u32::try_from(d).map_err(|_| RowsAddError::IntegerOverflow)?;
    Ok((elements != 0).then_some((elements, d)))
}

pub(crate) fn launch_rows_add(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    x: &mut DeviceBuffer<f32>,
    partial: &DeviceBuffer<f32>,
    device_rows: &DeviceBuffer<i32>,
    rows: &[i32],
    d: usize,
) -> Result<(), Box<dyn Error>> {
    let Some((elements, d)) =
        validate_rows_add(x.len(), partial.len(), device_rows.len(), rows, d)?
    else {
        return Ok(());
    };
    // SAFETY: validation proves the 1-D launch has one partial element per
    // thread, matching device row indices, in-range unique target rows, and
    // lossless kernel scalar conversions. This only enqueues work on stream.
    unsafe {
        module.rows_add_in_place(
            stream,
            LaunchConfig::for_num_elems(elements),
            x,
            partial,
            device_rows,
            d,
        )
    }?;
    Ok(())
}

pub(crate) fn launch_v4_fp4_decode(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    layout: v4_fp4::Fp4E2M1x2Layout,
) -> Result<(), Box<dyn Error>> {
    if packed.len() != layout.packed_len()
        || scales.len() != layout.scale_len()
        || output.len() != layout.element_len()
    {
        return Err("FP4 device buffer length disagrees with validated layout".into());
    }
    let elements =
        u32::try_from(layout.element_len()).map_err(|_| "FP4 decode exceeds 1-D launch")?;
    let logical_cols =
        u32::try_from(layout.logical_cols()).map_err(|_| "FP4 columns exceed u32")?;
    // SAFETY: Fp4E2M1x2Layout proves the packed half-column and one-scale-per-32
    // addressing used by the kernel; this only enqueues the format-oracle launch.
    unsafe {
        module.decode_v4_fp4_blockwise(
            stream,
            LaunchConfig::for_num_elems(elements),
            packed,
            scales,
            output,
            logical_cols,
        )
    }?;
    Ok(())
}

pub(crate) fn run_v4_router_gpu_oracle(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    input: &[u16],
    norm: &[u16],
    logits: &[f32],
    bias: &[f32],
    hash_ids: &[i64],
    hash_mode: bool,
) -> Result<(Vec<u16>, v4_router::RouterSelection), Box<dyn Error>> {
    let rows = input.len() / v4_router::V4_HIDDEN;
    if input.len() % v4_router::V4_HIDDEN != 0
        || norm.len() != v4_router::V4_HIDDEN
        || logits.len() != rows * v4_router::V4_ROUTED_EXPERTS
        || bias.len() != v4_router::V4_ROUTED_EXPERTS
        || (hash_mode && hash_ids.len() != rows * v4_router::V4_TOPK)
    {
        return Err("validated V4 router buffer length disagrees with launch shape".into());
    }
    let rows = u32::try_from(rows).map_err(|_| "V4 router rows exceed 1-D launch")?;
    let input = DeviceBuffer::from_host(stream, input)?;
    let norm = DeviceBuffer::from_host(stream, norm)?;
    let mut normalized = DeviceBuffer::from_host(stream, &vec![0_u16; input.len()])?;
    let logits = DeviceBuffer::from_host(stream, logits)?;
    let bias = DeviceBuffer::from_host(stream, bias)?;
    let hash_ids = DeviceBuffer::from_host(stream, hash_ids)?;
    let mut ids =
        DeviceBuffer::from_host(stream, &vec![0_i32; rows as usize * v4_router::V4_TOPK])?;
    let mut weights =
        DeviceBuffer::from_host(stream, &vec![0_f32; rows as usize * v4_router::V4_TOPK])?;
    let mut trace_softplus = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let mut trace_post_scores = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let mut trace_selected_scores = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    let mut trace_denom_steps = DeviceBuffer::<f32>::zeroed(stream, 0)?;
    // SAFETY: validated fixed V4 dimensions prove every kernel slice length and
    // scalar conversion. Both launches stay on the stream; terminal copies below
    // are smoke-only parity boundaries.
    unsafe {
        module.rmsnorm_v4_bf16(
            stream,
            LaunchConfig::for_num_elems(rows),
            &input,
            &norm,
            &mut normalized,
            v4_router::V4_HIDDEN as u32,
        )?;
        module.select_v4_router(
            stream,
            LaunchConfig::for_num_elems(rows),
            &logits,
            &bias,
            &hash_ids,
            &mut ids,
            &mut weights,
            &mut trace_softplus,
            &mut trace_post_scores,
            &mut trace_selected_scores,
            &mut trace_denom_steps,
            hash_mode as u32,
            u32::try_from(v4_router::V4_ROUTED_EXPERTS)?,
            v4_router::V4_ROUTE_SCALE,
            FlashRouterScoreArm::Canonical.kernel_flag(),
        )?;
    }
    Ok((
        normalized.to_host_vec(stream)?,
        v4_router::RouterSelection {
            ids: ids.to_host_vec(stream)?,
            weights: weights.to_host_vec(stream)?,
        },
    ))
}

pub(crate) fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: output length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2e-5 * expected.abs().max(1.0);
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "{name}: index {index}, expected {expected}, got {actual}"
        );
    }
}

pub(crate) fn v4_model_dir() -> Result<std::path::PathBuf, Box<dyn Error>> {
    v4_model_dir_for(&v4_spec::PRO)
}

pub(crate) fn v4_model_dir_for(
    spec: &'static v4_spec::V4Spec,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let path = std::env::var_os("COLIBRI_DSV4_MODEL_DIR")
        .or_else(|| std::env::var_os("COLIBRI_DSV4_DIR"))
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("COLIBRI_DSV4_EXPERT_SHARD")
                .map(std::path::PathBuf::from)
                .and_then(|path| path.parent().map(std::path::Path::to_owned))
        })
        .ok_or("set COLIBRI_DSV4_MODEL_DIR to the pinned ext4 snapshot")?;
    v4_state::validate_model_dir_for(spec, &path)?;
    let _ = v4_state::base_indexed_model(spec, &path)?;
    Ok(path)
}

pub(crate) fn v4_expert_shard() -> Result<std::path::PathBuf, Box<dyn Error>> {
    let path = std::env::var_os("COLIBRI_DSV4_EXPERT_SHARD")
        .map(std::path::PathBuf::from)
        .unwrap_or(v4_model_dir()?.join("model-00002-of-00064.safetensors"));
    if path.file_name().and_then(|name| name.to_str()) != Some("model-00002-of-00064.safetensors")
        || path.symlink_metadata()?.file_type().is_symlink()
        || !path.is_file()
    {
        return Err("COLIBRI_DSV4_EXPERT_SHARD must be the regular pinned shard 2".into());
    }
    v4_state::validate_model_dir(
        path.parent()
            .ok_or("COLIBRI_DSV4_EXPERT_SHARD has no model directory")?,
    )?;
    Ok(path)
}

pub(crate) fn synthetic_only() -> Result<bool, Box<dyn Error>> {
    match std::env::var_os("COLIBRI_SYNTHETIC_ONLY") {
        None => Ok(false),
        Some(value) if value == "1" => Ok(true),
        Some(_) => Err("COLIBRI_SYNTHETIC_ONLY must be exactly 1 when set".into()),
    }
}

pub(crate) fn sha256_file(path: &std::path::Path) -> Result<String, Box<dyn Error>> {
    let output = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let hash = std::str::from_utf8(&output.stdout)?
        .split_whitespace()
        .next()
        .ok_or("sha256sum returned no digest")?;
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "sha256sum returned an invalid digest for {}",
            path.display()
        )
        .into());
    }
    Ok(hash.to_ascii_lowercase())
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> Result<String, Box<dyn Error>> {
    let mut child = std::process::Command::new("sha256sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("sha256sum stdin is unavailable")?
        .write_all(bytes)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err("sha256sum failed for canonical bytes".into());
    }
    let hash = std::str::from_utf8(&output.stdout)?
        .split_whitespace()
        .next()
        .ok_or("sha256sum returned no digest")?;
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("sha256sum returned an invalid digest for canonical bytes".into());
    }
    Ok(hash.to_ascii_lowercase())
}

pub(crate) fn validate_source_identity(
    source_identity: &str,
    manifest_path: &std::path::Path,
    binary_path: &std::path::Path,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if source_identity.len() != 64
        || !source_identity
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("COLIBRI_SOURCE_IDENTITY must be 64 lowercase hexadecimal characters".into());
    }
    let manifest_metadata = manifest_path.symlink_metadata()?;
    if manifest_metadata.file_type().is_symlink() || !manifest_metadata.is_file() {
        return Err("source identity manifest must be a regular non-symlink file".into());
    }
    let manifest_path = manifest_path.canonicalize()?;
    if sha256_file(&manifest_path)? != source_identity {
        return Err("source identity does not match the manifest digest".into());
    }
    let source_root = manifest_path
        .parent()
        .ok_or("source identity manifest has no parent directory")?
        .canonicalize()?;
    let mut seen = HashSet::new();
    for (line_number, line) in std::fs::read_to_string(&manifest_path)?.lines().enumerate() {
        let (expected, relative) = line
            .split_once("  ")
            .ok_or_else(|| format!("invalid source identity manifest line {}", line_number + 1))?;
        if expected.len() != 64
            || !expected
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("invalid digest on manifest line {}", line_number + 1).into());
        }
        let relative = std::path::Path::new(relative);
        if relative.is_absolute()
            || relative
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(format!("unsafe source path on manifest line {}", line_number + 1).into());
        }
        let source = source_root.join(relative);
        let metadata = source.symlink_metadata()?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!(
                "source {} is not a regular non-symlink file",
                source.display()
            )
            .into());
        }
        let source = source.canonicalize()?;
        if !source.starts_with(&source_root) || !seen.insert(source.clone()) {
            return Err(format!("invalid or duplicate source {}", source.display()).into());
        }
        if sha256_file(&source)? != expected {
            return Err(format!("source identity mismatch for {}", source.display()).into());
        }
    }
    if seen.is_empty() {
        return Err("source identity manifest is empty".into());
    }
    let binary_metadata = binary_path.symlink_metadata()?;
    if binary_metadata.file_type().is_symlink() || !binary_metadata.is_file() {
        return Err("benchmark binary must be a regular non-symlink file".into());
    }
    let binary_path = binary_path.canonicalize()?;
    let binary_sha256 = sha256_file(&binary_path)?;
    Ok(serde_json::json!({
        "source_identity_sha256": source_identity,
        "source_manifest": manifest_path,
        "binary": binary_path,
        "binary_sha256": binary_sha256,
        "build_mode": if cfg!(debug_assertions) { "debug" } else { "release" },
    }))
}

pub(crate) fn benchmark_provenance() -> Result<serde_json::Value, Box<dyn Error>> {
    let identity = std::env::var("COLIBRI_SOURCE_IDENTITY")
        .map_err(|_| "benchmark requires COLIBRI_SOURCE_IDENTITY; use rust/run-identified.sh")?;
    let manifest = std::env::var_os("COLIBRI_SOURCE_IDENTITY_MANIFEST")
        .map(std::path::PathBuf::from)
        .ok_or("benchmark requires COLIBRI_SOURCE_IDENTITY_MANIFEST; use rust/run-identified.sh")?;
    validate_source_identity(&identity, &manifest, &std::env::current_exe()?)
}

pub(crate) struct PendingResultFile {
    path: std::path::PathBuf,
    pub(crate) temp_path: std::path::PathBuf,
    file: Option<std::fs::File>,
    committed: bool,
}

impl PendingResultFile {
    pub(crate) fn reserve(path: std::path::PathBuf) -> Result<Self, Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if path.exists() {
            return Err(format!(
                "refusing to overwrite V4 trace artifact {}: final path exists",
                path.display()
            )
            .into());
        }
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let file_name = path
            .file_name()
            .ok_or("V4 trace artifact path has no filename")?
            .to_string_lossy();
        static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let (temp_path, file) = loop {
            let sequence = NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let temp_path = parent.join(format!(
                ".{file_name}.tmp-{}-{sequence}",
                std::process::id()
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)
            {
                Ok(file) => break (temp_path, file),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "create pending V4 trace artifact {}: {error}",
                        temp_path.display()
                    )
                    .into());
                }
            }
        };
        Ok(Self {
            path,
            temp_path,
            file: Some(file),
            committed: false,
        })
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub(crate) fn file_mut(&mut self) -> &mut std::fs::File {
        self.file.as_mut().expect("pending result file disappeared")
    }

    pub(crate) fn commit(&mut self) -> Result<(), Box<dyn Error>> {
        self.file_mut().flush()?;
        self.file_mut().sync_all()?;
        drop(self.file.take());
        if self.path.exists() {
            return Err(format!(
                "refusing to overwrite V4 trace artifact {}: final path exists",
                self.path.display()
            )
            .into());
        }
        match std::fs::hard_link(&self.temp_path, &self.path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(format!(
                    "refusing to overwrite V4 trace artifact {}: final path exists",
                    self.path.display()
                )
                .into());
            }
            Err(error) => {
                return Err(format!(
                    "atomically publish V4 trace artifact {} from {}: {error}",
                    self.path.display(),
                    self.temp_path.display()
                )
                .into());
            }
        }
        std::fs::remove_file(&self.temp_path).map_err(|error| {
            format!(
                "remove published V4 trace temp artifact {}: {error}",
                self.temp_path.display()
            )
        })?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for PendingResultFile {
    fn drop(&mut self) {
        if !self.committed {
            drop(self.file.take());
            let _ = std::fs::remove_file(&self.temp_path);
        }
    }
}
