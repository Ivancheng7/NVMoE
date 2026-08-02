//! Fail-closed loader for the diagnostic Flash-0731 TileLang PTX bridge.
//!
//! The artifact is extracted from the locked official TileLang cache by
//! `tools/extract_flash_0731_tilelang_ref.js`. This module never invokes
//! Python or TileLang at runtime and deliberately makes no exactness claim.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use cuda_core::{CudaContext, CudaFunction, CudaModule};
use serde::Deserialize;

const SCHEMA: &str = "colibri-flash-tilelang-ref-v1";
const ARTIFACT_KIND: &str = "official_cached_ptx";
const TARGET: &str = "sm_89";
const PTX_VERSION: &str = "9.0";
const ENTRY: &str = "fp4_gemm_kernel__kernel";
const REFERENCE_DIR_ENV: &str = "COLIBRI_V4_FLASH_TILELANG_REF_DIR";

const RAW_ARGUMENT_ORDER: [&str; 6] = [
    "A: ptr<fp8_e4m3fn>",
    "B: ptr<fp4_e2m1fn_x2 packed along K>",
    "C: ptr<bf16>",
    "scales_a: ptr<fp8_e8m0>",
    "scales_b: ptr<fp8_e8m0>",
    "M: u32",
];

#[derive(Deserialize)]
struct Manifest {
    schema: String,
    artifact_kind: String,
    release_sha: String,
    kernel_py_sha256: String,
    target: String,
    ptx_version: String,
    entry: String,
    raw_argument_order: Vec<String>,
    entries: BTreeMap<String, ManifestEntry>,
}

#[derive(Deserialize)]
struct ManifestEntry {
    ptx_file: String,
    ptx_sha256: String,
    n: usize,
    k: usize,
    b_packed_shape: [usize; 2],
    scales_b_shape: [usize; 2],
    block: [u32; 3],
    dynamic_shared_bytes: u32,
}

/// A loaded PTX entry and its complete, source-validated launch contract.
pub struct TilelangFp4Kernel {
    _module: Arc<CudaModule>,
    function: CudaFunction,
    n: usize,
    k: usize,
    dynamic_shared_bytes: u32,
}

impl TilelangFp4Kernel {
    pub fn function(&self) -> &CudaFunction {
        &self.function
    }

    pub fn grid_x(&self) -> u32 {
        u32::try_from(self.n / 128).expect("locked Flash dimensions fit u32")
    }

    pub fn grid_y_for_rows(&self, rows: usize) -> Result<u32, Box<dyn Error>> {
        if rows == 0 {
            return Err("TileLang FP4 launch requires at least one row".into());
        }
        u32::try_from(rows.div_ceil(32)).map_err(|_| "TileLang FP4 grid y exceeds u32".into())
    }

    pub const fn block(&self) -> (u32, u32, u32) {
        (128, 1, 1)
    }

    pub const fn dynamic_shared_bytes(&self) -> u32 {
        self.dynamic_shared_bytes
    }

    pub const fn dimensions(&self) -> (usize, usize) {
        (self.n, self.k)
    }
}

/// Both geometries used by Flash routed experts: W1/W3 and W2.
pub struct TilelangFp4Reference {
    pub n2048_k4096: TilelangFp4Kernel,
    pub n4096_k2048: TilelangFp4Kernel,
}

impl TilelangFp4Reference {
    pub fn for_dimensions(&self, n: usize, k: usize) -> Result<&TilelangFp4Kernel, Box<dyn Error>> {
        match (n, k) {
            (2_048, 4_096) => Ok(&self.n2048_k4096),
            (4_096, 2_048) => Ok(&self.n4096_k2048),
            _ => Err(format!("TileLang Flash reference has no N={n}, K={k} entry").into()),
        }
    }
}

pub fn reference_dir_from_env() -> Result<PathBuf, Box<dyn Error>> {
    std::env::var_os(REFERENCE_DIR_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{REFERENCE_DIR_ENV} is required for tilelang_ref").into())
}

/// Validates every identity boundary before asking the CUDA driver to parse PTX.
pub fn load_for_flash(
    context: &Arc<CudaContext>,
    spec: &crate::v4_spec::V4Spec,
    model_dir: &Path,
    reference_dir: &Path,
) -> Result<TilelangFp4Reference, Box<dyn Error>> {
    if spec != &crate::v4_spec::FLASH_0731 {
        return Err("TileLang FP4 reference is only defined for Flash-0731".into());
    }
    let manifest_path = reference_dir.join("tilelang-ref-manifest.json");
    let manifest: Manifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.schema != SCHEMA || manifest.artifact_kind != ARTIFACT_KIND {
        return Err("TileLang reference manifest schema/artifact kind mismatch".into());
    }
    if manifest.release_sha != spec.revision
        || manifest.target != TARGET
        || manifest.ptx_version != PTX_VERSION
        || manifest.entry != ENTRY
        || manifest
            .raw_argument_order
            .iter()
            .map(String::as_str)
            .ne(RAW_ARGUMENT_ORDER)
    {
        return Err("TileLang reference release, target, or raw ABI mismatch".into());
    }
    let actual_kernel_sha = crate::sha256_file(&model_dir.join("inference/kernel.py"))?;
    if actual_kernel_sha != manifest.kernel_py_sha256 {
        return Err("TileLang reference kernel.py SHA-256 mismatch".into());
    }
    if manifest.entries.len() != 2 {
        return Err(
            "TileLang reference manifest must contain exactly two Flash FP4 entries".into(),
        );
    }
    Ok(TilelangFp4Reference {
        n2048_k4096: load_entry(
            context,
            reference_dir,
            required_entry(&manifest, "n2048_k4096")?,
            2_048,
            4_096,
        )?,
        n4096_k2048: load_entry(
            context,
            reference_dir,
            required_entry(&manifest, "n4096_k2048")?,
            4_096,
            2_048,
        )?,
    })
}

fn required_entry<'a>(
    manifest: &'a Manifest,
    name: &str,
) -> Result<&'a ManifestEntry, Box<dyn Error>> {
    manifest
        .entries
        .get(name)
        .ok_or_else(|| format!("TileLang reference manifest lacks {name}").into())
}

fn load_entry(
    context: &Arc<CudaContext>,
    reference_dir: &Path,
    entry: &ManifestEntry,
    expected_n: usize,
    expected_k: usize,
) -> Result<TilelangFp4Kernel, Box<dyn Error>> {
    let expected_scale_columns = expected_k / 32;
    if entry.n != expected_n
        || entry.k != expected_k
        || entry.b_packed_shape != [expected_n, expected_k / 2]
        || entry.scales_b_shape != [expected_n, expected_scale_columns]
        || entry.block != [128, 1, 1]
        || entry.dynamic_shared_bytes != 17_360
    {
        return Err("TileLang reference entry shape/launch contract mismatch".into());
    }
    let filename = Path::new(&entry.ptx_file);
    if filename.file_name().and_then(|name| name.to_str()) != Some(entry.ptx_file.as_str())
        || filename
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("ptx")
    {
        return Err("TileLang reference PTX filename is not a local .ptx file".into());
    }
    let ptx_path = reference_dir.join(filename);
    if crate::sha256_file(&ptx_path)? != entry.ptx_sha256 {
        return Err("TileLang reference PTX SHA-256 mismatch".into());
    }
    let ptx = fs::read_to_string(&ptx_path)?;
    validate_ptx(&ptx)?;
    let module = context.load_module_from_ptx_src(&ptx)?;
    let function = module.load_function(ENTRY)?;
    Ok(TilelangFp4Kernel {
        _module: module,
        function,
        n: entry.n,
        k: entry.k,
        dynamic_shared_bytes: entry.dynamic_shared_bytes,
    })
}

fn validate_ptx(ptx: &str) -> Result<(), Box<dyn Error>> {
    if ptx.contains('\0') || !ptx.starts_with(".version 9.0\n.target sm_89\n.address_size 64") {
        return Err("TileLang reference PTX header/NUL boundary mismatch".into());
    }
    let expected = [
        ".visible .entry fp4_gemm_kernel__kernel",
        ".param .u64 fp4_gemm_kernel__kernel_param_0",
        ".param .u64 fp4_gemm_kernel__kernel_param_1",
        ".param .u64 fp4_gemm_kernel__kernel_param_2",
        ".param .u64 fp4_gemm_kernel__kernel_param_3",
        ".param .u64 fp4_gemm_kernel__kernel_param_4",
        ".param .u32 fp4_gemm_kernel__kernel_param_5",
        ".maxntid 128, 1, 1",
    ];
    if expected.iter().any(|needle| !ptx.contains(needle))
        || ptx.matches(".visible .entry").count() != 1
    {
        return Err("TileLang reference PTX entry ABI mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_select_only_the_two_locked_flash_fp4_geometries() {
        assert_eq!(2_048_usize.div_ceil(128), 16);
        assert_eq!(4_096_usize.div_ceil(128), 32);
        assert_eq!(4_096_usize.div_ceil(32), 128);
        assert_eq!(2_048_usize.div_ceil(32), 64);
    }

    #[test]
    fn malformed_ptx_is_rejected_before_driver_load() {
        assert!(validate_ptx(".version 9.0\n.target sm_89\n.address_size 64\n").is_err());
        assert!(validate_ptx(".version 9.0\n.target sm_89\n.address_size 64\0").is_err());
    }
}
