use std::error::Error;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

pub const LIBRARY_SHA256: &str = "4cc45526449a95d3985389785e985b7832460ea3db178bc9e9963be2111fbee8";
pub const EXPECTED_VERSION: i32 = 120_405;
pub const WORKSPACE_BYTES: usize = 8_519_680;

const RTLD_LOCAL: c_int = 0;
const RTLD_NOW: c_int = 2;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUBLAS_DEFAULT_MATH: c_int = 0;

type CublasCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type CublasDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type CublasGetVersion = unsafe extern "C" fn(*mut c_void, *mut c_int) -> c_int;
type CublasSetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type CublasSetWorkspace = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int;
type CublasSetMathMode = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type CublasSgemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const c_void,
    c_int,
    *const c_void,
    c_int,
    *const f32,
    *mut c_void,
    c_int,
) -> c_int;

#[repr(C)]
struct DlInfo {
    dli_fname: *const c_char,
    dli_fbase: *mut c_void,
    dli_sname: *const c_char,
    dli_saddr: *mut c_void,
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
}

#[derive(Clone, Debug)]
pub struct Provenance {
    pub version: i32,
    pub resolved_path: PathBuf,
    pub dladdr_path: PathBuf,
    pub maps_path: PathBuf,
    pub actual_sha256: String,
}

#[derive(Clone, Copy)]
struct Symbols {
    destroy: CublasDestroy,
    set_stream: CublasSetStream,
    set_workspace: CublasSetWorkspace,
    set_math_mode: CublasSetMathMode,
    sgemm: CublasSgemm,
}

struct Library(*mut c_void);

impl Library {
    fn open(path: &Path) -> Result<Self, Box<dyn Error>> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())?;
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            return Err(format!("dlopen cuBLAS failed: {}", dl_error()).into());
        }
        Ok(Self(handle))
    }

    fn symbol(&self, name: &'static [u8]) -> Result<*mut c_void, Box<dyn Error>> {
        let symbol = unsafe { dlsym(self.0, name.as_ptr().cast()) };
        if symbol.is_null() {
            return Err(format!(
                "dlsym {} failed: {}",
                CStr::from_bytes_with_nul(name)?.to_string_lossy(),
                dl_error()
            )
            .into());
        }
        Ok(symbol)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                dlclose(self.0);
            }
        }
    }
}

pub struct CublasExact {
    _library: Library,
    handle: *mut c_void,
    symbols: Symbols,
    provenance: Provenance,
}

impl CublasExact {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        let requested = std::env::var_os("COLIBRI_CUBLAS_EXACT_LIB")
            .map(PathBuf::from)
            .ok_or("strict cuBLAS exact mode requires absolute COLIBRI_CUBLAS_EXACT_LIB")?;
        require_absolute_path(&requested)?;
        let metadata = requested.symlink_metadata()?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err("COLIBRI_CUBLAS_EXACT_LIB must be a regular non-symlink file".into());
        }
        let requested = requested.canonicalize()?;
        let actual_sha256 = sha256_file(&requested)?;
        if actual_sha256 != LIBRARY_SHA256 {
            return Err(format!(
                "pinned cuBLAS library SHA-256 drift: expected {LIBRARY_SHA256}, got {actual_sha256}"
            )
            .into());
        }
        let library = Library::open(&requested)?;
        let create = unsafe {
            std::mem::transmute::<*mut c_void, CublasCreate>(library.symbol(b"cublasCreate_v2\0")?)
        };
        let get_version = unsafe {
            std::mem::transmute::<*mut c_void, CublasGetVersion>(
                library.symbol(b"cublasGetVersion_v2\0")?,
            )
        };
        let symbols = Symbols {
            destroy: unsafe {
                std::mem::transmute::<*mut c_void, CublasDestroy>(
                    library.symbol(b"cublasDestroy_v2\0")?,
                )
            },
            set_stream: unsafe {
                std::mem::transmute::<*mut c_void, CublasSetStream>(
                    library.symbol(b"cublasSetStream_v2\0")?,
                )
            },
            set_workspace: unsafe {
                std::mem::transmute::<*mut c_void, CublasSetWorkspace>(
                    library.symbol(b"cublasSetWorkspace_v2\0")?,
                )
            },
            set_math_mode: unsafe {
                std::mem::transmute::<*mut c_void, CublasSetMathMode>(
                    library.symbol(b"cublasSetMathMode\0")?,
                )
            },
            sgemm: unsafe {
                std::mem::transmute::<*mut c_void, CublasSgemm>(
                    library.symbol(b"cublasSgemm_v2\0")?,
                )
            },
        };
        let dladdr_path = resolved_symbol_path(symbols.sgemm as *const c_void)?;
        let maps_path = resolved_maps_path(&requested)?;
        if dladdr_path != requested || maps_path != requested {
            return Err(format!(
                "cuBLAS resolved-path mismatch: requested={}, dladdr={}, maps={}",
                requested.display(),
                dladdr_path.display(),
                maps_path.display()
            )
            .into());
        }

        let mut handle = std::ptr::null_mut();
        check(unsafe { create(&mut handle) }, "cublasCreate_v2")?;
        if handle.is_null() {
            return Err("cublasCreate_v2 returned a null handle".into());
        }
        let mut version = 0;
        if let Err(error) = check(
            unsafe { get_version(handle, &mut version) },
            "cublasGetVersion_v2",
        ) {
            unsafe {
                (symbols.destroy)(handle);
            }
            return Err(error);
        }
        if version != EXPECTED_VERSION {
            unsafe {
                (symbols.destroy)(handle);
            }
            return Err(format!(
                "cuBLAS version drift: expected {EXPECTED_VERSION}, got {version}"
            )
            .into());
        }
        Ok(Self {
            _library: library,
            handle,
            symbols,
            provenance: Provenance {
                version,
                resolved_path: requested,
                dladdr_path,
                maps_path,
                actual_sha256,
            },
        })
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_mn1(
        &self,
        stream: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        weight: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        m: usize,
        k: usize,
    ) -> Result<(), Box<dyn Error>> {
        if workspace.is_null()
            || workspace_bytes != WORKSPACE_BYTES
            || weight.is_null()
            || input.is_null()
            || output.is_null()
            || m == 0
            || k == 0
        {
            return Err("cuBLAS exact SGEMM buffer/geometry contract mismatch".into());
        }
        let m = c_int::try_from(m)?;
        let k = c_int::try_from(k)?;
        check(
            unsafe { (self.symbols.set_stream)(self.handle, stream) },
            "cublasSetStream_v2",
        )?;
        check(
            unsafe { (self.symbols.set_workspace)(self.handle, workspace, workspace_bytes) },
            "cublasSetWorkspace_v2",
        )?;
        check(
            unsafe { (self.symbols.set_math_mode)(self.handle, CUBLAS_DEFAULT_MATH) },
            "cublasSetMathMode",
        )?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        check(
            unsafe {
                (self.symbols.sgemm)(
                    self.handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    m,
                    1,
                    k,
                    &alpha,
                    weight,
                    k,
                    input,
                    k,
                    &beta,
                    output,
                    m,
                )
            },
            "cublasSgemm_v2",
        )
    }
}

impl Drop for CublasExact {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.symbols.destroy)(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

fn check(status: c_int, operation: &str) -> Result<(), Box<dyn Error>> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{operation} failed with cublasStatus={status}").into())
    }
}

fn dl_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dynamic-loader error".into()
    } else {
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

fn resolved_symbol_path(symbol: *const c_void) -> Result<PathBuf, Box<dyn Error>> {
    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    if unsafe { dladdr(symbol, &mut info) } == 0 || info.dli_fname.is_null() {
        return Err(format!("dladdr cuBLAS failed: {}", dl_error()).into());
    }
    let path = unsafe { CStr::from_ptr(info.dli_fname) }
        .to_str()
        .map(PathBuf::from)?;
    Ok(path.canonicalize()?)
}

fn resolved_maps_path(requested: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let mut paths = std::fs::read_to_string("/proc/self/maps")?
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(PathBuf::from)
        .filter(|path| path.file_name() == requested.file_name())
        .map(|path| path.canonicalize())
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    paths.dedup();
    if paths.len() != 1 {
        return Err(format!("expected one mapped cuBLAS object, got {paths:?}").into());
    }
    Ok(paths.remove(0))
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let output = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let digest = std::str::from_utf8(&output.stdout)?
        .split_whitespace()
        .next()
        .ok_or("sha256sum returned no digest")?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "sha256sum returned an invalid digest for {}",
            path.display()
        )
        .into());
    }
    Ok(digest.to_ascii_lowercase())
}

fn require_absolute_path(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.is_absolute() {
        Ok(())
    } else {
        Err("COLIBRI_CUBLAS_EXACT_LIB must be an absolute path".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_library_path_rejects_relative_value() {
        assert!(require_absolute_path(Path::new("libcublas.so.12")).is_err());
    }
}
