//! Default cache for the prequantized ConvRot DiT weights.
//!
//! `--convrot` used to rotate + INT8-quantize every ConvRot linear at load
//! (231 of them: a bf16 fold GEMM, abs, max and quantize kernel each) after
//! reading the ~15 GB bf16 transformer. The prequantized file
//! (`prequantize-convrot`: `<prefix>.weight_i8` U8, `<prefix>.col_scale` f32
//! and the bf16 rest, ~6.8 GB) loads bit-identically with none of that work. This
//! module makes that file the default: it is built once per transformer
//! location into a cache dir, validated on every run, and rebuilt when the
//! source or the precision policy changes.
//!
//! Layout: `<root>/<snapshot>-<hash of the canonical transformer dir>/`
//! `transformer_convrot.safetensors`. Root: `--convrot-cache DIR` >
//! `$QIR_CONVROT_CACHE` > `$XDG_CACHE_HOME/qwen-image-rs/convrot` >
//! `$HOME/.cache/qwen-image-rs/convrot`.
//!
//! Validity: the file's safetensors `__metadata__` holds `qir.policy`
//! (`dit::convrot_policy_tag`) and `qir.source` (FNV-1a of the canonical dir
//! and each source file's name, size and mtime); both must equal the current
//! values and the header's data must end exactly at the file length (a torn
//! write fails that). Anything else is a rebuild, in place.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::Result;

/// File name of a cache entry (and of `scripts/convert.sh`'s output).
pub const CACHE_FILE: &str = "transformer_convrot.safetensors";
/// Metadata key: the precision-policy tag the file was built under.
pub const META_POLICY: &str = "qir.policy";
/// Metadata key: the source fingerprint the file was built from.
pub const META_SOURCE: &str = "qir.source";
/// Largest safetensors header we will read (the DiT's is ~60 KB).
const MAX_HEADER: u64 = 100 << 20;
/// A `*.tmp.<pid>` older than this is a dead builder's leftover.
const STALE_TMP: std::time::Duration = std::time::Duration::from_secs(3600);

/// What `--convrot` does about the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Load the cached file; build it first when missing or stale (default).
    Use,
    /// Ignore the cache: rotate + quantize on load (the pre-cache path).
    Off,
    /// Rebuild the cache entry unconditionally, then load it.
    Rebuild,
}

/// The cache settings of one run (value object).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheSpec {
    /// Cache root; `None` when no flag, env var or home dir names one.
    pub root: Option<PathBuf>,
    pub mode: CacheMode,
}

impl CacheSpec {
    /// From the CLI flags and the process environment.
    pub fn from_cli(dir: Option<&Path>, off: bool, rebuild: bool) -> Self {
        let root = cache_root(
            dir,
            std::env::var_os("QIR_CONVROT_CACHE"),
            std::env::var_os("XDG_CACHE_HOME"),
            std::env::var_os("HOME"),
        );
        let mode = if off {
            CacheMode::Off
        } else if rebuild {
            CacheMode::Rebuild
        } else {
            CacheMode::Use
        };
        Self { root, mode }
    }
}

/// Cache root by precedence: flag > `$QIR_CONVROT_CACHE` >
/// `$XDG_CACHE_HOME/qwen-image-rs/convrot` > `$HOME/.cache/qwen-image-rs/convrot`.
/// Empty env values count as unset.
pub fn cache_root(
    flag: Option<&Path>,
    env: Option<OsString>,
    xdg: Option<OsString>,
    home: Option<OsString>,
) -> Option<PathBuf> {
    let set = |v: Option<OsString>| v.filter(|s| !s.is_empty()).map(PathBuf::from);
    if let Some(f) = flag {
        return Some(f.to_path_buf());
    }
    if let Some(e) = set(env) {
        return Some(e);
    }
    let sub = Path::new("qwen-image-rs").join("convrot");
    if let Some(x) = set(xdg) {
        return Some(x.join(sub));
    }
    set(home).map(|h| h.join(".cache").join(sub))
}

/// FNV-1a 64 — stable across Rust releases (std's `DefaultHasher` is not).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The cache entry file for a transformer dir:
/// `<root>/<snapshot>-<fnv of the canonical dir>/transformer_convrot.safetensors`.
/// `<snapshot>` is the dir's parent name when the dir is called `transformer`
/// (an HF snapshot: the commit hash), else the dir's own name, reduced to
/// `[A-Za-z0-9._-]`.
pub fn entry_path(root: &Path, transformer_dir: &Path) -> Result<PathBuf> {
    let canon = transformer_dir
        .canonicalize()
        .with_context(|| format!("canonicalizing {}", transformer_dir.display()))?;
    let name_of = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    let own = name_of(&canon).unwrap_or_default();
    let raw = if own == "transformer" {
        canon.parent().and_then(name_of).unwrap_or(own)
    } else {
        own
    };
    let mut name: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if name.is_empty() || name.chars().all(|c| c == '.') {
        name = "dit".into();
    }
    let h = fnv1a64(canon.as_os_str().as_encoded_bytes());
    Ok(root.join(format!("{name}-{h:016x}")).join(CACHE_FILE))
}

/// Fingerprint of a transformer's weight files: FNV-1a over the canonical dir
/// and, per file (sorted), its name, byte size and mtime (ns). A content
/// change that keeps size AND mtime is not seen (`--rebuild-convrot-cache`).
pub fn source_fingerprint(dir: &Path, files: &[PathBuf]) -> Result<String> {
    let mut key = dir
        .canonicalize()
        .with_context(|| format!("canonicalizing {}", dir.display()))?
        .as_os_str()
        .as_encoded_bytes()
        .to_vec();
    let mut files = files.to_vec();
    files.sort();
    for f in &files {
        let md = std::fs::metadata(f).with_context(|| format!("stat {}", f.display()))?;
        let mtime = md
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = f.file_name().map(|n| n.as_encoded_bytes()).unwrap_or(&[]);
        key.push(0);
        key.extend_from_slice(name);
        key.extend_from_slice(format!("|{}|{mtime}", md.len()).as_bytes());
    }
    Ok(format!("{:016x}", fnv1a64(&key)))
}

/// The parsed safetensors header of one file.
#[derive(Debug)]
pub struct Header {
    pub metadata: HashMap<String, String>,
    pub names: Vec<String>,
    /// File length the header implies: 8 + header length + data end.
    pub expected_len: u64,
}

/// Read and parse a safetensors header without touching the data. Errors on
/// a truncated file, a header longer than `MAX_HEADER` or the file, or a
/// header that is not the safetensors JSON shape.
pub fn read_header(path: &Path) -> Result<Header> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_len = f.metadata()?.len();
    let mut n8 = [0u8; 8];
    f.read_exact(&mut n8)
        .with_context(|| format!("{}: shorter than a safetensors header", path.display()))?;
    let n = u64::from_le_bytes(n8);
    anyhow::ensure!(
        n <= MAX_HEADER && n <= file_len.saturating_sub(8),
        "{}: header length {n} exceeds the file or the {MAX_HEADER} B cap",
        path.display()
    );
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf)?;
    let json: serde_json::Value = serde_json::from_slice(&buf)
        .with_context(|| format!("{}: header is not JSON", path.display()))?;
    let obj = json
        .as_object()
        .with_context(|| format!("{}: header is not a JSON object", path.display()))?;
    let mut metadata = HashMap::new();
    let mut names = Vec::new();
    let mut data_end = 0u64;
    for (k, v) in obj {
        if k == "__metadata__" {
            for (mk, mv) in v.as_object().into_iter().flatten() {
                if let Some(s) = mv.as_str() {
                    metadata.insert(mk.clone(), s.to_string());
                }
            }
            continue;
        }
        let end = v["data_offsets"][1]
            .as_u64()
            .with_context(|| format!("{}: tensor {k} has no data_offsets", path.display()))?;
        data_end = data_end.max(end);
        names.push(k.clone());
    }
    Ok(Header {
        metadata,
        names,
        expected_len: 8 + n + data_end,
    })
}

/// True when any file already holds a ConvRot INT8 weight (`*.weight_i8`):
/// an explicit `prequantize-convrot` output, used as-is.
pub fn files_hold_prequant(files: &[PathBuf]) -> Result<bool> {
    for f in files {
        if read_header(f)?
            .names
            .iter()
            .any(|n| n.ends_with(".weight_i8"))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Why a cache entry cannot be used (`None` from `check_entry` = usable).
pub fn check_entry(path: &Path, policy: &str, source: &str) -> Option<String> {
    if !path.exists() {
        return Some("missing".into());
    }
    let h = match read_header(path) {
        Ok(h) => h,
        Err(e) => return Some(format!("unreadable header: {e:#}")),
    };
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if h.expected_len != len {
        return Some(format!(
            "torn file: header implies {} B, file is {len} B",
            h.expected_len
        ));
    }
    if h.metadata.get(META_POLICY).map(String::as_str) != Some(policy) {
        return Some("precision policy changed".into());
    }
    if h.metadata.get(META_SOURCE).map(String::as_str) != Some(source) {
        return Some("source weights changed".into());
    }
    None
}

/// Write `out` atomically: `write` fills `<out>.tmp.<pid>`, which is fsynced
/// and renamed over `out`. On any error the tmp is removed and `out` is
/// untouched. Stale tmps of dead builders are swept first.
pub fn write_atomic(out: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let dir = out.parent().context("cache file has no parent dir")?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    sweep_stale_tmps(out);
    let name = out.file_name().context("cache file has no name")?;
    let mut tmp_name = name.to_os_string();
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = dir.join(tmp_name);
    let res = (|| -> Result<()> {
        write(&tmp)?;
        std::fs::File::open(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, out)
            .with_context(|| format!("renaming {} -> {}", tmp.display(), out.display()))?;
        // Persist the rename itself (directory entry).
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
        Ok(())
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Remove `<out>.tmp.*` files older than `STALE_TMP` (a killed builder's).
fn sweep_stale_tmps(out: &Path) {
    let (Some(dir), Some(name)) = (out.parent(), out.file_name()) else {
        return;
    };
    let prefix = format!("{}.tmp.", name.to_string_lossy());
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let fname = e.file_name();
        if !fname.to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let old = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > STALE_TMP);
        if old {
            tracing::info!(path = %e.path().display(), "removing stale convrot cache tmp");
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// The weight files to mmap for the DiT at `dir` (its `*.safetensors`).
/// Under `convrot` with the cache on, that is the validated (or freshly built)
/// cache entry; on any cache failure it warns and returns the source files,
/// which `QwenImageDit::load` rotates + quantizes on load — identical output.
pub fn resolve_dit_files(dir: &Path, convrot: bool, spec: &CacheSpec) -> Result<Vec<PathBuf>> {
    let files = crate::loader::WeightSet::resolve(dir)?.files;
    if !convrot || spec.mode == CacheMode::Off {
        return Ok(files);
    }
    if files_hold_prequant(&files)? {
        tracing::info!(dir = %dir.display(), "DiT dir is already prequantized ConvRot; cache not used");
        return Ok(files);
    }
    let Some(root) = spec.root.as_deref() else {
        tracing::warn!("no convrot cache dir (no --convrot-cache, QIR_CONVROT_CACHE, XDG_CACHE_HOME or HOME): quantizing on load");
        return Ok(files);
    };
    match resolve_entry(dir, &files, root, spec.mode) {
        Ok(entry) => Ok(vec![entry]),
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "convrot cache unavailable: quantizing on load (same output)");
            Ok(files)
        }
    }
}

fn resolve_entry(dir: &Path, files: &[PathBuf], root: &Path, mode: CacheMode) -> Result<PathBuf> {
    let entry = entry_path(root, dir)?;
    let policy = crate::model::dit::convrot_policy_tag();
    let source = source_fingerprint(dir, files)?;
    let reason = if mode == CacheMode::Rebuild {
        Some("--rebuild-convrot-cache".to_string())
    } else {
        check_entry(&entry, &policy, &source)
    };
    let Some(reason) = reason else {
        tracing::info!(path = %entry.display(), "convrot cache hit");
        return Ok(entry);
    };
    tracing::info!(path = %entry.display(), reason, "building the convrot cache (one-time: rotate + INT8-quantize the DiT linears)");
    let t0 = std::time::Instant::now();
    let meta = HashMap::from([
        (META_POLICY.to_string(), policy.clone()),
        (META_SOURCE.to_string(), source.clone()),
    ]);
    let n = build(files, &entry, meta)?;
    // Re-check what we just wrote before trusting it.
    if let Some(why) = check_entry(&entry, &policy, &source) {
        anyhow::bail!("freshly built cache entry fails validation: {why}");
    }
    tracing::info!(
        path = %entry.display(),
        convrot_linears = n,
        bytes = std::fs::metadata(&entry).map(|m| m.len()).unwrap_or(0),
        elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f64()),
        "convrot cache built"
    );
    Ok(entry)
}

/// Build a prequantized ConvRot file at `out` from the bf16 weight `files`:
/// each `dit::is_convrot_target` weight becomes `<prefix>.weight_i8` (rotated
/// INT8 as U8) + `<prefix>.col_scale` (f32) via `ConvRotLinear::from_weight`
/// (the on-load math, so the file loads bit-identically); every other tensor
/// is copied as stored. Streams the sources through an mmap; written
/// atomically with `metadata` in the header. Returns the ConvRot linear count.
#[cfg(feature = "convrot")]
pub fn build(files: &[PathBuf], out: &Path, metadata: HashMap<String, String>) -> Result<usize> {
    use crate::convrot::ConvRotLinear;
    use candle_core::{DType, Device, Tensor};

    let dev = crate::device::best_device()?;
    anyhow::ensure!(
        matches!(dev, Device::Cuda(_)),
        "building ConvRot weights needs CUDA (the quantize kernel is CUDA-only)"
    );
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(files)? };
    let r = crate::model::rotation::regular_hadamard_256(&dev)?;
    let mut names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    names.sort();
    names.dedup();
    let mut outv: Vec<(String, Tensor)> = Vec::with_capacity(names.len() + 300);
    let mut n_quant = 0usize;
    for name in &names {
        let t = st.load(name, &Device::Cpu)?;
        let base = name
            .strip_suffix(".weight")
            .filter(|b| crate::model::dit::is_convrot_target(b));
        if let Some(base) = base {
            let w = t.to_device(&dev)?.to_dtype(DType::BF16)?; // (N,K) bf16
            let cr = ConvRotLinear::from_weight(&w, &r)?;
            let (wi8, cs) = cr.export();
            outv.push((format!("{base}.weight_i8"), wi8.to_device(&Device::Cpu)?));
            outv.push((format!("{base}.col_scale"), cs.to_device(&Device::Cpu)?));
            n_quant += 1;
        } else {
            outv.push((name.clone(), t));
        }
    }
    write_atomic(out, |tmp| {
        safetensors::serialize_to_file(outv.iter().map(|(k, v)| (k, v)), Some(metadata), tmp)
            .with_context(|| format!("writing {}", tmp.display()))?;
        Ok(())
    })?;
    Ok(n_quant)
}

#[cfg(not(feature = "convrot"))]
pub fn build(_: &[PathBuf], _: &Path, _: HashMap<String, String>) -> Result<usize> {
    anyhow::bail!("building ConvRot weights requires the `convrot` feature")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// A fresh, empty temp dir per test.
    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("qir-cc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_st(path: &Path, names: &[&str], meta: Option<HashMap<String, String>>) {
        let t = Tensor::zeros(4, candle_core::DType::F32, &Device::Cpu).unwrap();
        safetensors::serialize_to_file(names.iter().map(|n| (*n, &t)), meta, path).unwrap();
    }

    fn meta(policy: &str, source: &str) -> Option<HashMap<String, String>> {
        Some(HashMap::from([
            (META_POLICY.to_string(), policy.to_string()),
            (META_SOURCE.to_string(), source.to_string()),
        ]))
    }

    #[test]
    fn fnv_matches_the_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn cache_root_precedence() {
        let os = |s: &str| Some(OsString::from(s));
        let flag = Path::new("/f");
        assert_eq!(
            cache_root(Some(flag), os("/e"), os("/x"), os("/h")),
            Some(PathBuf::from("/f"))
        );
        assert_eq!(
            cache_root(None, os("/e"), os("/x"), os("/h")),
            Some(PathBuf::from("/e"))
        );
        assert_eq!(
            cache_root(None, os(""), os("/x"), os("/h")),
            Some(PathBuf::from("/x/qwen-image-rs/convrot"))
        );
        assert_eq!(
            cache_root(None, None, os(""), os("/h")),
            Some(PathBuf::from("/h/.cache/qwen-image-rs/convrot"))
        );
        assert_eq!(cache_root(None, None, None, os("")), None);
    }

    #[test]
    fn entry_path_names_the_snapshot_and_hashes_the_location() {
        let d = tmpdir("entry");
        let a = d.join("b3179ad3 snap").join("transformer");
        let b = d.join("other").join("transformer");
        let c = d.join("my dit");
        for p in [&a, &b, &c] {
            std::fs::create_dir_all(p).unwrap();
        }
        let root = Path::new("/cache");
        let ea = entry_path(root, &a).unwrap();
        let eb = entry_path(root, &b).unwrap();
        let ec = entry_path(root, &c).unwrap();
        let dname = |e: &Path| {
            e.parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        };
        assert!(dname(&ea).starts_with("b3179ad3_snap-"), "{}", ea.display());
        assert!(dname(&eb).starts_with("other-"));
        assert!(dname(&ec).starts_with("my_dit-"));
        assert_ne!(ea.parent(), eb.parent());
        assert_eq!(ea.file_name().unwrap(), CACHE_FILE);
        // Same location through a different spelling -> same entry.
        let a2 = a.join("..").join("transformer");
        assert_eq!(entry_path(root, &a2).unwrap(), ea);
        assert!(entry_path(root, &d.join("missing")).is_err());
    }

    #[test]
    fn fingerprint_tracks_size_and_name() {
        let d = tmpdir("fp");
        let f1 = d.join("a.safetensors");
        write_st(&f1, &["x"], None);
        let fp1 = source_fingerprint(&d, std::slice::from_ref(&f1)).unwrap();
        assert_eq!(
            fp1,
            source_fingerprint(&d, std::slice::from_ref(&f1)).unwrap()
        );
        write_st(&f1, &["x", "y"], None); // bigger file
        assert_ne!(
            fp1,
            source_fingerprint(&d, std::slice::from_ref(&f1)).unwrap()
        );
        let f2 = d.join("b.safetensors");
        write_st(&f2, &["x"], None);
        let two = source_fingerprint(&d, &[f2.clone(), f1.clone()]).unwrap();
        assert_eq!(
            two,
            source_fingerprint(&d, &[f1, f2]).unwrap(),
            "order-free"
        );
    }

    #[test]
    fn header_roundtrip_and_prequant_detection() {
        let d = tmpdir("hdr");
        let f = d.join("t.safetensors");
        write_st(&f, &["a.weight_i8", "a.col_scale"], meta("P", "S"));
        let h = read_header(&f).unwrap();
        assert_eq!(h.metadata.get(META_POLICY).unwrap(), "P");
        assert_eq!(h.expected_len, std::fs::metadata(&f).unwrap().len());
        assert!(files_hold_prequant(std::slice::from_ref(&f)).unwrap());
        let g = d.join("bf16.safetensors");
        write_st(&g, &["a.weight"], None);
        assert!(!files_hold_prequant(&[g]).unwrap());
    }

    #[test]
    fn check_entry_rejects_every_mismatch() {
        let d = tmpdir("chk");
        let f = d.join(CACHE_FILE);
        assert_eq!(check_entry(&f, "P", "S").as_deref(), Some("missing"));
        write_st(&f, &["w"], meta("P", "S"));
        assert_eq!(check_entry(&f, "P", "S"), None);
        assert!(check_entry(&f, "P2", "S").unwrap().contains("policy"));
        assert!(check_entry(&f, "P", "S2").unwrap().contains("source"));
        write_st(&f, &["w"], None);
        assert!(check_entry(&f, "P", "S").unwrap().contains("policy"));
        // Torn: valid header, data cut short.
        write_st(&f, &["w"], meta("P", "S"));
        let len = std::fs::metadata(&f).unwrap().len();
        let file = std::fs::OpenOptions::new().write(true).open(&f).unwrap();
        file.set_len(len - 4).unwrap();
        assert!(check_entry(&f, "P", "S").unwrap().contains("torn"));
    }

    #[test]
    fn malformed_headers_are_errors_not_panics() {
        let d = tmpdir("bad");
        let f = d.join("x.safetensors");
        std::fs::write(&f, [1u8, 2, 3]).unwrap(); // shorter than 8 bytes
        assert!(read_header(&f).is_err());
        let mut huge = u64::MAX.to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        std::fs::write(&f, &huge).unwrap(); // claims an absurd header length
        assert!(read_header(&f).is_err());
        let mut junk = 5u64.to_le_bytes().to_vec();
        junk.extend_from_slice(b"nope!");
        std::fs::write(&f, &junk).unwrap(); // not JSON
        assert!(read_header(&f).is_err());
        let mut arr = 2u64.to_le_bytes().to_vec();
        arr.extend_from_slice(b"[]");
        std::fs::write(&f, &arr).unwrap(); // JSON, wrong shape
        assert!(read_header(&f).is_err());
        assert!(check_entry(&f, "P", "S").unwrap().contains("unreadable"));
    }

    #[test]
    fn write_atomic_leaves_no_tmp_and_keeps_the_old_file_on_error() {
        let d = tmpdir("atomic");
        let out = d.join("sub").join(CACHE_FILE);
        write_atomic(&out, |t| Ok(std::fs::write(t, b"one")?)).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"one");
        let r = write_atomic(&out, |t| {
            std::fs::write(t, b"partial")?;
            anyhow::bail!("simulated failure")
        });
        assert!(r.is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"one", "old file untouched");
        let left: Vec<_> = std::fs::read_dir(out.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(left, vec![OsString::from(CACHE_FILE)], "no tmp left");
    }

    #[test]
    fn resolve_without_convrot_or_with_cache_off_returns_the_sources() {
        let d = tmpdir("resolve");
        let f = d.join("m.safetensors");
        write_st(&f, &["a.weight"], None);
        let root = d.join("cache");
        let on = CacheSpec {
            root: Some(root.clone()),
            mode: CacheMode::Use,
        };
        assert_eq!(resolve_dit_files(&d, false, &on).unwrap(), vec![f.clone()]);
        let off = CacheSpec {
            root: Some(root.clone()),
            mode: CacheMode::Off,
        };
        assert_eq!(resolve_dit_files(&d, true, &off).unwrap(), vec![f.clone()]);
        // An already-prequantized dir is used as-is.
        let p = tmpdir("resolve-pq");
        let pf = p.join("pq.safetensors");
        write_st(&pf, &["a.weight_i8"], None);
        assert_eq!(resolve_dit_files(&p, true, &on).unwrap(), vec![pf]);
        assert!(!root.exists(), "no cache entry created for these cases");
    }

    #[test]
    fn a_hit_is_served_without_building() {
        let d = tmpdir("hit");
        let src = d.join("snap").join("transformer");
        std::fs::create_dir_all(&src).unwrap();
        let f = src.join("m.safetensors");
        write_st(&f, &["a.weight"], None);
        let root = d.join("cache");
        let entry = entry_path(&root, &src).unwrap();
        std::fs::create_dir_all(entry.parent().unwrap()).unwrap();
        let source = source_fingerprint(&src, std::slice::from_ref(&f)).unwrap();
        let policy = crate::model::dit::convrot_policy_tag();
        write_st(&entry, &["a.weight_i8"], meta(&policy, &source));
        let spec = CacheSpec {
            root: Some(root),
            mode: CacheMode::Use,
        };
        assert_eq!(resolve_dit_files(&src, true, &spec).unwrap(), vec![entry]);
    }

    #[test]
    fn a_stale_entry_without_convrot_falls_back_to_the_sources() {
        // On the CPU (or a build without `convrot`) the rebuild fails; the run
        // must still get the source files, never an error or the stale file.
        if cfg!(feature = "convrot") {
            return; // would try a real GPU build
        }
        let d = tmpdir("stale");
        let src = d.join("snap").join("transformer");
        std::fs::create_dir_all(&src).unwrap();
        let f = src.join("m.safetensors");
        write_st(&f, &["a.weight"], None);
        let root = d.join("cache");
        let entry = entry_path(&root, &src).unwrap();
        std::fs::create_dir_all(entry.parent().unwrap()).unwrap();
        write_st(&entry, &["a.weight_i8"], meta("old-policy", "x"));
        let spec = CacheSpec {
            root: Some(root),
            mode: CacheMode::Use,
        };
        assert_eq!(resolve_dit_files(&src, true, &spec).unwrap(), vec![f]);
    }
}
