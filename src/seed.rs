//! `--seed <N|random>`: a fixed base seed (lane `i` uses `N + i`) or a fresh
//! random seed per image. Random seeds come from the std hasher's OS-seeded
//! keys (no extra dependency) and are always reported, so any image can be
//! reproduced later with `--seed <n>` (a single image) — see `write_manifest`.

use std::fmt;
use std::str::FromStr;

/// The `--seed` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seed {
    /// Base seed: image/lane `i` uses `base + i` (wrapping).
    Fixed(u64),
    /// Every image gets its own random seed.
    Random,
}

impl FromStr for Seed {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("random") || t.eq_ignore_ascii_case("rand") {
            return Ok(Seed::Random);
        }
        t.parse::<u64>()
            .map(Seed::Fixed)
            .map_err(|_| format!("--seed expects a non-negative integer or `random`, got `{s}`"))
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Seed::Fixed(n) => write!(f, "{n}"),
            Seed::Random => write!(f, "random"),
        }
    }
}

impl Seed {
    /// The concrete seed for each of `n` images.
    pub fn resolve(self, n: usize) -> Vec<u64> {
        match self {
            Seed::Fixed(base) => (0..n).map(|i| base.wrapping_add(i as u64)).collect(),
            Seed::Random => (0..n).map(|_| random_u64()).collect(),
        }
    }
}

/// A random u64 from the std `RandomState` (OS-seeded keys), mixed with the
/// clock and a process-wide counter so consecutive calls never repeat.
pub fn random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    h.write_u128(nanos);
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.finish()
}

/// Write `<dir>/seeds.tsv` (`file`, `seed`, `prompt`), one row per image, so a
/// batch with random seeds can be reproduced image by image.
pub fn write_manifest(
    dir: &std::path::Path,
    rows: &[(String, u64, String)],
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    let path = dir.join("seeds.tsv");
    let mut f = std::fs::File::create(&path)?;
    writeln!(f, "file\tseed\tprompt")?;
    for (file, seed, prompt) in rows {
        let p: String = prompt
            .chars()
            .map(|c| {
                if c == '\t' || c == '\n' || c == '\r' {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        writeln!(f, "{file}\t{seed}\t{p}")?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbers_and_random() {
        assert_eq!("42".parse::<Seed>(), Ok(Seed::Fixed(42)));
        assert_eq!(" 7 ".parse::<Seed>(), Ok(Seed::Fixed(7)));
        assert_eq!("random".parse::<Seed>(), Ok(Seed::Random));
        assert_eq!("RANDOM".parse::<Seed>(), Ok(Seed::Random));
        assert!("-1".parse::<Seed>().is_err());
        assert!("abc".parse::<Seed>().is_err());
        assert_eq!(Seed::Fixed(5).to_string(), "5");
        assert_eq!(Seed::Random.to_string(), "random");
    }

    #[test]
    fn fixed_lanes_are_consecutive_and_wrap() {
        assert_eq!(Seed::Fixed(42).resolve(3), vec![42, 43, 44]);
        assert_eq!(Seed::Fixed(u64::MAX).resolve(2), vec![u64::MAX, 0]);
    }

    #[test]
    fn random_lanes_differ() {
        let s = Seed::Random.resolve(64);
        let mut u = s.clone();
        u.sort_unstable();
        u.dedup();
        assert_eq!(u.len(), s.len(), "random seeds repeated");
        assert_ne!(Seed::Random.resolve(1), Seed::Random.resolve(1));
    }

    #[test]
    fn manifest_sanitizes_prompts() {
        let dir = std::env::temp_dir().join(format!("qir-seed-test-{}", random_u64()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_manifest(&dir, &[("000.png".into(), 9, "a\tb\nc".into())]).unwrap();
        let s = std::fs::read_to_string(&p).unwrap();
        assert_eq!(s, "file\tseed\tprompt\n000.png\t9\ta b c\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
