use super::*;

#[derive(Clone, Copy)]
pub(crate) struct LinuxMemorySnapshot {
    pub(crate) available_bytes: u64,
    pub(crate) cached_bytes: u64,
}

pub(crate) fn linux_memory_snapshot() -> Result<LinuxMemorySnapshot, Box<dyn Error>> {
    let contents = std::fs::read_to_string("/proc/meminfo")?;
    let value = |name: &str| -> Result<u64, Box<dyn Error>> {
        let kib = contents
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|tail| tail.split_whitespace().next())
            .ok_or_else(|| format!("{name} missing from /proc/meminfo"))?
            .parse::<u64>()?;
        kib.checked_mul(1_024)
            .ok_or_else(|| format!("{name} byte count overflow").into())
    };
    Ok(LinuxMemorySnapshot {
        available_bytes: value("MemAvailable:")?,
        cached_bytes: value("Cached:")?,
    })
}

pub(crate) fn model_mount_provenance(
    model_dir: &std::path::Path,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let canonical = model_dir.canonicalize()?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo")?;
    let (_, source, fstype) = mountinfo
        .lines()
        .filter_map(|line| {
            let (mount, filesystem) = line.split_once(" - ")?;
            let target = mount.split_whitespace().nth(4)?;
            let mut filesystem = filesystem.split_whitespace();
            let fstype = filesystem.next()?;
            let source = filesystem.next()?;
            canonical
                .starts_with(target)
                .then_some((target.len(), source, fstype))
        })
        .max_by_key(|(length, _, _)| *length)
        .ok_or("model mount provenance is unavailable")?;
    let source_canonical = std::fs::canonicalize(source)?;
    let uuid = std::fs::read_dir("/dev/disk/by-uuid")?
        .filter_map(Result::ok)
        .find_map(|entry| {
            std::fs::canonicalize(entry.path())
                .ok()
                .filter(|candidate| candidate == &source_canonical)
                .and_then(|_| entry.file_name().into_string().ok())
        })
        .ok_or("model block device UUID is unavailable")?;
    Ok(serde_json::json!({
        "source": source,
        "fstype": fstype,
        "uuid": uuid,
    }))
}

pub(crate) struct CacheOffSource {
    pub(crate) layer: usize,
    pub(crate) shard: safetensors::ShardFile,
    pub(crate) raw_name: String,
    pub(crate) raw: Vec<u8>,
    pub(crate) plan: v4_expert::RoutedExpertRangePlan,
    pub(crate) routed: [Vec<u8>; 6],
}

impl CacheOffSource {
    pub(crate) fn new(model_dir: &std::path::Path, layer: usize) -> Result<Self, Box<dyn Error>> {
        let shard = safetensors::ShardFile::open(
            model_dir.join(format!("model-{:05}-of-00064.safetensors", layer + 2)),
        )?;
        let raw_name = format!("layers.{layer}.attn.wo_a.weight");
        let raw = vec![0_u8; usize::try_from(shard.tensor(&raw_name)?.byte_len)?];
        let plan = v4_expert::RoutedExpertRangePlan::new(&shard, layer, 0)?;
        let routed = [
            vec![0_u8; v4_expert::V4_W1_BYTES],
            vec![0_u8; v4_expert::V4_W1_SCALE_BYTES],
            vec![0_u8; v4_expert::V4_W2_BYTES],
            vec![0_u8; v4_expert::V4_W2_SCALE_BYTES],
            vec![0_u8; v4_expert::V4_W1_BYTES],
            vec![0_u8; v4_expert::V4_W1_SCALE_BYTES],
        ];
        Ok(Self {
            layer,
            shard,
            raw_name,
            raw,
            plan,
            routed,
        })
    }

    pub(crate) fn evict(&self) -> Result<(), Box<dyn Error>> {
        self.shard
            .advise_range_dontneed(&self.raw_name, 0, self.raw.len())?;
        self.plan.advise_dontneed(&self.shard)
    }

    pub(crate) fn read(
        &mut self,
        mode: safetensors::RangeReadMode,
    ) -> Result<safetensors::RangeReadTelemetry, Box<dyn Error>> {
        let raw = self
            .shard
            .read_at_mode(&self.raw_name, 0, &mut self.raw, mode)?;
        let [w1, w1_scale, w2, w2_scale, w3, w3_scale] = &mut self.routed;
        let routed = self.plan.read_into_mode(
            &self.shard,
            [
                w1.as_mut_slice(),
                w1_scale.as_mut_slice(),
                w2.as_mut_slice(),
                w2_scale.as_mut_slice(),
                w3.as_mut_slice(),
                w3_scale.as_mut_slice(),
            ],
            mode,
        )?;
        Ok(safetensors::RangeReadTelemetry {
            logical_bytes: raw
                .logical_bytes
                .checked_add(routed.logical_bytes)
                .ok_or("cache-off logical byte count overflow")?,
            physical_bytes: raw
                .physical_bytes
                .checked_add(routed.physical_bytes)
                .ok_or("cache-off physical byte count overflow")?,
        })
    }

    pub(crate) fn update_digest(&self, mut hash: u64) -> u64 {
        for byte in self.raw.iter().chain(self.routed.iter().flatten()) {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
        hash
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CacheOffSample {
    pub(crate) read_wall_ns: u64,
    pub(crate) logical_bytes: usize,
    pub(crate) physical_bytes: usize,
    pub(crate) digest: u64,
    pub(crate) available_delta_bytes: i64,
    pub(crate) cached_delta_bytes: i64,
}

pub(crate) fn run_cache_off_arm(
    sources: &mut [CacheOffSource],
    mode: safetensors::RangeReadMode,
) -> Result<CacheOffSample, Box<dyn Error>> {
    for source in sources.iter() {
        source.evict()?;
    }
    let before = linux_memory_snapshot()?;
    let started = std::time::Instant::now();
    let mut logical_bytes = 0_usize;
    let mut physical_bytes = 0_usize;
    for source in sources.iter_mut() {
        let read = source.read(mode)?;
        logical_bytes = logical_bytes
            .checked_add(read.logical_bytes)
            .ok_or("cache-off logical byte total overflow")?;
        physical_bytes = physical_bytes
            .checked_add(read.physical_bytes)
            .ok_or("cache-off physical byte total overflow")?;
    }
    let read_wall_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let after = linux_memory_snapshot()?;
    let digest = sources
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, source| {
            source.update_digest(hash)
        });
    Ok(CacheOffSample {
        read_wall_ns,
        logical_bytes,
        physical_bytes,
        digest,
        available_delta_bytes: i64::try_from(after.available_bytes)?
            - i64::try_from(before.available_bytes)?,
        cached_delta_bytes: i64::try_from(after.cached_bytes)?
            - i64::try_from(before.cached_bytes)?,
    })
}

pub(crate) fn median_i64(samples: &[i64]) -> i64 {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] as i128 + samples[middle] as i128) as i64 / 2
    } else {
        samples[middle]
    }
}
