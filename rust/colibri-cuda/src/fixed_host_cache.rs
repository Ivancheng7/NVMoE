use super::*;

pub(crate) const FIXED_HOST_RESIDENT_GIB_ENV: &str = "COLIBRI_V4_FIXED_HOST_RESIDENT_GIB";

/// Budget used when the env var is absent. 8 GiB is the configuration the
/// residency A/B measured, and it holds the whole 5.6 GiB fixed model, so the
/// default is full residency. `=0` turns the cache off; any other whole number
/// keeps that many GiB of layers resident as a prefix.
pub(crate) const FIXED_HOST_RESIDENT_DEFAULT_GIB: u64 = 8;

/// Keeps the decoded fixed host payload of a layer alive between tokens.
///
/// Without it every token rebuilds all 43 host payloads from the shards: 137 MB
/// allocated, faulted in and freed per layer, 5.9 GB per token, byte for byte
/// identical to the previous token. With a budget the payload is built once and
/// later tokens only re-upload it (H2D), which is all the layer still needs.
pub(crate) struct FixedHostCache {
    layers: Vec<Option<std::sync::Arc<ResidentLayerHost>>>,
    budget_bytes: usize,
    retained_bytes: usize,
}

impl FixedHostCache {
    pub(crate) fn from_env(layer_count: usize) -> Result<Self, Box<dyn Error>> {
        let budget_gib = match std::env::var_os(FIXED_HOST_RESIDENT_GIB_ENV) {
            None => FIXED_HOST_RESIDENT_DEFAULT_GIB,
            Some(raw) => raw
                .to_string_lossy()
                .trim()
                .parse::<u64>()
                .map_err(|_| format!("{FIXED_HOST_RESIDENT_GIB_ENV} must be a whole number of GiB"))?,
        };
        let budget_bytes = usize::try_from(budget_gib)
            .ok()
            .and_then(|gib| gib.checked_mul(1024 * 1024 * 1024))
            .ok_or("fixed host residency budget overflows the address space")?;
        Ok(Self {
            layers: vec![None; layer_count],
            budget_bytes,
            retained_bytes: 0,
        })
    }

    fn get(&self, layer: usize) -> Option<std::sync::Arc<ResidentLayerHost>> {
        self.layers
            .get(layer)
            .and_then(|slot| slot.as_ref())
            .cloned()
    }

    /// Retains the payload when there is room for it. Layers are visited in
    /// order, so a budget smaller than the model keeps a prefix resident and
    /// leaves the rest on the existing per-token path.
    fn insert(&mut self, layer: usize, host: &std::sync::Arc<ResidentLayerHost>, bytes: usize) {
        let within_budget = self
            .retained_bytes
            .checked_add(bytes)
            .is_some_and(|total| total <= self.budget_bytes);
        if self.budget_bytes == 0 || !within_budget || self.layers[layer].is_some() {
            return;
        }
        self.retained_bytes += bytes;
        self.layers[layer] = Some(host.clone());
    }
}

/// A retained fixed layer payload.
///
/// The raw `wo_a` sits next to the host instead of inside it, the same shape the
/// stationary pinned path uses: `LayerDeviceWeights::new_layer` refuses a host
/// that still carries raw `wo_a`, so the take has to happen once, at load time.
pub(crate) struct ResidentLayerHost {
    pub(crate) host: DecodedLayerHost,
    pub(crate) wo_a_raw: Vec<u8>,
}

impl ResidentLayerHost {
    pub(crate) fn bytes(&self) -> usize {
        self.host.num_bytes() + self.wo_a_raw.len()
    }
}

/// Returns the layer's decoded host payload: from the retention cache when it is
/// already there, from the shard otherwise.
pub(crate) fn resolve_host_layer(
    spec: &'static v4_spec::V4Spec,
    model_dir: &std::path::Path,
    layer: usize,
    cache: &std::sync::Mutex<FixedHostCache>,
) -> Result<(std::sync::Arc<ResidentLayerHost>, FixedLoadTelemetry), Box<dyn Error>> {
    if let Some(cached) = cache
        .lock()
        .map_err(|_| "fixed host cache lock was poisoned")?
        .get(layer)
    {
        return Ok((
            cached.clone(),
            FixedLoadTelemetry {
                // Nothing was read from the shard for this layer: the ledger has
                // to show the saving, so only the H2D side counts.
                fixed_source_bytes: 0,
                cache_status: "host_resident_hit",
                ..FixedLoadTelemetry::default()
            },
        ));
    }
    let (mut host, telemetry) = load_layer_host_inner_for(spec, model_dir, layer, &[], true)?;
    let wo_a_raw = host
        .tail
        .wo_a_raw
        .take()
        .ok_or("resident fixed host payload omitted raw wo_a")?;
    let resident = std::sync::Arc::new(ResidentLayerHost { host, wo_a_raw });
    let resident_bytes = resident.bytes();
    cache
        .lock()
        .map_err(|_| "fixed host cache lock was poisoned")?
        .insert(layer, &resident, resident_bytes);
    Ok((resident, telemetry))
}

/// The Flash B1 base decode loop's own one-layer-ahead prefetch task.
///
/// It carries an `Arc` because the payload now outlives the layer: the cache
/// holds one reference and the consumer another, so consuming a layer no longer
/// frees it. The stationary prefill path keeps its owned-payload
/// `FixedPrefetchTask`; this loop does not touch it.
pub(crate) struct ResidentPrefetch<'scope> {
    pub(crate) layer: usize,
    pub(crate) worker:
        std::thread::ScopedJoinHandle<'scope, Result<ResidentPrefetched, String>>,
}

pub(crate) struct ResidentPrefetched {
    pub(crate) layer: usize,
    pub(crate) host: std::sync::Arc<ResidentLayerHost>,
    pub(crate) telemetry: FixedLoadTelemetry,
    pub(crate) worker_start_ns: u64,
    pub(crate) worker_end_ns: u64,
}

